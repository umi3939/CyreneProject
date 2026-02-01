"""
psyche/tone.py - Humor / Light-Tone Decision Mode

Introduces a "Tone" attribute to decision candidates, allowing the system
to select responses with varying tones (neutral, light, serious) based on
psychological state.

Design principles (from design_humor_tone.md):
- Tone is a structural tag, NOT content generation
- Light tone is one option among equals, not special-cased
- No hardcoded conditions - uses weighted bias
- Tone is temporary (single-turn), not a persistent mode
- Compatible with silence (silence can have a tone)

Usage::

    from psyche.tone import (
        Tone,
        ToneConfig,
        ToneModifier,
        compute_tone_bias,
        apply_tone_to_candidate,
        generate_tone_variants,
        select_candidate_tone,
    )

    # Compute tone bias from state
    bias = compute_tone_bias(state, responsibility_influence)

    # Apply tone to a candidate
    toned = apply_tone_to_candidate(candidate, Tone.LIGHT, bias)

    # Generate candidates with tone variants
    variants = generate_tone_variants(base_candidates, state, config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import random

from .state import PsycheState, Percept
from .responsibility import ResponsibilityInfluence
from .decision_bias import DecisionBias


class Tone(Enum):
    """
    Tone categories for decision candidates.

    These are abstract categories - the actual expression
    is determined by the expression module, not here.
    """

    # Default, balanced tone
    NEUTRAL = "neutral"

    # Light, playful, humorous tone
    LIGHT = "light"

    # Serious, thoughtful, careful tone
    SERIOUS = "serious"

    # Warm, gentle, caring tone
    WARM = "warm"

    # Reserved, minimal tone
    RESERVED = "reserved"


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class ToneConfig:
    """
    Configuration for tone selection behavior.

    All parameters are externally configurable.
    No hardcoded conditions for when to use specific tones.
    """

    # Base weights for each tone (before state-based modification)
    base_weights: dict[str, float] = field(default_factory=lambda: {
        Tone.NEUTRAL.value: 1.0,
        Tone.LIGHT.value: 0.3,
        Tone.SERIOUS.value: 0.3,
        Tone.WARM.value: 0.4,
        Tone.RESERVED.value: 0.2,
    })

    # State influence factors
    # How much positive mood increases LIGHT tone weight
    positive_mood_light_factor: float = 1.5

    # How much negative mood increases SERIOUS tone weight
    negative_mood_serious_factor: float = 1.2

    # How much low fear increases LIGHT tone weight
    low_fear_light_factor: float = 1.3

    # How much high fear increases SERIOUS tone weight
    high_fear_serious_factor: float = 1.5

    # How much responsibility weight increases SERIOUS tone weight
    responsibility_serious_factor: float = 1.4

    # How much low responsibility increases LIGHT tone weight
    low_responsibility_light_factor: float = 1.2

    # Threshold for "low" fear/responsibility (below this = low)
    low_threshold: float = 0.2

    # Threshold for "high" fear/responsibility (above this = high)
    high_threshold: float = 0.5

    # Whether to allow LIGHT tone for silence
    allow_light_silence: bool = True

    # Maximum consecutive same-tone selections (0 = no limit)
    max_consecutive_same_tone: int = 3

    # Decay factor for consecutive same-tone (reduces weight)
    consecutive_decay: float = 0.7

    # Minimum weight (tones never go below this)
    minimum_weight: float = 0.1


@dataclass
class ToneModifier:
    """
    Computed tone weights based on current state.

    This is the interface between state and tone selection.
    Weights are multipliers on base weights.
    """

    # Per-tone weight multipliers
    weights: dict[str, float] = field(default_factory=lambda: {
        Tone.NEUTRAL.value: 1.0,
        Tone.LIGHT.value: 1.0,
        Tone.SERIOUS.value: 1.0,
        Tone.WARM.value: 1.0,
        Tone.RESERVED.value: 1.0,
    })

    # Recommended tone (highest weighted, for reference)
    recommended: Tone = Tone.NEUTRAL

    # State summary that influenced this modifier
    state_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class ToneState:
    """
    Tracks tone-related state across turns.

    Ensures tone variety and prevents getting stuck
    in one tone mode.
    """

    # Last selected tone
    last_tone: Optional[Tone] = None

    # Consecutive same-tone count
    consecutive_count: int = 0

    # Total tone selections (for statistics)
    tone_history: dict[str, int] = field(default_factory=lambda: {
        Tone.NEUTRAL.value: 0,
        Tone.LIGHT.value: 0,
        Tone.SERIOUS.value: 0,
        Tone.WARM.value: 0,
        Tone.RESERVED.value: 0,
    })

    def record_tone(self, tone: Tone) -> "ToneState":
        """Record a tone selection."""
        new_history = self.tone_history.copy()
        new_history[tone.value] = new_history.get(tone.value, 0) + 1

        if self.last_tone == tone:
            new_consecutive = self.consecutive_count + 1
        else:
            new_consecutive = 1

        return ToneState(
            last_tone=tone,
            consecutive_count=new_consecutive,
            tone_history=new_history,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "last_tone": self.last_tone.value if self.last_tone else None,
            "consecutive_count": self.consecutive_count,
            "tone_history": self.tone_history.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToneState":
        """Deserialize from dict."""
        last_tone = None
        if data.get("last_tone"):
            try:
                last_tone = Tone(data["last_tone"])
            except ValueError:
                pass

        return cls(
            last_tone=last_tone,
            consecutive_count=data.get("consecutive_count", 0),
            tone_history=data.get("tone_history", {}),
        )


# ── Core Computation ───────────────────────────────────────────────


def compute_tone_bias(
    state: PsycheState,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    config: Optional[ToneConfig] = None,
    tone_state: Optional[ToneState] = None,
) -> ToneModifier:
    """
    Compute tone weights based on psychological state.

    This function uses weighted bias, NOT hardcoded conditions.
    All influences are multiplicative on base weights.

    Args:
        state: Current psychological state (read-only)
        responsibility_influence: Responsibility influence (optional)
        decision_bias: Decision bias from STM/dynamics (optional)
        config: Tone configuration
        tone_state: Tone tracking state (optional)

    Returns:
        ToneModifier with computed weights
    """
    cfg = config or ToneConfig()
    ts = tone_state or ToneState()

    # Start with base weights
    weights = cfg.base_weights.copy()

    # Track state influences for debugging
    state_summary = {}

    # ── Mood-based influence ──

    mood_val = state.mood.valence
    state_summary["mood_valence"] = mood_val

    if mood_val > 0.2:
        # Positive mood → increase LIGHT weight
        factor = 1.0 + (mood_val * cfg.positive_mood_light_factor)
        weights[Tone.LIGHT.value] *= factor
        weights[Tone.WARM.value] *= (1.0 + mood_val * 0.5)
    elif mood_val < -0.2:
        # Negative mood → increase SERIOUS weight
        factor = 1.0 + (abs(mood_val) * cfg.negative_mood_serious_factor)
        weights[Tone.SERIOUS.value] *= factor
        weights[Tone.RESERVED.value] *= (1.0 + abs(mood_val) * 0.5)

    # ── Fear-based influence ──

    fear_level = state.fear_level
    state_summary["fear_level"] = fear_level

    if fear_level < cfg.low_threshold:
        # Low fear → increase LIGHT weight
        factor = cfg.low_fear_light_factor
        weights[Tone.LIGHT.value] *= factor
    elif fear_level > cfg.high_threshold:
        # High fear → increase SERIOUS weight
        factor = 1.0 + (fear_level * cfg.high_fear_serious_factor)
        weights[Tone.SERIOUS.value] *= factor
        weights[Tone.RESERVED.value] *= (1.0 + fear_level * 0.3)
        # Reduce LIGHT weight
        weights[Tone.LIGHT.value] *= 0.5

    # ── Arousal-based influence ──

    arousal = state.mood.arousal
    state_summary["arousal"] = arousal

    if arousal > 0.6:
        # High arousal → more expressive tones
        weights[Tone.LIGHT.value] *= 1.2
        weights[Tone.WARM.value] *= 1.1
    elif arousal < 0.3:
        # Low arousal → more reserved tones
        weights[Tone.RESERVED.value] *= 1.3
        weights[Tone.SERIOUS.value] *= 1.1

    # ── Emotion-based influence ──

    emotions = state.emotions.as_dict()
    joy = emotions.get("joy", 0)
    fun = emotions.get("fun", 0)
    sorrow = emotions.get("sorrow", 0)
    fear_emo = emotions.get("fear", 0)

    state_summary["joy"] = joy
    state_summary["fun"] = fun

    # High joy/fun → increase LIGHT weight
    if joy > 0.3 or fun > 0.3:
        playfulness = max(joy, fun)
        weights[Tone.LIGHT.value] *= (1.0 + playfulness)

    # High sorrow → increase WARM weight
    if sorrow > 0.3:
        weights[Tone.WARM.value] *= (1.0 + sorrow * 0.8)

    # ── Responsibility influence ──

    if responsibility_influence:
        anxiety = responsibility_influence.anxiety_baseline
        caution = responsibility_influence.caution_bias

        state_summary["responsibility_anxiety"] = anxiety
        state_summary["responsibility_caution"] = caution

        if anxiety < cfg.low_threshold and caution < cfg.low_threshold:
            # Low responsibility → increase LIGHT weight
            weights[Tone.LIGHT.value] *= cfg.low_responsibility_light_factor
        elif anxiety > cfg.high_threshold or caution > cfg.high_threshold:
            # High responsibility → increase SERIOUS weight
            total = anxiety + caution
            weights[Tone.SERIOUS.value] *= (1.0 + total * cfg.responsibility_serious_factor)
            # Reduce LIGHT weight
            weights[Tone.LIGHT.value] *= max(0.3, 1.0 - total)

    # ── Consecutive tone decay ──

    if ts.last_tone and ts.consecutive_count >= cfg.max_consecutive_same_tone:
        # Reduce weight for same tone
        last_tone_key = ts.last_tone.value
        weights[last_tone_key] *= (cfg.consecutive_decay ** ts.consecutive_count)

    # ── Ensure minimum weights ──

    for key in weights:
        weights[key] = max(cfg.minimum_weight, weights[key])

    # ── Determine recommended tone ──

    recommended = max(weights.keys(), key=lambda k: weights[k])

    return ToneModifier(
        weights={k: round(v, 3) for k, v in weights.items()},
        recommended=Tone(recommended),
        state_summary=state_summary,
    )


# ── Tone Application ───────────────────────────────────────────────


def apply_tone_to_candidate(
    candidate: dict[str, Any],
    tone: Tone,
    modifier: Optional[ToneModifier] = None,
) -> dict[str, Any]:
    """
    Apply a tone to a decision candidate.

    This adds tone metadata to the candidate without
    modifying the core decision content.

    Args:
        candidate: Policy candidate dict
        tone: Tone to apply
        modifier: Optional modifier (for weight info)

    Returns:
        New candidate dict with tone metadata
    """
    toned = candidate.copy()

    # Add tone metadata
    toned["_tone"] = tone.value
    toned["_tone_weight"] = modifier.weights.get(tone.value, 1.0) if modifier else 1.0

    return toned


def get_candidate_tone(candidate: dict[str, Any]) -> Tone:
    """Get the tone of a candidate (NEUTRAL if not set)."""
    tone_value = candidate.get("_tone", Tone.NEUTRAL.value)
    try:
        return Tone(tone_value)
    except ValueError:
        return Tone.NEUTRAL


def select_candidate_tone(
    candidate: dict[str, Any],
    modifier: ToneModifier,
    config: Optional[ToneConfig] = None,
    use_weighted_random: bool = True,
) -> Tone:
    """
    Select a tone for a candidate based on computed weights.

    Args:
        candidate: Policy candidate
        modifier: Computed tone modifier
        config: Tone configuration
        use_weighted_random: If True, use weighted random selection.
                            If False, always use highest weight.

    Returns:
        Selected Tone
    """
    cfg = config or ToneConfig()

    # Check if this is a silence candidate
    is_silence = candidate.get("_is_silence", False)

    # Filter available tones
    available_weights = modifier.weights.copy()

    # If silence and light tone not allowed, reduce LIGHT weight
    if is_silence and not cfg.allow_light_silence:
        available_weights[Tone.LIGHT.value] = cfg.minimum_weight

    if use_weighted_random:
        # Weighted random selection
        tones = list(available_weights.keys())
        weights = [available_weights[t] for t in tones]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            selected = random.choices(tones, weights=weights, k=1)[0]
            return Tone(selected)

    # Deterministic: highest weight
    return Tone(max(available_weights.keys(), key=lambda k: available_weights[k]))


# ── Candidate Generation with Tone ─────────────────────────────────


def generate_tone_variants(
    base_candidates: list[dict[str, Any]],
    state: PsycheState,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    decision_bias: Optional[DecisionBias] = None,
    config: Optional[ToneConfig] = None,
    tone_state: Optional[ToneState] = None,
    add_tone_variants: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate candidates with tone variants.

    This can either:
    1. Add tone to existing candidates (each gets one tone)
    2. Create tone variants (each candidate gets multiple toned versions)

    Args:
        base_candidates: Original policy candidates
        state: Current psychological state
        responsibility_influence: Optional responsibility influence
        decision_bias: Optional decision bias
        config: Tone configuration
        tone_state: Tone tracking state
        add_tone_variants: If True, create additional toned variants.
                          If False, just add tone to existing candidates.

    Returns:
        List of candidates with tone metadata
    """
    cfg = config or ToneConfig()

    # Compute tone bias
    modifier = compute_tone_bias(
        state=state,
        responsibility_influence=responsibility_influence,
        decision_bias=decision_bias,
        config=cfg,
        tone_state=tone_state,
    )

    result = []

    for candidate in base_candidates:
        if add_tone_variants:
            # Create multiple toned variants for each candidate
            # Only create variants for tones with significant weight
            for tone in Tone:
                weight = modifier.weights.get(tone.value, 0)
                if weight >= cfg.minimum_weight * 2:  # Only significant tones
                    toned = apply_tone_to_candidate(candidate, tone, modifier)
                    # Adjust score based on tone weight
                    original_score = candidate.get("_score", 0)
                    tone_bonus = (weight - 1.0) * 0.1  # Small bonus for high-weight tones
                    toned["_score"] = original_score + tone_bonus
                    result.append(toned)
        else:
            # Just select one tone for each candidate
            tone = select_candidate_tone(candidate, modifier, cfg)
            toned = apply_tone_to_candidate(candidate, tone, modifier)
            result.append(toned)

    # Sort by score
    result.sort(key=lambda c: c.get("_score", 0), reverse=True)

    return result


def add_tone_to_candidates(
    candidates: list[dict[str, Any]],
    state: PsycheState,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    config: Optional[ToneConfig] = None,
    tone_state: Optional[ToneState] = None,
) -> list[dict[str, Any]]:
    """
    Add tone to existing candidates (one tone per candidate).

    This is a simpler version that doesn't create variants.

    Args:
        candidates: Original candidates
        state: Current state
        responsibility_influence: Optional responsibility influence
        config: Tone configuration
        tone_state: Tone tracking state

    Returns:
        Candidates with tone metadata added
    """
    return generate_tone_variants(
        base_candidates=candidates,
        state=state,
        responsibility_influence=responsibility_influence,
        config=config,
        tone_state=tone_state,
        add_tone_variants=False,
    )


# ── Silence Integration ────────────────────────────────────────────


def apply_tone_to_silence(
    silence_candidate: dict[str, Any],
    state: PsycheState,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    config: Optional[ToneConfig] = None,
) -> dict[str, Any]:
    """
    Apply tone to a silence candidate.

    Silence can have a tone that affects how it's perceived:
    - NEUTRAL silence: simple pause
    - LIGHT silence: playful withholding
    - SERIOUS silence: contemplative silence
    - WARM silence: empathetic listening
    - RESERVED silence: cautious withholding

    Args:
        silence_candidate: Silence policy candidate
        state: Current state
        responsibility_influence: Optional responsibility influence
        config: Tone configuration

    Returns:
        Silence candidate with tone metadata
    """
    cfg = config or ToneConfig()

    # Compute tone bias
    modifier = compute_tone_bias(
        state=state,
        responsibility_influence=responsibility_influence,
        config=cfg,
    )

    # Select tone for silence
    tone = select_candidate_tone(
        silence_candidate, modifier, cfg, use_weighted_random=True
    )

    # Apply tone
    return apply_tone_to_candidate(silence_candidate, tone, modifier)


# ── Utility Functions ──────────────────────────────────────────────


def get_tone_summary(modifier: ToneModifier) -> str:
    """Get human-readable summary of tone modifier."""
    weights_str = ", ".join(
        f"{k}={v:.2f}" for k, v in sorted(
            modifier.weights.items(),
            key=lambda x: -x[1]
        )
    )
    return f"Tone: recommended={modifier.recommended.value}, weights=[{weights_str}]"


def get_tone_from_candidate(candidate: dict[str, Any]) -> str:
    """Get tone string from candidate."""
    return candidate.get("_tone", Tone.NEUTRAL.value)


def is_light_tone(candidate: dict[str, Any]) -> bool:
    """Check if candidate has light tone."""
    return get_candidate_tone(candidate) == Tone.LIGHT


def is_serious_tone(candidate: dict[str, Any]) -> bool:
    """Check if candidate has serious tone."""
    return get_candidate_tone(candidate) == Tone.SERIOUS


def create_tone_config(
    light_weight: float = 0.3,
    serious_weight: float = 0.3,
    positive_mood_factor: float = 1.5,
    high_fear_factor: float = 1.5,
) -> ToneConfig:
    """Create a tone configuration with custom parameters."""
    config = ToneConfig()
    config.base_weights[Tone.LIGHT.value] = light_weight
    config.base_weights[Tone.SERIOUS.value] = serious_weight
    config.positive_mood_light_factor = positive_mood_factor
    config.high_fear_serious_factor = high_fear_factor
    return config


def to_dict(config: ToneConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "base_weights": config.base_weights.copy(),
        "positive_mood_light_factor": config.positive_mood_light_factor,
        "negative_mood_serious_factor": config.negative_mood_serious_factor,
        "low_fear_light_factor": config.low_fear_light_factor,
        "high_fear_serious_factor": config.high_fear_serious_factor,
        "responsibility_serious_factor": config.responsibility_serious_factor,
        "low_responsibility_light_factor": config.low_responsibility_light_factor,
        "low_threshold": config.low_threshold,
        "high_threshold": config.high_threshold,
        "allow_light_silence": config.allow_light_silence,
        "max_consecutive_same_tone": config.max_consecutive_same_tone,
        "consecutive_decay": config.consecutive_decay,
        "minimum_weight": config.minimum_weight,
    }


def from_dict(data: dict[str, Any]) -> ToneConfig:
    """Deserialize config from dict."""
    config = ToneConfig()

    if "base_weights" in data:
        config.base_weights.update(data["base_weights"])

    config.positive_mood_light_factor = data.get(
        "positive_mood_light_factor", config.positive_mood_light_factor
    )
    config.negative_mood_serious_factor = data.get(
        "negative_mood_serious_factor", config.negative_mood_serious_factor
    )
    config.low_fear_light_factor = data.get(
        "low_fear_light_factor", config.low_fear_light_factor
    )
    config.high_fear_serious_factor = data.get(
        "high_fear_serious_factor", config.high_fear_serious_factor
    )
    config.responsibility_serious_factor = data.get(
        "responsibility_serious_factor", config.responsibility_serious_factor
    )
    config.low_responsibility_light_factor = data.get(
        "low_responsibility_light_factor", config.low_responsibility_light_factor
    )
    config.low_threshold = data.get("low_threshold", config.low_threshold)
    config.high_threshold = data.get("high_threshold", config.high_threshold)
    config.allow_light_silence = data.get(
        "allow_light_silence", config.allow_light_silence
    )
    config.max_consecutive_same_tone = data.get(
        "max_consecutive_same_tone", config.max_consecutive_same_tone
    )
    config.consecutive_decay = data.get("consecutive_decay", config.consecutive_decay)
    config.minimum_weight = data.get("minimum_weight", config.minimum_weight)

    return config
