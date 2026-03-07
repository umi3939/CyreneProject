"""
psyche/context_sensitivity.py - Context Sensitivity Bias (空気読みバイアス)

Creates a mechanism to receive abstract external context and generate
sensitivity bias that modulates decision candidate scoring.

Design principles (from design_context_sensitivity.md):
- Context is NOT interpreted for emotion/intent (no LLM guessing)
- Context is treated as pure signal intensity/weight
- Bias acts as "resistance" or "lubricant" for decisions
- Bias applies universally to all candidates
- Temporary effect with natural decay

Usage::

    from psyche.context_sensitivity import (
        ExternalContext,
        SensitivityBias,
        ContextSensitivityConfig,
        compute_sensitivity_bias,
        apply_sensitivity_to_candidates,
        create_external_context,
    )

    # Receive external context signals
    context = create_external_context(
        pace=0.3,      # Slow/heavy conversation
        density=0.7,   # Dense topic
        weight=0.6,    # Heavy atmosphere
    )

    # Compute sensitivity bias
    bias = compute_sensitivity_bias(context, config)

    # Apply to candidates
    adjusted = apply_sensitivity_to_candidates(candidates, bias)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time


# ── External Context ───────────────────────────────────────────────


@dataclass
class ExternalContext:
    """
    Abstract external context signals.

    These are NOT interpretations of emotion or intent.
    They are pure signal intensities representing:
    - How the conversation is flowing
    - The weight/density of the interaction
    - The pace and rhythm of exchange

    All values are 0.0 to 1.0 where:
    - 0.0 = minimal signal
    - 1.0 = maximal signal
    """

    # Pace: how fast/slow the exchange is (0=slow, 1=fast)
    pace: float = 0.5

    # Weight: heaviness of the atmosphere (0=light, 1=heavy)
    weight: float = 0.5

    # Density: information/emotional density (0=sparse, 1=dense)
    density: float = 0.5

    # Continuity: how connected to previous exchange (0=discontinuous, 1=continuous)
    continuity: float = 0.5

    # Responsiveness: how engaged the other party seems (0=unresponsive, 1=highly responsive)
    responsiveness: float = 0.5

    # Timestamp for decay calculation
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Clamp all values to valid range."""
        self.pace = max(0.0, min(1.0, self.pace))
        self.weight = max(0.0, min(1.0, self.weight))
        self.density = max(0.0, min(1.0, self.density))
        self.continuity = max(0.0, min(1.0, self.continuity))
        self.responsiveness = max(0.0, min(1.0, self.responsiveness))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "pace": self.pace,
            "weight": self.weight,
            "density": self.density,
            "continuity": self.continuity,
            "responsiveness": self.responsiveness,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalContext":
        """Deserialize from dict."""
        return cls(
            pace=data.get("pace", 0.5),
            weight=data.get("weight", 0.5),
            density=data.get("density", 0.5),
            continuity=data.get("continuity", 0.5),
            responsiveness=data.get("responsiveness", 0.5),
            timestamp=data.get("timestamp", time.time()),
        )


def create_external_context(
    pace: float = 0.5,
    weight: float = 0.5,
    density: float = 0.5,
    continuity: float = 0.5,
    responsiveness: float = 0.5,
) -> ExternalContext:
    """Create an external context with specified signals."""
    return ExternalContext(
        pace=pace,
        weight=weight,
        density=density,
        continuity=continuity,
        responsiveness=responsiveness,
    )


def create_neutral_context() -> ExternalContext:
    """Create a neutral context (all signals at midpoint)."""
    return ExternalContext()


def create_heavy_context() -> ExternalContext:
    """Create a heavy context (slow, dense, weighty)."""
    return ExternalContext(
        pace=0.2,
        weight=0.8,
        density=0.7,
        continuity=0.6,
        responsiveness=0.4,
    )


def create_light_context() -> ExternalContext:
    """Create a light context (fast, light, sparse)."""
    return ExternalContext(
        pace=0.8,
        weight=0.2,
        density=0.3,
        continuity=0.5,
        responsiveness=0.8,
    )


# ── Sensitivity Bias ───────────────────────────────────────────────


@dataclass
class SensitivityBias:
    """
    Computed sensitivity bias from external context.

    This bias modulates ALL decision candidates equally.
    It does NOT:
    - Judge right or wrong
    - Interpret emotion or intent
    - Block or force any specific candidate

    It acts as:
    - Resistance (high caution → dampen bold choices)
    - Lubricant (low caution → ease bold choices)
    """

    # Overall caution level (0.0 = no caution, 1.0 = maximum caution)
    # High caution dampens risky/bold candidates
    caution_level: float = 0.5

    # Score dampening factor for "risky" candidates
    # Applied as: adjusted_score = score * (1 - risk_dampening * candidate_risk)
    risk_dampening: float = 0.0

    # Score boosting factor for "safe" candidates
    # Applied as: adjusted_score = score + (safety_boost * candidate_safety)
    safety_boost: float = 0.0

    # Overall score multiplier (applies to all candidates)
    # < 1.0 = contract all scores, > 1.0 = expand all scores
    score_multiplier: float = 1.0

    # Threshold shift for selection
    # Positive = harder to select any candidate, Negative = easier
    selection_threshold_shift: float = 0.0

    # Context summary (for debugging)
    context_summary: dict[str, float] = field(default_factory=dict)

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "caution_level": self.caution_level,
            "risk_dampening": self.risk_dampening,
            "safety_boost": self.safety_boost,
            "score_multiplier": self.score_multiplier,
            "selection_threshold_shift": self.selection_threshold_shift,
            "context_summary": self.context_summary.copy(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensitivityBias":
        """Deserialize from dict."""
        return cls(
            caution_level=data.get("caution_level", 0.5),
            risk_dampening=data.get("risk_dampening", 0.0),
            safety_boost=data.get("safety_boost", 0.0),
            score_multiplier=data.get("score_multiplier", 1.0),
            selection_threshold_shift=data.get("selection_threshold_shift", 0.0),
            context_summary=data.get("context_summary", {}),
            timestamp=data.get("timestamp", time.time()),
        )


def create_neutral_bias() -> SensitivityBias:
    """Create a neutral bias (no effect)."""
    return SensitivityBias(
        caution_level=0.5,
        risk_dampening=0.0,
        safety_boost=0.0,
        score_multiplier=1.0,
        selection_threshold_shift=0.0,
    )


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class ContextSensitivityConfig:
    """
    Configuration for context sensitivity behavior.

    All parameters are externally configurable.
    No hardcoded interpretations of context signals.
    """

    # Base caution level (before context modification)
    base_caution: float = 0.5

    # How much "weight" signal increases caution
    weight_caution_factor: float = 0.5

    # How much "density" signal increases caution
    density_caution_factor: float = 0.3

    # How much slow "pace" increases caution
    slow_pace_caution_factor: float = 0.3

    # How much low "responsiveness" increases caution
    low_responsiveness_caution_factor: float = 0.4

    # How much "continuity" reduces caution (familiar = less cautious)
    continuity_relief_factor: float = 0.2

    # Maximum risk dampening (prevents complete blocking)
    max_risk_dampening: float = 0.3

    # Maximum safety boost
    max_safety_boost: float = 0.2

    # Score multiplier range
    min_score_multiplier: float = 0.8
    max_score_multiplier: float = 1.2

    # Decay rate for context influence (per second)
    decay_rate: float = 0.1

    # Minimum caution (never fully reckless)
    min_caution: float = 0.1

    # Maximum caution (never fully paralyzed)
    max_caution: float = 0.9

    # Policy-specific risk levels (for dampening calculation)
    # Higher = more risky, more affected by caution
    policy_risk_levels: dict[str, float] = field(default_factory=lambda: {
        "からかう": 0.8,        # Teasing is risky
        "話題を変える": 0.5,    # Topic change moderate risk
        "感想を述べる": 0.4,    # Sharing opinion moderate
        "質問で会話を広げる": 0.3,  # Questions lower risk
        "共感する": 0.2,        # Empathy low risk
        "励ます": 0.2,          # Encouragement low risk
        "沈黙する": 0.1,        # Silence very low risk
    })

    # Default risk for unknown policies
    default_policy_risk: float = 0.5


# ── Context State Tracking ─────────────────────────────────────────


@dataclass
class ContextState:
    """
    Tracks context-related state across turns.

    Enables decay and smoothing of context signals.
    """

    # Last received context
    last_context: Optional[ExternalContext] = None

    # Smoothed context values (exponential moving average)
    smoothed_pace: float = 0.5
    smoothed_weight: float = 0.5
    smoothed_density: float = 0.5
    smoothed_continuity: float = 0.5
    smoothed_responsiveness: float = 0.5

    # Last update timestamp
    last_update: float = field(default_factory=time.time)

    # Smoothing factor (0 = no smoothing, 1 = full smoothing).
    # Limitation: this alpha is fixed and does not adapt to the actual tick
    # interval. If tick intervals vary significantly (e.g. 0.5s vs 5s), the
    # effective smoothing window changes proportionally. For the current
    # system where tick intervals are relatively stable, this is acceptable.
    smoothing_alpha: float = 0.3

    def update(self, context: ExternalContext) -> "ContextState":
        """Update state with new context (applies smoothing)."""
        alpha = self.smoothing_alpha

        return ContextState(
            last_context=context,
            smoothed_pace=alpha * context.pace + (1 - alpha) * self.smoothed_pace,
            smoothed_weight=alpha * context.weight + (1 - alpha) * self.smoothed_weight,
            smoothed_density=alpha * context.density + (1 - alpha) * self.smoothed_density,
            smoothed_continuity=alpha * context.continuity + (1 - alpha) * self.smoothed_continuity,
            smoothed_responsiveness=alpha * context.responsiveness + (1 - alpha) * self.smoothed_responsiveness,
            last_update=time.time(),
            smoothing_alpha=self.smoothing_alpha,
        )

    def apply_decay(self, delta_time: float, decay_rate: float = 0.1) -> "ContextState":
        """Apply decay toward neutral (0.5)."""
        decay = (1 - decay_rate) ** delta_time
        neutral = 0.5

        def decay_toward_neutral(value: float) -> float:
            return neutral + (value - neutral) * decay

        return ContextState(
            last_context=self.last_context,
            smoothed_pace=decay_toward_neutral(self.smoothed_pace),
            smoothed_weight=decay_toward_neutral(self.smoothed_weight),
            smoothed_density=decay_toward_neutral(self.smoothed_density),
            smoothed_continuity=decay_toward_neutral(self.smoothed_continuity),
            smoothed_responsiveness=decay_toward_neutral(self.smoothed_responsiveness),
            last_update=time.time(),
            smoothing_alpha=self.smoothing_alpha,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "last_context": self.last_context.to_dict() if self.last_context else None,
            "smoothed_pace": self.smoothed_pace,
            "smoothed_weight": self.smoothed_weight,
            "smoothed_density": self.smoothed_density,
            "smoothed_continuity": self.smoothed_continuity,
            "smoothed_responsiveness": self.smoothed_responsiveness,
            "last_update": self.last_update,
            "smoothing_alpha": self.smoothing_alpha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextState":
        """Deserialize from dict."""
        last_ctx = None
        if data.get("last_context"):
            last_ctx = ExternalContext.from_dict(data["last_context"])

        return cls(
            last_context=last_ctx,
            smoothed_pace=data.get("smoothed_pace", 0.5),
            smoothed_weight=data.get("smoothed_weight", 0.5),
            smoothed_density=data.get("smoothed_density", 0.5),
            smoothed_continuity=data.get("smoothed_continuity", 0.5),
            smoothed_responsiveness=data.get("smoothed_responsiveness", 0.5),
            last_update=data.get("last_update", time.time()),
            smoothing_alpha=data.get("smoothing_alpha", 0.3),
        )


# ── Core Computation ───────────────────────────────────────────────


def compute_sensitivity_bias(
    context: ExternalContext,
    config: Optional[ContextSensitivityConfig] = None,
    context_state: Optional[ContextState] = None,
) -> SensitivityBias:
    """
    Compute sensitivity bias from external context.

    This function:
    1. Reads context signals (NOT interpreting emotion/intent)
    2. Computes caution level from signal weights
    3. Generates bias values that apply universally

    Args:
        context: External context signals
        config: Sensitivity configuration
        context_state: State for smoothing/decay (optional)

    Returns:
        SensitivityBias that can be applied to candidates
    """
    cfg = config or ContextSensitivityConfig()

    # Use smoothed values if state available
    if context_state:
        pace = context_state.smoothed_pace
        weight = context_state.smoothed_weight
        density = context_state.smoothed_density
        continuity = context_state.smoothed_continuity
        responsiveness = context_state.smoothed_responsiveness
    else:
        pace = context.pace
        weight = context.weight
        density = context.density
        continuity = context.continuity
        responsiveness = context.responsiveness

    # ── Compute caution level ──

    caution = cfg.base_caution

    # High weight → more caution
    if weight > 0.5:
        caution += (weight - 0.5) * cfg.weight_caution_factor

    # High density → more caution
    if density > 0.5:
        caution += (density - 0.5) * cfg.density_caution_factor

    # Slow pace → more caution
    if pace < 0.5:
        caution += (0.5 - pace) * cfg.slow_pace_caution_factor

    # Low responsiveness → more caution
    if responsiveness < 0.5:
        caution += (0.5 - responsiveness) * cfg.low_responsiveness_caution_factor

    # High continuity → less caution (familiar context)
    if continuity > 0.5:
        caution -= (continuity - 0.5) * cfg.continuity_relief_factor

    # Clamp caution to valid range
    caution = max(cfg.min_caution, min(cfg.max_caution, caution))

    # ── Compute derived bias values ──

    # Risk dampening: higher caution = more dampening
    risk_dampening = 0.0
    if caution > 0.5:
        risk_dampening = (caution - 0.5) * 2 * cfg.max_risk_dampening

    # Safety boost: higher caution = more boost for safe options
    safety_boost = 0.0
    if caution > 0.5:
        safety_boost = (caution - 0.5) * 2 * cfg.max_safety_boost

    # Score multiplier: high caution = contract scores, low caution = expand
    if caution > 0.5:
        # Contract scores (harder to reach threshold)
        multiplier = 1.0 - (caution - 0.5) * 2 * (1.0 - cfg.min_score_multiplier)
    else:
        # Expand scores (easier to reach threshold)
        multiplier = 1.0 + (0.5 - caution) * 2 * (cfg.max_score_multiplier - 1.0)

    # Selection threshold shift: high caution = raise threshold
    threshold_shift = (caution - 0.5) * 0.5

    return SensitivityBias(
        caution_level=round(caution, 3),
        risk_dampening=round(risk_dampening, 3),
        safety_boost=round(safety_boost, 3),
        score_multiplier=round(multiplier, 3),
        selection_threshold_shift=round(threshold_shift, 3),
        context_summary={
            "pace": pace,
            "weight": weight,
            "density": density,
            "continuity": continuity,
            "responsiveness": responsiveness,
        },
    )


# ── Candidate Application ──────────────────────────────────────────


def get_policy_risk(
    candidate: dict[str, Any],
    config: Optional[ContextSensitivityConfig] = None,
) -> float:
    """
    Get the risk level of a policy candidate.

    This is NOT a judgment of right/wrong.
    It's a structural property: some policies are inherently
    more "bold" and thus more affected by caution.
    """
    cfg = config or ContextSensitivityConfig()
    label = candidate.get("policy_label", "")

    # Check for silence (always low risk)
    if candidate.get("_is_silence", False):
        return cfg.policy_risk_levels.get("沈黙する", 0.1)

    # Check for light tone (slightly higher risk)
    tone = candidate.get("_tone", "neutral")
    tone_risk_modifier = 0.0
    if tone == "light":
        tone_risk_modifier = 0.1  # Light tone is slightly riskier

    base_risk = cfg.policy_risk_levels.get(label, cfg.default_policy_risk)

    return min(1.0, base_risk + tone_risk_modifier)


def apply_sensitivity_to_candidate(
    candidate: dict[str, Any],
    bias: SensitivityBias,
    config: Optional[ContextSensitivityConfig] = None,
) -> dict[str, Any]:
    """
    Apply sensitivity bias to a single candidate.

    The bias acts as:
    - Resistance: high caution dampens risky candidates
    - Lubricant: low caution eases selection

    Does NOT block or force any candidate.

    Args:
        candidate: Policy candidate dict
        bias: Computed sensitivity bias
        config: Sensitivity configuration

    Returns:
        New candidate dict with adjusted score
    """
    cfg = config or ContextSensitivityConfig()

    result = candidate.copy()
    original_score = candidate.get("_score", 0.0)

    # Get policy risk level
    risk = get_policy_risk(candidate, cfg)
    safety = 1.0 - risk

    # Apply risk dampening
    risk_penalty = bias.risk_dampening * risk
    score = original_score * (1.0 - risk_penalty)

    # Apply safety boost
    safety_bonus = bias.safety_boost * safety
    score = score + safety_bonus

    # Apply score multiplier
    score = score * bias.score_multiplier

    # Store adjusted score and metadata
    result["_score"] = round(score, 4)
    result["_original_score"] = original_score
    result["_sensitivity_adjusted"] = True
    result["_caution_level"] = bias.caution_level
    result["_policy_risk"] = risk

    return result


def apply_sensitivity_to_candidates(
    candidates: list[dict[str, Any]],
    bias: SensitivityBias,
    config: Optional[ContextSensitivityConfig] = None,
) -> list[dict[str, Any]]:
    """
    Apply sensitivity bias to all candidates.

    Bias applies universally - all candidates are affected.
    The list is re-sorted by adjusted scores.

    Args:
        candidates: List of policy candidates
        bias: Computed sensitivity bias
        config: Sensitivity configuration

    Returns:
        New list of candidates with adjusted scores, sorted
    """
    adjusted = [
        apply_sensitivity_to_candidate(c, bias, config)
        for c in candidates
    ]

    # Re-sort by adjusted score
    adjusted.sort(key=lambda c: c.get("_score", 0), reverse=True)

    return adjusted


# ── Full Pipeline ──────────────────────────────────────────────────


def process_with_context_sensitivity(
    candidates: list[dict[str, Any]],
    context: ExternalContext,
    config: Optional[ContextSensitivityConfig] = None,
    context_state: Optional[ContextState] = None,
) -> tuple[list[dict[str, Any]], SensitivityBias, ContextState]:
    """
    Full pipeline: compute bias and apply to candidates.

    Args:
        candidates: Original candidates
        context: External context signals
        config: Sensitivity configuration
        context_state: State for smoothing/decay

    Returns:
        Tuple of (adjusted_candidates, bias, updated_state)
    """
    cfg = config or ContextSensitivityConfig()
    state = context_state or ContextState()

    # Update state with new context
    state = state.update(context)

    # Compute bias
    bias = compute_sensitivity_bias(context, cfg, state)

    # Apply to candidates
    adjusted = apply_sensitivity_to_candidates(candidates, bias, cfg)

    return adjusted, bias, state


# ── Utility Functions ──────────────────────────────────────────────


def get_sensitivity_summary(bias: SensitivityBias) -> str:
    """Get human-readable summary of sensitivity bias."""
    caution_desc = "neutral"
    if bias.caution_level > 0.6:
        caution_desc = "cautious"
    elif bias.caution_level < 0.4:
        caution_desc = "relaxed"

    return (
        f"Context Sensitivity: {caution_desc} "
        f"(caution={bias.caution_level:.2f}, "
        f"risk_damp={bias.risk_dampening:.2f}, "
        f"safety_boost={bias.safety_boost:.2f})"
    )


def get_context_summary(context: ExternalContext) -> str:
    """Get human-readable summary of external context."""
    return (
        f"Context: pace={context.pace:.2f}, "
        f"weight={context.weight:.2f}, "
        f"density={context.density:.2f}, "
        f"continuity={context.continuity:.2f}, "
        f"resp={context.responsiveness:.2f}"
    )


def is_high_caution(bias: SensitivityBias, threshold: float = 0.6) -> bool:
    """Check if bias indicates high caution."""
    return bias.caution_level >= threshold


def is_low_caution(bias: SensitivityBias, threshold: float = 0.4) -> bool:
    """Check if bias indicates low caution."""
    return bias.caution_level <= threshold


def create_config(
    base_caution: float = 0.5,
    weight_factor: float = 0.5,
    max_risk_dampening: float = 0.3,
) -> ContextSensitivityConfig:
    """Create a sensitivity configuration with custom parameters."""
    return ContextSensitivityConfig(
        base_caution=base_caution,
        weight_caution_factor=weight_factor,
        max_risk_dampening=max_risk_dampening,
    )


def to_dict(config: ContextSensitivityConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "base_caution": config.base_caution,
        "weight_caution_factor": config.weight_caution_factor,
        "density_caution_factor": config.density_caution_factor,
        "slow_pace_caution_factor": config.slow_pace_caution_factor,
        "low_responsiveness_caution_factor": config.low_responsiveness_caution_factor,
        "continuity_relief_factor": config.continuity_relief_factor,
        "max_risk_dampening": config.max_risk_dampening,
        "max_safety_boost": config.max_safety_boost,
        "min_score_multiplier": config.min_score_multiplier,
        "max_score_multiplier": config.max_score_multiplier,
        "decay_rate": config.decay_rate,
        "min_caution": config.min_caution,
        "max_caution": config.max_caution,
        "policy_risk_levels": config.policy_risk_levels.copy(),
        "default_policy_risk": config.default_policy_risk,
    }


def from_dict(data: dict[str, Any]) -> ContextSensitivityConfig:
    """Deserialize config from dict."""
    config = ContextSensitivityConfig()

    config.base_caution = data.get("base_caution", config.base_caution)
    config.weight_caution_factor = data.get("weight_caution_factor", config.weight_caution_factor)
    config.density_caution_factor = data.get("density_caution_factor", config.density_caution_factor)
    config.slow_pace_caution_factor = data.get("slow_pace_caution_factor", config.slow_pace_caution_factor)
    config.low_responsiveness_caution_factor = data.get(
        "low_responsiveness_caution_factor", config.low_responsiveness_caution_factor
    )
    config.continuity_relief_factor = data.get("continuity_relief_factor", config.continuity_relief_factor)
    config.max_risk_dampening = data.get("max_risk_dampening", config.max_risk_dampening)
    config.max_safety_boost = data.get("max_safety_boost", config.max_safety_boost)
    config.min_score_multiplier = data.get("min_score_multiplier", config.min_score_multiplier)
    config.max_score_multiplier = data.get("max_score_multiplier", config.max_score_multiplier)
    config.decay_rate = data.get("decay_rate", config.decay_rate)
    config.min_caution = data.get("min_caution", config.min_caution)
    config.max_caution = data.get("max_caution", config.max_caution)

    if "policy_risk_levels" in data:
        config.policy_risk_levels.update(data["policy_risk_levels"])

    config.default_policy_risk = data.get("default_policy_risk", config.default_policy_risk)

    return config
