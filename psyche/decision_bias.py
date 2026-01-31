"""
psyche/decision_bias.py - Decision Bias Injection Structure

Provides connection points for short-term memory and dynamics state
to influence decision scoring without changing the core decision logic.

Design principles (from design_decision_bias.md):
- Bias is lightweight and temporary, doesn't lock decisions
- Influence expressed as score modifiers (additive/multiplicative)
- Naturally weakens with existing decay structures
- No hardcoded values - all scale factors configurable
- Presence/absence of bias doesn't change algorithm flow

Usage::

    from psyche.decision_bias import compute_decision_bias, apply_bias_to_score

    bias = compute_decision_bias(short_term_memory, dynamics_state)
    modified_score = apply_bias_to_score(base_score, bias, policy_label, policy_def)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .short_term_memory import ShortTermMemory, ResidueInfluence, compute_residue_influence
from .dynamics import DynamicsState, DynamicsPhase, get_intensity_modifier


@dataclass
class DecisionBiasConfig:
    """
    Configuration for decision bias computation.

    All parameters are externally configurable.
    Defaults are neutral (no effect) to avoid hardcoding behavior.
    """

    # ── Scale factors for different bias sources ────────────────

    # Residue emotion influence scale
    residue_emotion_scale: float = 1.0

    # Residue valence influence scale
    residue_valence_scale: float = 1.0

    # Residue intent influence scale
    residue_intent_scale: float = 1.0

    # Dynamics phase boost/dampening scale
    dynamics_phase_scale: float = 1.0

    # Context continuity influence scale
    continuity_scale: float = 1.0

    # Overall bias scale (master volume)
    global_scale: float = 1.0

    # ── Mapping functions (injectable) ──────────────────────────

    # How emotion labels map to policy preferences
    # Default: None (uses built-in neutral mapping)
    emotion_to_policy_map: Optional[dict[str, dict[str, float]]] = None

    # How intents map to policy preferences
    # Default: None (uses built-in neutral mapping)
    intent_to_policy_map: Optional[dict[str, dict[str, float]]] = None


@dataclass
class DecisionBias:
    """
    Computed decision bias from short-term memory and dynamics.

    This structure is passed to the scoring function as a modifier layer.
    All values are raw aggregations - interpretation happens at application.
    """

    # ── Residue-derived biases ──────────────────────────────────

    # Emotion-specific biases from unprocessed residue
    # Keys are emotion labels, values are weighted influence
    emotion_biases: dict[str, float] = field(default_factory=dict)

    # Intent-specific biases from recent stimuli
    intent_biases: dict[str, float] = field(default_factory=dict)

    # Valence accumulation (positive/negative tendency)
    valence_bias: float = 0.0

    # Total residue intensity
    residue_intensity: float = 0.0

    # ── Dynamics-derived modifiers ──────────────────────────────

    # Current dynamics phase
    dynamics_phase: DynamicsPhase = DynamicsPhase.NORMAL

    # Peak boost (from dynamics intensity modifier)
    peak_boost: float = 0.0

    # Rebound dampening (from dynamics intensity modifier)
    rebound_dampening: float = 0.0

    # Peak emotion (if in peak/rebound phase)
    peak_emotion: str = ""

    # ── Context modifiers ───────────────────────────────────────

    # Context continuity score (0-1)
    continuity: float = 0.0

    # ── Configuration ───────────────────────────────────────────

    config: DecisionBiasConfig = field(default_factory=DecisionBiasConfig)

    def is_neutral(self) -> bool:
        """Check if bias has no effect (all values neutral)."""
        # Check if emotion biases have any non-zero values
        has_emotion_bias = any(v != 0.0 for v in self.emotion_biases.values())
        has_intent_bias = any(v != 0.0 for v in self.intent_biases.values())

        return (
            not has_emotion_bias
            and not has_intent_bias
            and self.valence_bias == 0.0
            and self.residue_intensity == 0.0
            and self.peak_boost == 0.0
            and self.rebound_dampening == 0.0
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary for diagnostics."""
        return {
            "phase": self.dynamics_phase.value,
            "residue_intensity": self.residue_intensity,
            "valence_bias": self.valence_bias,
            "peak_boost": self.peak_boost,
            "rebound_dampening": self.rebound_dampening,
            "continuity": self.continuity,
            "emotion_count": len(self.emotion_biases),
            "intent_count": len(self.intent_biases),
        }


# ── Bias Computation ────────────────────────────────────────────


def compute_decision_bias(
    memory: Optional[ShortTermMemory] = None,
    dynamics: Optional[DynamicsState] = None,
    config: Optional[DecisionBiasConfig] = None,
) -> DecisionBias:
    """
    Compute decision bias from short-term memory and dynamics state.

    This aggregates influence sources into a unified bias structure
    that can be passed to the scoring function.

    Args:
        memory: Short-term memory state (optional)
        dynamics: Dynamics state (optional)
        config: Bias configuration (optional)

    Returns:
        DecisionBias with computed influence values
    """
    cfg = config or DecisionBiasConfig()

    # Initialize with neutral values
    emotion_biases: dict[str, float] = {}
    intent_biases: dict[str, float] = {}
    valence_bias = 0.0
    residue_intensity = 0.0
    continuity = 0.0

    # ── Extract from ShortTermMemory ────────────────────────────

    if memory is not None:
        # Get residue influence
        residue = compute_residue_influence(memory)

        # Emotion biases from residue (scaled)
        for emo, weight in residue.emotion_influences.items():
            scaled = weight * cfg.residue_emotion_scale * cfg.global_scale
            emotion_biases[emo] = scaled

        # Residue intensity
        residue_intensity = residue.total_intensity * cfg.global_scale

        # Continuity
        continuity = residue.continuity * cfg.continuity_scale * cfg.global_scale

        # Valence accumulation from entries
        for entry in memory.get_unprocessed_residue():
            scaled_valence = entry.valence * entry.residue_weight * cfg.residue_valence_scale
            valence_bias += scaled_valence

            # Intent biases
            intent = entry.intent
            if intent:
                scaled_intent = entry.residue_weight * cfg.residue_intent_scale * cfg.global_scale
                intent_biases[intent] = intent_biases.get(intent, 0.0) + scaled_intent

        valence_bias *= cfg.global_scale

    # ── Extract from DynamicsState ──────────────────────────────

    dynamics_phase = DynamicsPhase.NORMAL
    peak_boost = 0.0
    rebound_dampening = 0.0
    peak_emotion = ""

    if dynamics is not None:
        dynamics_phase = dynamics.phase
        peak_emotion = dynamics.peak_emotion

        # Get intensity modifiers from dynamics
        boost, dampening = get_intensity_modifier(dynamics)
        peak_boost = boost * cfg.dynamics_phase_scale * cfg.global_scale
        rebound_dampening = dampening * cfg.dynamics_phase_scale * cfg.global_scale

    return DecisionBias(
        emotion_biases=emotion_biases,
        intent_biases=intent_biases,
        valence_bias=valence_bias,
        residue_intensity=residue_intensity,
        dynamics_phase=dynamics_phase,
        peak_boost=peak_boost,
        rebound_dampening=rebound_dampening,
        peak_emotion=peak_emotion,
        continuity=continuity,
        config=cfg,
    )


# ── Bias Application ────────────────────────────────────────────


# Default emotion-to-policy preference mapping
# Positive values = bonus, negative = penalty
# These are CONNECTION POINTS, not fixed logic
_DEFAULT_EMOTION_POLICY_MAP: dict[str, dict[str, float]] = {
    # When "angry" residue is present
    "angry": {
        "共感する": 0.5,      # Empathy helps
        "励ます": 0.3,        # Encouragement helps
        "からかう": -0.5,     # Teasing is risky
    },
    # When "sad" residue is present
    "sad": {
        "共感する": 0.5,
        "励ます": 0.5,
        "からかう": -0.3,
    },
    # When "happy" residue is present
    "happy": {
        "からかう": 0.3,      # Playfulness is ok
        "感想を述べる": 0.2,  # Sharing feelings is ok
    },
    # When "scared" residue is present
    "scared": {
        "共感する": 0.4,
        "励ます": 0.4,
        "からかう": -0.4,
        "話題を変える": -0.2,  # Avoiding might seem dismissive
    },
    # When "surprised" residue is present
    "surprised": {
        "質問で会話を広げる": 0.3,  # Curiosity matches
    },
    # When "loving" residue is present
    "loving": {
        "共感する": 0.3,
        "励ます": 0.2,
    },
}

# Default intent-to-policy preference mapping
_DEFAULT_INTENT_POLICY_MAP: dict[str, dict[str, float]] = {
    "sharing": {
        "共感する": 0.3,
        "質問で会話を広げる": 0.2,
    },
    "complaint": {
        "共感する": 0.4,
        "励ます": 0.3,
        "からかう": -0.4,
    },
    "question": {
        "質問で会話を広げる": 0.3,
    },
    "greeting": {
        "感想を述べる": 0.2,
    },
    "joke": {
        "からかう": 0.3,
    },
}


def apply_bias_to_score(
    base_score: float,
    bias: DecisionBias,
    policy_label: str,
    policy_def: Optional[dict] = None,
) -> float:
    """
    Apply decision bias to a candidate score.

    This is the INJECTION POINT where bias modifies scoring.
    The modification is additive (score + bias_adjustment).

    Args:
        base_score: Original score from base scoring logic
        bias: Computed decision bias
        policy_label: Label of the policy being scored
        policy_def: Optional policy definition (for future extensibility)

    Returns:
        Modified score with bias applied
    """
    if bias.is_neutral():
        return base_score

    adjustment = 0.0
    cfg = bias.config

    # ── Emotion-based adjustments ───────────────────────────────

    emo_map = cfg.emotion_to_policy_map or _DEFAULT_EMOTION_POLICY_MAP
    for emotion, weight in bias.emotion_biases.items():
        if emotion in emo_map and policy_label in emo_map[emotion]:
            preference = emo_map[emotion][policy_label]
            adjustment += weight * preference

    # ── Intent-based adjustments ────────────────────────────────

    intent_map = cfg.intent_to_policy_map or _DEFAULT_INTENT_POLICY_MAP
    for intent, weight in bias.intent_biases.items():
        if intent in intent_map and policy_label in intent_map[intent]:
            preference = intent_map[intent][policy_label]
            adjustment += weight * preference

    # ── Valence-based adjustments ───────────────────────────────

    # Negative valence → favor empathy/encouragement
    if bias.valence_bias < -0.1:
        if policy_label in ("共感する", "励ます"):
            adjustment += abs(bias.valence_bias) * 0.5
        elif policy_label == "からかう":
            adjustment -= abs(bias.valence_bias) * 0.3

    # Positive valence → slightly favor expression
    elif bias.valence_bias > 0.1:
        if policy_label in ("からかう", "感想を述べる"):
            adjustment += bias.valence_bias * 0.2

    # ── Dynamics phase adjustments ──────────────────────────────

    # Peak phase: boost scores for peak-emotion-aligned policies
    if bias.dynamics_phase == DynamicsPhase.PEAK:
        adjustment += bias.peak_boost
        # If peak emotion matches a policy preference, extra boost
        if bias.peak_emotion in emo_map and policy_label in emo_map[bias.peak_emotion]:
            adjustment += bias.peak_boost * 0.5

    # Rebound phase: dampen all adjustments
    elif bias.dynamics_phase == DynamicsPhase.REBOUND:
        # Reduce the adjustment magnitude during rebound
        adjustment *= (1.0 - bias.rebound_dampening)
        # Slight preference for calmer choices
        if policy_label in ("共感する", "質問で会話を広げる"):
            adjustment += 0.1

    # ── Continuity adjustments ──────────────────────────────────

    # High continuity: slight boost to consistent responses
    if bias.continuity > 0.5:
        # When context is continuous, favor empathy and questions
        if policy_label in ("共感する", "質問で会話を広げる"):
            adjustment += bias.continuity * 0.2

    return base_score + adjustment


def get_policy_bias_breakdown(
    bias: DecisionBias,
    policy_labels: list[str],
) -> dict[str, float]:
    """
    Get breakdown of bias adjustments for each policy.

    Useful for diagnostics and understanding bias influence.

    Returns:
        Dict mapping policy_label to bias adjustment amount
    """
    return {
        label: apply_bias_to_score(0.0, bias, label) - 0.0
        for label in policy_labels
    }


# ── Convenience Functions ───────────────────────────────────────


def create_neutral_bias() -> DecisionBias:
    """Create a bias with no effect (all neutral values)."""
    return DecisionBias()


def merge_biases(biases: list[DecisionBias]) -> DecisionBias:
    """
    Merge multiple bias sources into one.

    Values are summed. Config from first bias is used.
    """
    if not biases:
        return create_neutral_bias()

    merged_emotions: dict[str, float] = {}
    merged_intents: dict[str, float] = {}
    merged_valence = 0.0
    merged_residue = 0.0
    merged_peak_boost = 0.0
    merged_rebound = 0.0
    merged_continuity = 0.0

    # Use dynamics phase from first non-NORMAL bias
    dynamics_phase = DynamicsPhase.NORMAL
    peak_emotion = ""

    for b in biases:
        # Sum emotion biases
        for emo, val in b.emotion_biases.items():
            merged_emotions[emo] = merged_emotions.get(emo, 0.0) + val

        # Sum intent biases
        for intent, val in b.intent_biases.items():
            merged_intents[intent] = merged_intents.get(intent, 0.0) + val

        merged_valence += b.valence_bias
        merged_residue += b.residue_intensity
        merged_peak_boost += b.peak_boost
        merged_rebound += b.rebound_dampening
        merged_continuity = max(merged_continuity, b.continuity)

        if b.dynamics_phase != DynamicsPhase.NORMAL and dynamics_phase == DynamicsPhase.NORMAL:
            dynamics_phase = b.dynamics_phase
            peak_emotion = b.peak_emotion

    return DecisionBias(
        emotion_biases=merged_emotions,
        intent_biases=merged_intents,
        valence_bias=merged_valence,
        residue_intensity=merged_residue,
        dynamics_phase=dynamics_phase,
        peak_boost=merged_peak_boost,
        rebound_dampening=merged_rebound,
        peak_emotion=peak_emotion,
        continuity=merged_continuity,
        config=biases[0].config if biases else DecisionBiasConfig(),
    )
