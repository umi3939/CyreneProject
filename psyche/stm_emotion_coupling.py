"""
psyche/stm_emotion_coupling.py - Short-Term Memory to Emotion Coupling

Connects ShortTermMemory residue/context to MultiEmotion update logic.
Implements mechanisms where active STM items influence:
- Emotion persistence (slower decay for STM-supported emotions)
- Re-activation chance (past emotions can resurface)
- Accumulation (repeated emotions stack within context)

Design principles (from design_stm_emotion_coupling.md):
- STM affects "persistence/re-activation/accumulation ease", NOT emotion strength
- STM does NOT generate new emotions
- Compatible with Multi-Emotion system (no single emotion reduction)
- Read-only access to STM
- Influence is temporary and decays naturally

Usage::

    from psyche.stm_emotion_coupling import (
        STMEmotionCouplingConfig,
        CouplingInfluence,
        compute_coupling_influence,
        apply_persistence_modifier,
        apply_reactivation,
        apply_accumulation,
        compute_decay_modifier_from_stm,
    )

    # Compute coupling influence (read-only STM access)
    influence = compute_coupling_influence(stm, current_emotions)

    # Apply persistence modifier to decay
    modified_decay = apply_persistence_modifier(base_decay, influence)

    # Apply re-activation for continuous context
    reactivated = apply_reactivation(emotions, influence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .short_term_memory import ShortTermMemory, StimulusEntry
from .state import EmotionVector


# ── Emotion Label Mapping ──────────────────────────────────────────

# Maps STM emotion labels to EmotionVector field names
_STM_TO_EMOTION_FIELD: dict[str, str] = {
    "happy": "joy",
    "sad": "sorrow",
    "angry": "anger",
    "surprised": "surprise",
    "scared": "fear",
    "loving": "love",
    "teasing": "fun",
    "neutral": "",
    # Direct mappings
    "joy": "joy",
    "sorrow": "sorrow",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "love": "love",
    "fun": "fun",
}


# ── Configuration ──────────────────────────────────────────────────

@dataclass
class STMEmotionCouplingConfig:
    """
    Configuration for STM-Emotion coupling behavior.

    All parameters are externally configurable.
    No hardcoded emotion meanings or evaluations.
    """

    # Persistence: how much STM slows down emotion decay
    # 1.0 = no effect, 0.5 = decay halved, 0.0 = no decay
    persistence_factor_base: float = 0.7

    # Minimum residue weight to contribute to persistence
    persistence_residue_threshold: float = 0.1

    # Re-activation: boost for emotions present in STM
    # Applied when context is continuous
    reactivation_boost_base: float = 0.05

    # Context continuity threshold for re-activation
    reactivation_continuity_threshold: float = 0.3

    # Accumulation: how repeated emotions stack
    # Higher = faster accumulation within continuous context
    accumulation_rate: float = 0.1

    # Maximum accumulation boost (prevents runaway)
    accumulation_cap: float = 0.3

    # Decay modifier range
    decay_modifier_min: float = 0.5  # Slowest decay (high STM support)
    decay_modifier_max: float = 1.0  # Normal decay (no STM support)

    # Weight threshold for considering an STM entry active
    active_entry_threshold: float = 0.05


@dataclass
class EmotionCouplingData:
    """
    Per-emotion coupling data computed from STM.

    Each emotion has its own coupling state, computed independently.
    No emotion affects another's coupling.
    """

    # How much STM supports this emotion's persistence (0.0-1.0)
    persistence_support: float = 0.0

    # Re-activation potential from STM (0.0-1.0)
    reactivation_potential: float = 0.0

    # Accumulated weight from repeated stimuli (0.0-cap)
    accumulation_weight: float = 0.0

    # Number of STM entries supporting this emotion
    supporting_entry_count: int = 0


@dataclass
class CouplingInfluence:
    """
    Computed coupling influence from STM to emotions.

    This is the interface between STM and emotion updates.
    Read-only - does not modify STM.
    """

    # Per-emotion coupling data
    emotion_data: dict[str, EmotionCouplingData] = field(default_factory=dict)

    # Overall context continuity (from STM)
    context_continuity: float = 0.0

    # Whether context is considered continuous
    is_continuous: bool = False

    # Total active STM entries
    active_entry_count: int = 0

    # Configuration used (for reference)
    config: STMEmotionCouplingConfig = field(
        default_factory=STMEmotionCouplingConfig
    )


# ── Core Computation (Read-Only STM Access) ────────────────────────


def compute_coupling_influence(
    stm: ShortTermMemory,
    current_emotions: Optional[EmotionVector] = None,
    config: Optional[STMEmotionCouplingConfig] = None,
) -> CouplingInfluence:
    """
    Compute coupling influence from STM (read-only access).

    This function:
    1. Reads STM entries without modification
    2. Computes per-emotion persistence/reactivation/accumulation data
    3. Returns a CouplingInfluence for use in emotion updates

    Args:
        stm: Short-term memory (read-only access)
        current_emotions: Current emotion state (optional, for context)
        config: Optional configuration

    Returns:
        CouplingInfluence structure
    """
    cfg = config or STMEmotionCouplingConfig()

    # Get active entries from STM (read-only)
    active_entries = [
        e for e in stm.entries
        if e.residue_weight >= cfg.active_entry_threshold
    ]

    # Compute per-emotion data
    emotion_data: dict[str, EmotionCouplingData] = {}

    # Initialize for all standard emotions
    for field_name in ["joy", "sorrow", "anger", "fear", "surprise", "love", "fun"]:
        emotion_data[field_name] = _compute_emotion_coupling(
            field_name, active_entries, cfg
        )

    # Determine context continuity
    is_continuous = stm.context_continuity_score >= cfg.reactivation_continuity_threshold

    return CouplingInfluence(
        emotion_data=emotion_data,
        context_continuity=stm.context_continuity_score,
        is_continuous=is_continuous,
        active_entry_count=len(active_entries),
        config=cfg,
    )


def _compute_emotion_coupling(
    emotion_field: str,
    entries: list[StimulusEntry],
    config: STMEmotionCouplingConfig,
) -> EmotionCouplingData:
    """
    Compute coupling data for a single emotion.

    Each emotion is computed independently - no cross-emotion effects.
    """
    # Find entries that support this emotion
    supporting_entries = []
    for entry in entries:
        mapped_field = _STM_TO_EMOTION_FIELD.get(entry.emotion_label, "")
        if mapped_field == emotion_field:
            supporting_entries.append(entry)

    if not supporting_entries:
        return EmotionCouplingData()

    # Persistence support: weighted average of residue weights
    total_weight = sum(e.residue_weight for e in supporting_entries)
    persistence_support = min(1.0, total_weight / len(supporting_entries))

    # Re-activation potential: based on intensity and recency
    reactivation_potential = 0.0
    for entry in supporting_entries:
        if entry.residue_weight >= config.persistence_residue_threshold:
            contribution = entry.raw_intensity * entry.residue_weight
            reactivation_potential += contribution
    reactivation_potential = min(1.0, reactivation_potential)

    # Accumulation weight: sum of intensities for repeated stimuli
    accumulation_weight = 0.0
    for entry in supporting_entries:
        accumulation_weight += entry.raw_intensity * config.accumulation_rate
    accumulation_weight = min(config.accumulation_cap, accumulation_weight)

    return EmotionCouplingData(
        persistence_support=persistence_support,
        reactivation_potential=reactivation_potential,
        accumulation_weight=accumulation_weight,
        supporting_entry_count=len(supporting_entries),
    )


# ── Persistence Modifier ───────────────────────────────────────────


def compute_decay_modifier_from_stm(
    emotion_name: str,
    influence: CouplingInfluence,
) -> float:
    """
    Compute decay rate modifier for an emotion based on STM coupling.

    Higher STM support = slower decay (lower modifier).
    No STM support = normal decay (modifier = 1.0).

    Args:
        emotion_name: Name of the emotion field
        influence: Computed coupling influence

    Returns:
        Decay modifier (0.5 to 1.0, where lower = slower decay)
    """
    cfg = influence.config
    data = influence.emotion_data.get(emotion_name)

    if data is None or data.persistence_support == 0:
        return cfg.decay_modifier_max  # Normal decay

    # Interpolate between max (no support) and min (full support)
    support = data.persistence_support
    modifier = cfg.decay_modifier_max - (
        support * (cfg.decay_modifier_max - cfg.decay_modifier_min)
    )

    return max(cfg.decay_modifier_min, min(cfg.decay_modifier_max, modifier))


def apply_persistence_modifier(
    emotions: EmotionVector,
    base_decay_rate: float,
    delta_time: float,
    influence: CouplingInfluence,
) -> EmotionVector:
    """
    Apply STM-influenced decay to emotions (persistence effect).

    Emotions with STM support decay slower.
    Each emotion decays independently based on its own STM support.

    Args:
        emotions: Current emotion vector
        base_decay_rate: Base decay rate (e.g., 0.95 per second)
        delta_time: Time elapsed
        influence: Computed coupling influence

    Returns:
        New EmotionVector with STM-modified decay applied
    """
    emo = emotions.as_dict()

    for emotion_name, value in emo.items():
        # Get STM-based decay modifier
        modifier = compute_decay_modifier_from_stm(emotion_name, influence)

        # Adjust decay rate: higher modifier = faster decay
        adjusted_rate = base_decay_rate ** modifier

        # Apply decay
        decay_factor = adjusted_rate ** delta_time
        emo[emotion_name] = max(0.0, min(1.0, value * decay_factor))

    return EmotionVector(**emo)


# ── Re-activation ──────────────────────────────────────────────────


def apply_reactivation(
    emotions: EmotionVector,
    influence: CouplingInfluence,
) -> EmotionVector:
    """
    Apply re-activation effect from STM.

    When context is continuous, emotions present in STM can be boosted.
    This is NOT generating new emotions - only boosting existing ones
    that have STM support.

    Args:
        emotions: Current emotion vector
        influence: Computed coupling influence

    Returns:
        New EmotionVector with re-activation applied
    """
    # Only apply if context is continuous
    if not influence.is_continuous:
        return emotions

    cfg = influence.config
    emo = emotions.as_dict()

    for emotion_name, data in influence.emotion_data.items():
        if emotion_name not in emo:
            continue

        # Only re-activate if there's potential AND current value > 0
        # (We don't create emotions from nothing)
        if data.reactivation_potential > 0 and emo[emotion_name] > 0:
            # Boost proportional to reactivation potential
            boost = cfg.reactivation_boost_base * data.reactivation_potential

            # Scale boost by context continuity
            scaled_boost = boost * influence.context_continuity

            emo[emotion_name] = max(0.0, min(1.0, emo[emotion_name] + scaled_boost))

    return EmotionVector(**emo)


def apply_reactivation_to_existing(
    emotions: EmotionVector,
    influence: CouplingInfluence,
    existence_threshold: float = 0.01,
) -> EmotionVector:
    """
    Apply re-activation only to emotions that already exist (above threshold).

    This ensures we don't "create" emotions from STM, only boost existing ones.

    Args:
        emotions: Current emotion vector
        influence: Computed coupling influence
        existence_threshold: Minimum value to consider emotion "existing"

    Returns:
        New EmotionVector with re-activation applied
    """
    if not influence.is_continuous:
        return emotions

    cfg = influence.config
    emo = emotions.as_dict()

    for emotion_name, data in influence.emotion_data.items():
        if emotion_name not in emo:
            continue

        # Only boost if emotion already exists above threshold
        if emo[emotion_name] >= existence_threshold and data.reactivation_potential > 0:
            boost = cfg.reactivation_boost_base * data.reactivation_potential
            scaled_boost = boost * influence.context_continuity
            emo[emotion_name] = max(0.0, min(1.0, emo[emotion_name] + scaled_boost))

    return EmotionVector(**emo)


# ── Accumulation ───────────────────────────────────────────────────


def apply_accumulation(
    emotions: EmotionVector,
    influence: CouplingInfluence,
) -> EmotionVector:
    """
    Apply accumulation effect from STM.

    When the same emotion is stimulated repeatedly within continuous context,
    it accumulates (stacks). This is bounded by accumulation_cap.

    Args:
        emotions: Current emotion vector
        influence: Computed coupling influence

    Returns:
        New EmotionVector with accumulation applied
    """
    # Only accumulate if context is continuous
    if not influence.is_continuous:
        return emotions

    emo = emotions.as_dict()

    for emotion_name, data in influence.emotion_data.items():
        if emotion_name not in emo:
            continue

        # Add accumulation weight (already capped in compute)
        if data.accumulation_weight > 0 and emo[emotion_name] > 0:
            # Accumulation scales with context continuity
            scaled_accumulation = data.accumulation_weight * influence.context_continuity
            emo[emotion_name] = max(0.0, min(1.0, emo[emotion_name] + scaled_accumulation))

    return EmotionVector(**emo)


# ── Combined Application ───────────────────────────────────────────


def apply_stm_coupling(
    emotions: EmotionVector,
    stm: ShortTermMemory,
    base_decay_rate: float = 0.95,
    delta_time: float = 1.0,
    config: Optional[STMEmotionCouplingConfig] = None,
    apply_persistence: bool = True,
    apply_reactivation_effect: bool = True,
    apply_accumulation_effect: bool = True,
) -> tuple[EmotionVector, CouplingInfluence]:
    """
    Apply all STM coupling effects to emotions.

    This is the main entry point for STM-emotion coupling.
    Order of operations:
    1. Compute coupling influence (read-only STM access)
    2. Apply persistence (slower decay for STM-supported emotions)
    3. Apply re-activation (boost for emotions in STM during continuous context)
    4. Apply accumulation (stacking for repeated emotions)

    Args:
        emotions: Current emotion vector
        stm: Short-term memory (read-only)
        base_decay_rate: Base decay rate
        delta_time: Time elapsed
        config: Optional configuration
        apply_persistence: Whether to apply persistence effect
        apply_reactivation_effect: Whether to apply re-activation
        apply_accumulation_effect: Whether to apply accumulation

    Returns:
        Tuple of (new_emotions, coupling_influence)
    """
    # Step 1: Compute coupling influence (read-only)
    influence = compute_coupling_influence(stm, emotions, config)

    result = emotions

    # Step 2: Apply persistence (modified decay)
    if apply_persistence:
        result = apply_persistence_modifier(
            result, base_decay_rate, delta_time, influence
        )

    # Step 3: Apply re-activation
    if apply_reactivation_effect:
        result = apply_reactivation_to_existing(result, influence)

    # Step 4: Apply accumulation
    if apply_accumulation_effect:
        result = apply_accumulation(result, influence)

    return result, influence


# ── Utility Functions ──────────────────────────────────────────────


def get_coupling_summary(influence: CouplingInfluence) -> str:
    """
    Get human-readable summary of coupling state.

    Args:
        influence: Computed coupling influence

    Returns:
        Summary string
    """
    if influence.active_entry_count == 0:
        return "STM-Emotion Coupling: No active STM entries"

    parts = [f"STM entries: {influence.active_entry_count}"]
    parts.append(f"continuity: {influence.context_continuity:.2f}")

    if influence.is_continuous:
        parts.append("[CONTINUOUS]")

    # List emotions with significant support
    supported = []
    for name, data in influence.emotion_data.items():
        if data.persistence_support > 0.1:
            supported.append(f"{name}({data.supporting_entry_count})")

    if supported:
        parts.append(f"supported: {', '.join(supported)}")

    return "STM-Emotion Coupling: " + " | ".join(parts)


def get_emotion_persistence_breakdown(
    influence: CouplingInfluence,
) -> dict[str, float]:
    """
    Get per-emotion decay modifier breakdown.

    Useful for debugging and understanding persistence effects.

    Returns:
        Dict of emotion_name -> decay_modifier
    """
    return {
        name: compute_decay_modifier_from_stm(name, influence)
        for name in influence.emotion_data.keys()
    }


def create_coupling_config(
    persistence_factor: float = 0.7,
    reactivation_boost: float = 0.05,
    accumulation_rate: float = 0.1,
    continuity_threshold: float = 0.3,
) -> STMEmotionCouplingConfig:
    """Create a coupling configuration with custom parameters."""
    return STMEmotionCouplingConfig(
        persistence_factor_base=persistence_factor,
        reactivation_boost_base=reactivation_boost,
        accumulation_rate=accumulation_rate,
        reactivation_continuity_threshold=continuity_threshold,
    )


def to_dict(config: STMEmotionCouplingConfig) -> dict[str, Any]:
    """Serialize config to dict."""
    return {
        "persistence_factor_base": config.persistence_factor_base,
        "persistence_residue_threshold": config.persistence_residue_threshold,
        "reactivation_boost_base": config.reactivation_boost_base,
        "reactivation_continuity_threshold": config.reactivation_continuity_threshold,
        "accumulation_rate": config.accumulation_rate,
        "accumulation_cap": config.accumulation_cap,
        "decay_modifier_min": config.decay_modifier_min,
        "decay_modifier_max": config.decay_modifier_max,
        "active_entry_threshold": config.active_entry_threshold,
    }


def from_dict(data: dict[str, Any]) -> STMEmotionCouplingConfig:
    """Deserialize config from dict."""
    return STMEmotionCouplingConfig(
        persistence_factor_base=data.get("persistence_factor_base", 0.7),
        persistence_residue_threshold=data.get("persistence_residue_threshold", 0.1),
        reactivation_boost_base=data.get("reactivation_boost_base", 0.05),
        reactivation_continuity_threshold=data.get("reactivation_continuity_threshold", 0.3),
        accumulation_rate=data.get("accumulation_rate", 0.1),
        accumulation_cap=data.get("accumulation_cap", 0.3),
        decay_modifier_min=data.get("decay_modifier_min", 0.5),
        decay_modifier_max=data.get("decay_modifier_max", 1.0),
        active_entry_threshold=data.get("active_entry_threshold", 0.05),
    )
