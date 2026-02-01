"""
psyche/self_reference.py - Self-Reference Loop Structure

Implements a self-reference loop where internal state is:
1. Acquired (read-only)
2. Summarized (coarse compression)
3. Tagged (temporary self-reference markers)
4. Made available for decision/emotion modification

Design principles (from design_self_reference.md):
- Self-reference does NOT modify state
- Summary is coarse, not precise
- Self-tags are temporary, not permanent definitions
- Multiple tags can exist simultaneously
- Presence/absence of tags does NOT branch logic
- No hardcoded values, coefficients, or thresholds

Usage::

    from psyche.self_reference import (
        execute_self_reference,
        apply_self_tags_to_bias,
    )

    # Execute self-reference loop (read-only)
    self_ref_state = execute_self_reference(
        psyche_state, responsibility_state, short_term_memory, dynamics_state
    )

    # Apply tags to decision bias
    modified_bias = apply_self_tags_to_bias(decision_bias, self_ref_state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class SelfTagCategory(Enum):
    """Categories for self-reference tags."""
    EMOTION = "emotion"
    FEAR = "fear"
    RESPONSIBILITY = "responsibility"
    MEMORY = "memory"
    TENDENCY = "tendency"


@dataclass
class SelfTag:
    """
    A single self-reference tag representing an aspect of current state.

    Tags are temporary markers, not permanent definitions.
    """

    # Tag identification
    category: SelfTagCategory
    label: str  # Coarse label (e.g., "high_fear", "negative_mood")

    # Raw value that was summarized (for reference, not for logic)
    source_value: float = 0.0

    # Weight for this tag (configurable, not hardcoded meaning)
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "label": self.label,
            "source_value": self.source_value,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfTag":
        return cls(
            category=SelfTagCategory(data.get("category", "emotion")),
            label=data.get("label", ""),
            source_value=data.get("source_value", 0.0),
            weight=data.get("weight", 1.0),
        )


@dataclass
class SelfReferenceConfig:
    """
    Configuration for self-reference behavior.

    All parameters are externally configurable.
    No hardcoded thresholds or meanings.
    """

    # Scale factors for different tag categories
    category_scales: dict[str, float] = field(default_factory=lambda: {
        "emotion": 1.0,
        "fear": 1.0,
        "responsibility": 1.0,
        "memory": 1.0,
        "tendency": 1.0,
    })

    # Global scale for all self-reference influence
    global_scale: float = 1.0

    # Custom summarization function (injectable)
    summarize_function: Optional[Callable[[dict], list[SelfTag]]] = None

    # Custom tag generation function (injectable)
    tag_generator: Optional[Callable[[dict], list[SelfTag]]] = None


@dataclass
class SelfReferenceState:
    """
    Current state of self-reference loop.

    Holds active self-tags and configuration.
    Does NOT hold or modify the actual internal state.
    """

    # Active self-reference tags
    tags: list[SelfTag] = field(default_factory=list)

    # Configuration
    config: SelfReferenceConfig = field(default_factory=SelfReferenceConfig)

    # Turn counter for circulation tracking
    reference_count: int = 0

    def get_tags_by_category(self, category: SelfTagCategory) -> list[SelfTag]:
        """Get all tags of a specific category."""
        return [t for t in self.tags if t.category == category]

    def get_tag_labels(self) -> list[str]:
        """Get all tag labels."""
        return [t.label for t in self.tags]

    def has_tag(self, label: str) -> bool:
        """Check if a tag with given label exists."""
        return any(t.label == label for t in self.tags)

    def get_summary(self) -> dict[str, Any]:
        """Get summary for diagnostics."""
        return {
            "tag_count": len(self.tags),
            "categories": list(set(t.category.value for t in self.tags)),
            "labels": self.get_tag_labels(),
            "reference_count": self.reference_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": [t.to_dict() for t in self.tags],
            "reference_count": self.reference_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Optional[SelfReferenceConfig] = None) -> "SelfReferenceState":
        tags = [SelfTag.from_dict(t) for t in data.get("tags", [])]
        return cls(
            tags=tags,
            config=config or SelfReferenceConfig(),
            reference_count=data.get("reference_count", 0),
        )


# ── Step 1: Acquire Self-Reference Targets ──────────────────────


def acquire_self_reference_targets(
    psyche_state: Any = None,
    responsibility_state: Any = None,
    short_term_memory: Any = None,
    dynamics_state: Any = None,
) -> dict[str, Any]:
    """
    Acquire internal state for self-reference (read-only).

    This function ONLY reads state, never modifies it.
    Returns a dict of raw values for summarization.

    Args:
        psyche_state: Current PsycheState (optional)
        responsibility_state: Current ResponsibilityState (optional)
        short_term_memory: Current ShortTermMemory (optional)
        dynamics_state: Current DynamicsState (optional)

    Returns:
        Dict of acquired state values for summarization
    """
    targets: dict[str, Any] = {}

    # Acquire from PsycheState (read-only)
    if psyche_state is not None:
        targets["emotions"] = {}
        if hasattr(psyche_state, "emotions"):
            targets["emotions"] = psyche_state.emotions.as_dict()

        if hasattr(psyche_state, "mood"):
            targets["mood_valence"] = psyche_state.mood.valence
            targets["mood_arousal"] = psyche_state.mood.arousal

        if hasattr(psyche_state, "fear_level"):
            targets["fear_level"] = psyche_state.fear_level

        if hasattr(psyche_state, "fear_index") and psyche_state.fear_index:
            targets["fear_index"] = {
                "identity_risk": psyche_state.fear_index.identity_risk,
                "attachment_risk": psyche_state.fear_index.attachment_risk,
                "continuity_risk": psyche_state.fear_index.continuity_risk,
                "projection_risk": psyche_state.fear_index.projection_risk,
            }

        if hasattr(psyche_state, "drives"):
            targets["drives"] = psyche_state.drives.as_dict()

    # Acquire from ResponsibilityState (read-only)
    if responsibility_state is not None:
        targets["responsibility"] = {
            "total_weight": getattr(responsibility_state, "total_weight", 0.0),
            "accumulated_harm": getattr(responsibility_state, "accumulated_harm", 0.0),
            "accumulated_confidence": getattr(responsibility_state, "accumulated_confidence", 0.0),
            "pending_decisions": getattr(responsibility_state, "pending_decisions", 0),
        }

    # Acquire from ShortTermMemory (read-only)
    if short_term_memory is not None:
        entry_count = len(getattr(short_term_memory, "entries", []))
        continuity = getattr(short_term_memory, "context_continuity_score", 0.0)

        # Compute residue intensity without modifying memory
        residue_intensity = 0.0
        if hasattr(short_term_memory, "get_unprocessed_residue"):
            for entry in short_term_memory.get_unprocessed_residue():
                residue_intensity += entry.residue_weight * entry.raw_intensity

        targets["short_term_memory"] = {
            "entry_count": entry_count,
            "continuity": continuity,
            "residue_intensity": residue_intensity,
        }

    # Acquire from DynamicsState (read-only)
    if dynamics_state is not None:
        targets["dynamics"] = {
            "phase": dynamics_state.phase.value if hasattr(dynamics_state, "phase") else "normal",
            "peak_emotion": getattr(dynamics_state, "peak_emotion", ""),
            "accumulated_intensity": getattr(dynamics_state, "accumulated_intensity", 0.0),
        }

    return targets


# ── Step 2: Summarize State (Coarse Compression) ────────────────


def summarize_state(
    targets: dict[str, Any],
    config: Optional[SelfReferenceConfig] = None,
) -> dict[str, Any]:
    """
    Summarize acquired state into coarse representation.

    This is a compression step, not an evaluation.
    No normalization or optimization.

    Args:
        targets: Acquired state from acquire_self_reference_targets()
        config: Optional configuration with custom summarize_function

    Returns:
        Coarse summary dict for tag generation
    """
    cfg = config or SelfReferenceConfig()

    # Use custom function if provided
    if cfg.summarize_function is not None:
        return cfg.summarize_function(targets)

    # Default coarse summarization (structure only, no interpretation)
    summary: dict[str, Any] = {}

    # Emotion summary
    emotions = targets.get("emotions", {})
    if emotions:
        max_emotion = max(emotions.items(), key=lambda x: x[1]) if emotions else ("", 0.0)
        summary["dominant_emotion"] = max_emotion[0]
        summary["dominant_emotion_value"] = max_emotion[1]
        summary["emotion_sum"] = sum(emotions.values())

    # Mood summary
    if "mood_valence" in targets:
        summary["mood_valence"] = targets["mood_valence"]
    if "mood_arousal" in targets:
        summary["mood_arousal"] = targets["mood_arousal"]

    # Fear summary
    if "fear_level" in targets:
        summary["fear_level"] = targets["fear_level"]
    if "fear_index" in targets:
        summary["fear_index"] = targets["fear_index"]

    # Responsibility summary
    if "responsibility" in targets:
        summary["responsibility"] = targets["responsibility"]

    # Short-term memory summary
    if "short_term_memory" in targets:
        summary["stm"] = targets["short_term_memory"]

    # Dynamics summary
    if "dynamics" in targets:
        summary["dynamics"] = targets["dynamics"]

    # Drives summary
    if "drives" in targets:
        drives = targets["drives"]
        if drives:
            max_drive = max(drives.items(), key=lambda x: x[1]) if drives else ("", 0.0)
            summary["dominant_drive"] = max_drive[0]
            summary["dominant_drive_value"] = max_drive[1]

    return summary


# ── Step 3: Generate Self-Tags ──────────────────────────────────


def generate_self_tags(
    summary: dict[str, Any],
    config: Optional[SelfReferenceConfig] = None,
) -> list[SelfTag]:
    """
    Generate self-reference tags from summary.

    Tags are temporary markers, not permanent definitions.
    Multiple tags can be generated from one summary.

    Args:
        summary: Coarse summary from summarize_state()
        config: Optional configuration with custom tag_generator

    Returns:
        List of SelfTag instances
    """
    cfg = config or SelfReferenceConfig()

    # Use custom function if provided
    if cfg.tag_generator is not None:
        return cfg.tag_generator(summary)

    tags: list[SelfTag] = []

    # Generate emotion-based tags
    if "dominant_emotion" in summary:
        tags.append(SelfTag(
            category=SelfTagCategory.EMOTION,
            label=f"dominant_{summary['dominant_emotion']}",
            source_value=summary.get("dominant_emotion_value", 0.0),
            weight=cfg.category_scales.get("emotion", 1.0) * cfg.global_scale,
        ))

    # Generate mood-based tags
    if "mood_valence" in summary:
        valence = summary["mood_valence"]
        if valence > 0:
            label = "positive_mood"
        elif valence < 0:
            label = "negative_mood"
        else:
            label = "neutral_mood"
        tags.append(SelfTag(
            category=SelfTagCategory.EMOTION,
            label=label,
            source_value=valence,
            weight=cfg.category_scales.get("emotion", 1.0) * cfg.global_scale,
        ))

    # Generate fear-based tags
    if "fear_level" in summary:
        fear = summary["fear_level"]
        tags.append(SelfTag(
            category=SelfTagCategory.FEAR,
            label="fear_present" if fear > 0 else "fear_absent",
            source_value=fear,
            weight=cfg.category_scales.get("fear", 1.0) * cfg.global_scale,
        ))

    # Generate responsibility-based tags
    if "responsibility" in summary:
        resp = summary["responsibility"]
        if resp.get("total_weight", 0) > 0:
            tags.append(SelfTag(
                category=SelfTagCategory.RESPONSIBILITY,
                label="responsibility_present",
                source_value=resp.get("total_weight", 0.0),
                weight=cfg.category_scales.get("responsibility", 1.0) * cfg.global_scale,
            ))
        if resp.get("accumulated_harm", 0) > 0:
            tags.append(SelfTag(
                category=SelfTagCategory.RESPONSIBILITY,
                label="harm_accumulated",
                source_value=resp.get("accumulated_harm", 0.0),
                weight=cfg.category_scales.get("responsibility", 1.0) * cfg.global_scale,
            ))

    # Generate memory-based tags
    if "stm" in summary:
        stm = summary["stm"]
        if stm.get("entry_count", 0) > 0:
            tags.append(SelfTag(
                category=SelfTagCategory.MEMORY,
                label="memory_active",
                source_value=float(stm.get("entry_count", 0)),
                weight=cfg.category_scales.get("memory", 1.0) * cfg.global_scale,
            ))
        if stm.get("residue_intensity", 0) > 0:
            tags.append(SelfTag(
                category=SelfTagCategory.MEMORY,
                label="residue_present",
                source_value=stm.get("residue_intensity", 0.0),
                weight=cfg.category_scales.get("memory", 1.0) * cfg.global_scale,
            ))

    # Generate dynamics-based tags
    if "dynamics" in summary:
        dyn = summary["dynamics"]
        phase = dyn.get("phase", "normal")
        if phase != "normal":
            tags.append(SelfTag(
                category=SelfTagCategory.TENDENCY,
                label=f"dynamics_{phase}",
                source_value=dyn.get("accumulated_intensity", 0.0),
                weight=cfg.category_scales.get("tendency", 1.0) * cfg.global_scale,
            ))

    # Generate drive-based tags
    if "dominant_drive" in summary:
        tags.append(SelfTag(
            category=SelfTagCategory.TENDENCY,
            label=f"drive_{summary['dominant_drive']}",
            source_value=summary.get("dominant_drive_value", 0.0),
            weight=cfg.category_scales.get("tendency", 1.0) * cfg.global_scale,
        ))

    return tags


# ── Step 4 & 5: Execute Self-Reference Loop ─────────────────────


def execute_self_reference(
    psyche_state: Any = None,
    responsibility_state: Any = None,
    short_term_memory: Any = None,
    dynamics_state: Any = None,
    previous_state: Optional[SelfReferenceState] = None,
    config: Optional[SelfReferenceConfig] = None,
) -> SelfReferenceState:
    """
    Execute one iteration of the self-reference loop.

    This function:
    1. Acquires internal state (read-only)
    2. Summarizes to coarse representation
    3. Generates self-tags
    4. Returns new SelfReferenceState

    Does NOT modify any input state.

    Args:
        psyche_state: Current PsycheState
        responsibility_state: Current ResponsibilityState
        short_term_memory: Current ShortTermMemory
        dynamics_state: Current DynamicsState
        previous_state: Previous SelfReferenceState (for circulation tracking)
        config: Self-reference configuration

    Returns:
        New SelfReferenceState with updated tags
    """
    cfg = config or SelfReferenceConfig()

    # Step 1: Acquire targets (read-only)
    targets = acquire_self_reference_targets(
        psyche_state=psyche_state,
        responsibility_state=responsibility_state,
        short_term_memory=short_term_memory,
        dynamics_state=dynamics_state,
    )

    # Step 2: Summarize
    summary = summarize_state(targets, cfg)

    # Step 3: Generate tags
    tags = generate_self_tags(summary, cfg)

    # Step 5: Update circulation counter
    reference_count = 0
    if previous_state is not None:
        reference_count = previous_state.reference_count + 1

    return SelfReferenceState(
        tags=tags,
        config=cfg,
        reference_count=reference_count,
    )


# ── Step 4: Self-Reference Result Usage ─────────────────────────


def apply_self_tags_to_bias(
    decision_bias: Any,
    self_ref_state: SelfReferenceState,
) -> Any:
    """
    Apply self-reference tags to decision bias.

    This is a MODIFIER, not a replacement.
    Does NOT branch logic based on tag presence/absence.

    Args:
        decision_bias: DecisionBias instance to modify
        self_ref_state: Current SelfReferenceState

    Returns:
        Modified DecisionBias (or same if no modification needed)
    """
    if decision_bias is None or not self_ref_state.tags:
        return decision_bias

    # Import here to avoid circular dependency
    from .decision_bias import DecisionBias

    if not isinstance(decision_bias, DecisionBias):
        return decision_bias

    # Compute adjustment from self-tags
    # This is additive, not replacement
    valence_adjustment = 0.0
    intensity_adjustment = 0.0

    for tag in self_ref_state.tags:
        # Negative mood tag → slight valence adjustment
        if tag.label == "negative_mood":
            valence_adjustment += tag.source_value * tag.weight * 0.1

        # Positive mood tag → slight valence adjustment
        elif tag.label == "positive_mood":
            valence_adjustment += tag.source_value * tag.weight * 0.1

        # Fear present → slight intensity
        elif tag.label == "fear_present":
            intensity_adjustment += tag.source_value * tag.weight * 0.1

        # Responsibility present → slight intensity
        elif tag.label == "responsibility_present":
            intensity_adjustment += tag.source_value * tag.weight * 0.05

    # Create new bias with adjustments (immutable pattern)
    return DecisionBias(
        emotion_biases=decision_bias.emotion_biases.copy(),
        intent_biases=decision_bias.intent_biases.copy(),
        valence_bias=decision_bias.valence_bias + valence_adjustment,
        residue_intensity=decision_bias.residue_intensity + intensity_adjustment,
        dynamics_phase=decision_bias.dynamics_phase,
        peak_boost=decision_bias.peak_boost,
        rebound_dampening=decision_bias.rebound_dampening,
        peak_emotion=decision_bias.peak_emotion,
        continuity=decision_bias.continuity,
        config=decision_bias.config,
    )


def get_self_reference_summary(self_ref_state: SelfReferenceState) -> str:
    """Get human-readable summary of self-reference state."""
    if not self_ref_state.tags:
        return "SELF-REF: no tags"

    labels = [t.label for t in self_ref_state.tags]
    return f"SELF-REF: {', '.join(labels)} (count={self_ref_state.reference_count})"


# ── Convenience Functions ───────────────────────────────────────


def create_self_reference_state(
    config: Optional[SelfReferenceConfig] = None,
) -> SelfReferenceState:
    """Create initial empty self-reference state."""
    return SelfReferenceState(config=config or SelfReferenceConfig())
