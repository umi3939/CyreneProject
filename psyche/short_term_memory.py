"""
psyche/short_term_memory.py - Short-Term Emotional Memory System

Holds recent emotional stimuli and context independently from long-term memory.
Designed for the short-term emotional loop where:
- Stimuli accumulate when context is continuous
- Memory decays naturally over time
- Unprocessed residue influences emotion updates

All values, weights, and thresholds are kept as configurable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time


@dataclass
class StimulusEntry:
    """A single emotional stimulus entry in short-term memory."""

    # Content identifiers
    source_text: str = ""
    topics: list[str] = field(default_factory=list)
    emotion_label: str = "neutral"
    intent: str = "unknown"

    # Intensity values (not interpreted, just stored)
    raw_intensity: float = 0.0
    valence: float = 0.0

    # Timestamp for decay calculation
    timestamp: float = field(default_factory=time.time)

    # Residue weight (starts at 1.0, decays over time)
    residue_weight: float = 1.0

    # Whether this entry has been processed in emotion update
    processed: bool = False


@dataclass
class ShortTermMemory:
    """
    Short-term memory state holding recent emotional stimuli.

    This is kept separate from PsycheState to maintain clear separation
    between "memory" and "current emotional state".

    Design principles:
    - No hardcoded interpretation of values
    - All scale factors are configurable
    - Context continuity is detected but not interpreted
    - Decay is applied at pipeline end, once per turn
    """

    # Recent stimuli entries (oldest first)
    entries: list[StimulusEntry] = field(default_factory=list)

    # Maximum number of entries to retain
    max_entries: int = 10

    # Timestamp of last update (for decay calculation)
    last_update_time: float = field(default_factory=time.time)

    # Accumulated context (for continuity detection)
    current_context_topics: list[str] = field(default_factory=list)
    context_continuity_score: float = 0.0

    # ── Configurable Parameters (no hardcoded values) ──────────

    # Scale factors as dict for easy external configuration
    scale_factors: dict[str, float] = field(default_factory=lambda: {
        "residue_influence": 1.0,      # How much residue affects emotions
        "accumulation_rate": 1.0,      # How fast context accumulates
        "continuity_threshold": 0.0,   # Threshold for continuity (configured externally)
        "decay_base": 1.0,             # Base decay rate (configured externally)
    })

    # ── Core Operations ────────────────────────────────────────

    def add_stimulus(
        self,
        source_text: str,
        topics: list[str],
        emotion_label: str,
        intent: str,
        raw_intensity: float,
        valence: float,
    ) -> "ShortTermMemory":
        """
        Add a new stimulus entry and return updated memory.

        Does NOT interpret values - just stores them.
        Continuity detection happens separately.
        """
        new_entry = StimulusEntry(
            source_text=source_text,
            topics=topics,
            emotion_label=emotion_label,
            intent=intent,
            raw_intensity=raw_intensity,
            valence=valence,
            timestamp=time.time(),
            residue_weight=1.0,
            processed=False,
        )

        new_entries = self.entries.copy()
        new_entries.append(new_entry)

        # Trim to max entries (remove oldest)
        if len(new_entries) > self.max_entries:
            new_entries = new_entries[-self.max_entries:]

        return ShortTermMemory(
            entries=new_entries,
            max_entries=self.max_entries,
            last_update_time=time.time(),
            current_context_topics=self.current_context_topics,
            context_continuity_score=self.context_continuity_score,
            scale_factors=self.scale_factors.copy(),
        )

    def get_unprocessed_residue(self) -> list[StimulusEntry]:
        """
        Return all unprocessed stimulus entries.

        These are available for the emotion update phase to reference
        as "unprocessed emotional residue".
        """
        return [e for e in self.entries if not e.processed]

    def get_weighted_residue_summary(self) -> dict[str, float]:
        """
        Compute weighted sum of residue by emotion label.

        Returns raw aggregation - interpretation is left to caller.
        """
        summary: dict[str, float] = {}
        for entry in self.get_unprocessed_residue():
            label = entry.emotion_label
            weight = entry.residue_weight * entry.raw_intensity
            summary[label] = summary.get(label, 0.0) + weight
        return summary

    def mark_processed(self, entry_indices: Optional[list[int]] = None) -> "ShortTermMemory":
        """
        Mark entries as processed.

        If entry_indices is None, marks all entries as processed.
        """
        new_entries = []
        for i, entry in enumerate(self.entries):
            if entry_indices is None or i in entry_indices:
                new_entry = StimulusEntry(
                    source_text=entry.source_text,
                    topics=entry.topics,
                    emotion_label=entry.emotion_label,
                    intent=entry.intent,
                    raw_intensity=entry.raw_intensity,
                    valence=entry.valence,
                    timestamp=entry.timestamp,
                    residue_weight=entry.residue_weight,
                    processed=True,
                )
                new_entries.append(new_entry)
            else:
                new_entries.append(entry)

        return ShortTermMemory(
            entries=new_entries,
            max_entries=self.max_entries,
            last_update_time=self.last_update_time,
            current_context_topics=self.current_context_topics,
            context_continuity_score=self.context_continuity_score,
            scale_factors=self.scale_factors.copy(),
        )

    # ── Context Continuity ─────────────────────────────────────

    def compute_context_overlap(self, new_topics: list[str]) -> float:
        """
        Compute topic overlap between current context and new input.

        Returns raw overlap score (0.0 to 1.0).
        Interpretation of "continuous" vs "discontinuous" is left to caller.
        """
        if not self.current_context_topics or not new_topics:
            return 0.0

        current_set = set(self.current_context_topics)
        new_set = set(new_topics)

        if not current_set or not new_set:
            return 0.0

        intersection = current_set & new_set
        union = current_set | new_set

        return len(intersection) / len(union) if union else 0.0

    def update_context(
        self,
        new_topics: list[str],
        is_continuous: bool,
    ) -> "ShortTermMemory":
        """
        Update context topics based on continuity decision.

        If continuous: accumulate topics
        If not continuous: replace topics
        """
        if is_continuous:
            # Accumulate (union of topics)
            merged = list(set(self.current_context_topics) | set(new_topics))
        else:
            # Replace
            merged = new_topics.copy()

        # Compute new continuity score
        overlap = self.compute_context_overlap(new_topics)

        return ShortTermMemory(
            entries=self.entries,
            max_entries=self.max_entries,
            last_update_time=self.last_update_time,
            current_context_topics=merged,
            context_continuity_score=overlap,
            scale_factors=self.scale_factors.copy(),
        )

    def reset_on_discontinuity(self) -> "ShortTermMemory":
        """
        Reset short-term memory on context discontinuity.

        Clears accumulated context but preserves scale factors.
        """
        return ShortTermMemory(
            entries=[],
            max_entries=self.max_entries,
            last_update_time=time.time(),
            current_context_topics=[],
            context_continuity_score=0.0,
            scale_factors=self.scale_factors.copy(),
        )

    # ── Decay ──────────────────────────────────────────────────

    def apply_decay(
        self,
        decay_function: Optional[Callable[[float, float], float]] = None,
        current_time: Optional[float] = None,
    ) -> "ShortTermMemory":
        """
        Apply time-based decay to all entries.

        Args:
            decay_function: Optional custom decay function.
                            Signature: (residue_weight, elapsed_seconds) -> new_weight
                            If None, uses a simple exponential decay.
            current_time: Optional current timestamp. Uses time.time() if None.

        Returns:
            New ShortTermMemory with decayed entries.
        """
        now = current_time if current_time is not None else time.time()
        elapsed = now - self.last_update_time

        if decay_function is None:
            # Default decay: exponential with configurable base
            decay_base = self.scale_factors.get("decay_base", 0.95)
            def decay_function(weight: float, dt: float) -> float:
                return weight * (decay_base ** dt)

        new_entries = []
        for entry in self.entries:
            entry_elapsed = now - entry.timestamp
            new_weight = decay_function(entry.residue_weight, entry_elapsed)

            # Keep entry if weight is still significant
            # (threshold is configurable, default very low to avoid hardcoding)
            min_weight = self.scale_factors.get("min_residue_weight", 0.001)
            if new_weight >= min_weight:
                new_entry = StimulusEntry(
                    source_text=entry.source_text,
                    topics=entry.topics,
                    emotion_label=entry.emotion_label,
                    intent=entry.intent,
                    raw_intensity=entry.raw_intensity,
                    valence=entry.valence,
                    timestamp=entry.timestamp,
                    residue_weight=new_weight,
                    processed=entry.processed,
                )
                new_entries.append(new_entry)

        return ShortTermMemory(
            entries=new_entries,
            max_entries=self.max_entries,
            last_update_time=now,
            current_context_topics=self.current_context_topics,
            context_continuity_score=self.context_continuity_score,
            scale_factors=self.scale_factors.copy(),
        )

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for persistence if needed."""
        return {
            "entries": [
                {
                    "source_text": e.source_text,
                    "topics": e.topics,
                    "emotion_label": e.emotion_label,
                    "intent": e.intent,
                    "raw_intensity": e.raw_intensity,
                    "valence": e.valence,
                    "timestamp": e.timestamp,
                    "residue_weight": e.residue_weight,
                    "processed": e.processed,
                }
                for e in self.entries
            ],
            "max_entries": self.max_entries,
            "last_update_time": self.last_update_time,
            "current_context_topics": self.current_context_topics,
            "context_continuity_score": self.context_continuity_score,
            "scale_factors": self.scale_factors,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShortTermMemory":
        """Deserialize from dict."""
        entries = [
            StimulusEntry(**e) for e in d.get("entries", [])
        ]
        return cls(
            entries=entries,
            max_entries=d.get("max_entries", 10),
            last_update_time=d.get("last_update_time", time.time()),
            current_context_topics=d.get("current_context_topics", []),
            context_continuity_score=d.get("context_continuity_score", 0.0),
            scale_factors=d.get("scale_factors", {}),
        )


# ── Residue Influence Interface ────────────────────────────────

@dataclass
class ResidueInfluence:
    """
    Computed influence from short-term memory residue.

    This is an intermediate structure passed to emotion update.
    Values here are raw aggregations - interpretation happens in reaction.py.
    """

    # Weighted emotion influences by label
    emotion_influences: dict[str, float] = field(default_factory=dict)

    # Overall residue intensity
    total_intensity: float = 0.0

    # Context continuity score
    continuity: float = 0.0

    # Scale factor to apply (configurable)
    scale: float = 1.0


def compute_residue_influence(
    memory: ShortTermMemory,
    scale_factor: Optional[float] = None,
) -> ResidueInfluence:
    """
    Compute the residue influence from short-term memory.

    This function aggregates unprocessed stimuli into an influence structure
    that can be passed to the emotion update phase.

    The scale_factor allows external control over influence strength
    without modifying the core logic.
    """
    residue_summary = memory.get_weighted_residue_summary()

    total = sum(residue_summary.values())

    scale = scale_factor if scale_factor is not None else memory.scale_factors.get("residue_influence", 1.0)

    return ResidueInfluence(
        emotion_influences=residue_summary,
        total_intensity=total,
        continuity=memory.context_continuity_score,
        scale=scale,
    )
