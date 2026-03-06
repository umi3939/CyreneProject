"""
psyche/introspection_trace.py - Introspective Trace Generation (内省ログ生成)

Creates structured trace logs that link internal states to decisions
for later analysis, without modifying any state.

Design principles (from design_introspection_trace.md):
- READ-ONLY access to all states (observation only)
- Generated post-hoc, does not interfere with real-time decisions
- Factors are "possible influences" not "definitive reasons"
- Logs are append-only, never modified
- No self-evaluation or right/wrong labeling

Usage::

    from psyche.introspection_trace import (
        IntrospectionSystem,
        TraceLog,
        generate_trace,
    )

    # Create system
    system = IntrospectionSystem()

    # Generate trace after a decision
    trace = system.generate_trace(
        emotion_state=emotion_vector,
        responsibility_state=responsibility_state,
        value_orientation=orientation,
        decision_outcome={"policy_label": "沈黙する", "_score": 0.7},
    )

    # Get human-readable summary
    print(trace.to_readable())

    # Get JSON-compatible dict
    data = trace.to_dict()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import time
import uuid


# ── Enums ───────────────────────────────────────────────────────────


class InfluenceDirection(Enum):
    """Direction of influence on the outcome."""
    POSITIVE = "positive"      # Contributed toward this outcome
    NEGATIVE = "negative"      # Contributed against this outcome
    NEUTRAL = "neutral"        # Present but unclear direction
    AMPLIFYING = "amplifying"  # Amplified other factors
    DAMPENING = "dampening"    # Dampened other factors


class FactorCategory(Enum):
    """Category of contributing factor."""
    EMOTION = "emotion"
    RESPONSIBILITY = "responsibility"
    VALUE_ORIENTATION = "value_orientation"
    CONTEXT = "context"
    MEMORY = "memory"
    FEAR = "fear"
    TONE = "tone"
    OTHER = "other"


class OutcomeType(Enum):
    """Type of decision outcome."""
    SPEECH = "speech"          # Normal speech response
    SILENCE = "silence"        # Chose to be silent
    HESITATION = "hesitation"  # Hesitated before responding
    LIGHT_TONE = "light_tone"  # Light/playful response
    SERIOUS_TONE = "serious_tone"  # Serious response
    UNKNOWN = "unknown"


# ── Contributing Factor ─────────────────────────────────────────────


@dataclass
class ContributingFactor:
    """
    A factor that may have contributed to the outcome.

    NOTE: This is a "possible influence" not a "definitive reason".
    We express influence in terms of degree and direction, not certainty.
    """

    # Unique identifier for this factor
    factor_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Category of the factor
    category: FactorCategory = FactorCategory.OTHER

    # Name/label of the factor (e.g., "fear", "dim_a", "caution_level")
    name: str = ""

    # The observed value at trace time
    observed_value: float = 0.0

    # Threshold that was considered significant
    threshold: float = 0.0

    # Direction of influence on outcome
    direction: InfluenceDirection = InfluenceDirection.NEUTRAL

    # Estimated contribution strength (0.0 to 1.0)
    # This is NOT a certainty, just an estimate
    contribution_strength: float = 0.0

    # Human-readable description (optional)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-compatible dict."""
        return {
            "factor_id": self.factor_id,
            "category": self.category.value,
            "name": self.name,
            "observed_value": round(self.observed_value, 4),
            "threshold": round(self.threshold, 4),
            "direction": self.direction.value,
            "contribution_strength": round(self.contribution_strength, 4),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContributingFactor":
        """Create from dict."""
        return cls(
            factor_id=data.get("factor_id", str(uuid.uuid4())[:8]),
            category=FactorCategory(data.get("category", "other")),
            name=data.get("name", ""),
            observed_value=data.get("observed_value", 0.0),
            threshold=data.get("threshold", 0.0),
            direction=InfluenceDirection(data.get("direction", "neutral")),
            contribution_strength=data.get("contribution_strength", 0.0),
            description=data.get("description", ""),
        )


# ── State Snapshots ─────────────────────────────────────────────────


@dataclass
class EmotionSnapshot:
    """Read-only snapshot of emotion state."""
    joy: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0
    surprise: float = 0.0

    # Derived metrics
    dominant_emotion: str = ""
    emotion_intensity: float = 0.0  # Sum of all emotions
    valence: float = 0.0  # Positive vs negative balance

    def to_dict(self) -> dict[str, Any]:
        return {
            "joy": round(self.joy, 4),
            "anger": round(self.anger, 4),
            "sadness": round(self.sadness, 4),
            "fear": round(self.fear, 4),
            "disgust": round(self.disgust, 4),
            "surprise": round(self.surprise, 4),
            "dominant_emotion": self.dominant_emotion,
            "emotion_intensity": round(self.emotion_intensity, 4),
            "valence": round(self.valence, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionSnapshot":
        return cls(
            joy=data.get("joy", 0.0),
            anger=data.get("anger", 0.0),
            sadness=data.get("sadness", 0.0),
            fear=data.get("fear", 0.0),
            disgust=data.get("disgust", 0.0),
            surprise=data.get("surprise", 0.0),
            dominant_emotion=data.get("dominant_emotion", ""),
            emotion_intensity=data.get("emotion_intensity", 0.0),
            valence=data.get("valence", 0.0),
        )


@dataclass
class ResponsibilitySnapshot:
    """Read-only snapshot of responsibility state."""
    total_weight: float = 0.0
    active_count: int = 0
    avg_distance: float = 0.0
    caution_level: float = 0.0
    anxiety_baseline: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_weight": round(self.total_weight, 4),
            "active_count": self.active_count,
            "avg_distance": round(self.avg_distance, 4),
            "caution_level": round(self.caution_level, 4),
            "anxiety_baseline": round(self.anxiety_baseline, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponsibilitySnapshot":
        return cls(
            total_weight=data.get("total_weight", 0.0),
            active_count=data.get("active_count", 0),
            avg_distance=data.get("avg_distance", 0.0),
            caution_level=data.get("caution_level", 0.0),
            anxiety_baseline=data.get("anxiety_baseline", 0.0),
        )


@dataclass
class ValueOrientationSnapshot:
    """Read-only snapshot of value orientation."""
    dim_a: float = 0.0
    dim_b: float = 0.0
    dim_c: float = 0.0
    dim_d: float = 0.0
    dim_e: float = 0.0
    overall_stability: float = 0.0
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim_a": round(self.dim_a, 4),
            "dim_b": round(self.dim_b, 4),
            "dim_c": round(self.dim_c, 4),
            "dim_d": round(self.dim_d, 4),
            "dim_e": round(self.dim_e, 4),
            "overall_stability": round(self.overall_stability, 4),
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValueOrientationSnapshot":
        return cls(
            dim_a=data.get("dim_a", 0.0),
            dim_b=data.get("dim_b", 0.0),
            dim_c=data.get("dim_c", 0.0),
            dim_d=data.get("dim_d", 0.0),
            dim_e=data.get("dim_e", 0.0),
            overall_stability=data.get("overall_stability", 0.0),
            update_count=data.get("update_count", 0),
        )


@dataclass
class DecisionSnapshot:
    """Read-only snapshot of the decision outcome."""
    policy_label: str = ""
    score: float = 0.0
    outcome_type: OutcomeType = OutcomeType.UNKNOWN
    tone: str = "neutral"
    is_silence: bool = False
    candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_label": self.policy_label,
            "score": round(self.score, 4),
            "outcome_type": self.outcome_type.value,
            "tone": self.tone,
            "is_silence": self.is_silence,
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionSnapshot":
        return cls(
            policy_label=data.get("policy_label", ""),
            score=data.get("score", 0.0),
            outcome_type=OutcomeType(data.get("outcome_type", "unknown")),
            tone=data.get("tone", "neutral"),
            is_silence=data.get("is_silence", False),
            candidate_count=data.get("candidate_count", 0),
        )


# ── Trace Log ───────────────────────────────────────────────────────


@dataclass
class TraceLog:
    """
    Complete introspection trace log.

    Links internal states to decision outcomes for later analysis.
    This is structured data (JSON-compatible) not just text.
    """

    # Unique trace identifier
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Timestamp of trace generation
    timestamp: float = field(default_factory=time.time)

    # Generation/sequence number (for ordering)
    generation: int = 0

    # State snapshots (READ-ONLY copies)
    emotion_snapshot: EmotionSnapshot = field(default_factory=EmotionSnapshot)
    responsibility_snapshot: ResponsibilitySnapshot = field(default_factory=ResponsibilitySnapshot)
    value_orientation_snapshot: ValueOrientationSnapshot = field(default_factory=ValueOrientationSnapshot)
    decision_snapshot: DecisionSnapshot = field(default_factory=DecisionSnapshot)

    # Extracted contributing factors
    contributing_factors: list[ContributingFactor] = field(default_factory=list)

    # Summary of factor categories present
    factor_summary: dict[str, int] = field(default_factory=dict)

    # Optional context information
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-compatible dict for storage/analysis."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "generation": self.generation,
            "emotion_snapshot": self.emotion_snapshot.to_dict(),
            "responsibility_snapshot": self.responsibility_snapshot.to_dict(),
            "value_orientation_snapshot": self.value_orientation_snapshot.to_dict(),
            "decision_snapshot": self.decision_snapshot.to_dict(),
            "contributing_factors": [f.to_dict() for f in self.contributing_factors],
            "factor_summary": self.factor_summary.copy(),
            "context": self.context.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceLog":
        """Create from dict."""
        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            generation=data.get("generation", 0),
            emotion_snapshot=EmotionSnapshot.from_dict(data.get("emotion_snapshot", {})),
            responsibility_snapshot=ResponsibilitySnapshot.from_dict(data.get("responsibility_snapshot", {})),
            value_orientation_snapshot=ValueOrientationSnapshot.from_dict(data.get("value_orientation_snapshot", {})),
            decision_snapshot=DecisionSnapshot.from_dict(data.get("decision_snapshot", {})),
            contributing_factors=[
                ContributingFactor.from_dict(f) for f in data.get("contributing_factors", [])
            ],
            factor_summary=data.get("factor_summary", {}),
            context=data.get("context", {}),
        )

    def to_readable(self) -> str:
        """
        Generate human-readable summary.

        NOTE: Uses "may have contributed" language, not definitive claims.
        """
        lines = []
        lines.append(f"=== Introspection Trace [{self.trace_id[:8]}] ===")
        lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}")
        lines.append(f"Generation: {self.generation}")
        lines.append("")

        # Decision
        ds = self.decision_snapshot
        lines.append(f"Decision: {ds.policy_label} (score={ds.score:.2f}, type={ds.outcome_type.value})")
        if ds.is_silence:
            lines.append("  -> Chose silence/hesitation")
        lines.append("")

        # Contributing factors
        if self.contributing_factors:
            lines.append("Possible Contributing Factors:")
            for factor in self.contributing_factors:
                direction_symbol = {
                    InfluenceDirection.POSITIVE: "+",
                    InfluenceDirection.NEGATIVE: "-",
                    InfluenceDirection.NEUTRAL: "~",
                    InfluenceDirection.AMPLIFYING: "^",
                    InfluenceDirection.DAMPENING: "v",
                }.get(factor.direction, "?")

                lines.append(
                    f"  [{direction_symbol}] {factor.category.value}/{factor.name}: "
                    f"{factor.observed_value:.2f} (strength={factor.contribution_strength:.2f})"
                )
                if factor.description:
                    lines.append(f"      -> {factor.description}")
        else:
            lines.append("No significant contributing factors identified.")

        lines.append("")
        lines.append("Note: Factors are possibilities, not definitive causes.")

        return "\n".join(lines)

    def get_top_factors(self, n: int = 3) -> list[ContributingFactor]:
        """Get top N factors by contribution strength."""
        sorted_factors = sorted(
            self.contributing_factors,
            key=lambda f: f.contribution_strength,
            reverse=True,
        )
        return sorted_factors[:n]

    def has_factor_category(self, category: FactorCategory) -> bool:
        """Check if any factor of given category is present."""
        return any(f.category == category for f in self.contributing_factors)


# ── Introspection System ────────────────────────────────────────────


@dataclass
class IntrospectionConfig:
    """Configuration for introspection trace generation."""

    # Thresholds for considering factors significant
    emotion_threshold: float = 0.2
    responsibility_threshold: float = 0.3
    value_orientation_threshold: float = 0.15
    fear_threshold: float = 0.3

    # Whether to include weak factors
    include_weak_factors: bool = False
    weak_factor_threshold: float = 0.1

    # Maximum factors to include
    max_factors: int = 10


class IntrospectionSystem:
    """
    System for generating introspection traces.

    IMPORTANT: This system has READ-ONLY access to all states.
    It does NOT modify emotions, decisions, or any other state.
    """

    _TRACE_HISTORY_LIMIT: int = 200

    def __init__(self, config: Optional[IntrospectionConfig] = None):
        self.config = config or IntrospectionConfig()
        self._generation_counter = 0
        self._trace_history: deque[TraceLog] = deque(maxlen=self._TRACE_HISTORY_LIMIT)

    def generate_trace(
        self,
        emotion_state: Optional[Any] = None,
        responsibility_state: Optional[Any] = None,
        value_orientation: Optional[Any] = None,
        decision_outcome: Optional[dict[str, Any]] = None,
        fear_index: Optional[Any] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> TraceLog:
        """
        Generate an introspection trace from current states.

        All state access is READ-ONLY. No modifications are made.

        Args:
            emotion_state: EmotionVector or similar
            responsibility_state: ResponsibilityInfluence or similar
            value_orientation: ValueOrientation
            decision_outcome: Dict with policy_label, _score, etc.
            fear_index: FearIndex if available
            context: Additional context information

        Returns:
            TraceLog with snapshots and contributing factors
        """
        self._generation_counter += 1

        # Create READ-ONLY snapshots
        emotion_snapshot = self._snapshot_emotion(emotion_state)
        responsibility_snapshot = self._snapshot_responsibility(responsibility_state)
        value_snapshot = self._snapshot_value_orientation(value_orientation)
        decision_snapshot = self._snapshot_decision(decision_outcome)

        # Extract contributing factors (READ-ONLY analysis)
        factors = self._extract_factors(
            emotion_snapshot,
            responsibility_snapshot,
            value_snapshot,
            decision_snapshot,
            fear_index,
        )

        # Build factor summary
        factor_summary: dict[str, int] = {}
        for factor in factors:
            cat = factor.category.value
            factor_summary[cat] = factor_summary.get(cat, 0) + 1

        # Create trace
        trace = TraceLog(
            generation=self._generation_counter,
            emotion_snapshot=emotion_snapshot,
            responsibility_snapshot=responsibility_snapshot,
            value_orientation_snapshot=value_snapshot,
            decision_snapshot=decision_snapshot,
            contributing_factors=factors,
            factor_summary=factor_summary,
            context=context or {},
        )

        # Store in history (append-only)
        self._trace_history.append(trace)

        return trace

    def _snapshot_emotion(self, emotion_state: Optional[Any]) -> EmotionSnapshot:
        """Create read-only snapshot of emotion state."""
        if emotion_state is None:
            return EmotionSnapshot()

        joy = getattr(emotion_state, "joy", 0.0)
        anger = getattr(emotion_state, "anger", 0.0)
        sadness = getattr(emotion_state, "sadness", 0.0)
        fear = getattr(emotion_state, "fear", 0.0)
        disgust = getattr(emotion_state, "disgust", 0.0)
        surprise = getattr(emotion_state, "surprise", 0.0)

        # Calculate derived metrics
        emotions = {"joy": joy, "anger": anger, "sadness": sadness,
                   "fear": fear, "disgust": disgust, "surprise": surprise}
        dominant = max(emotions, key=emotions.get) if any(emotions.values()) else ""
        intensity = sum(emotions.values())
        valence = joy - (anger + sadness + fear + disgust) / 4

        return EmotionSnapshot(
            joy=joy,
            anger=anger,
            sadness=sadness,
            fear=fear,
            disgust=disgust,
            surprise=surprise,
            dominant_emotion=dominant,
            emotion_intensity=intensity,
            valence=valence,
        )

    def _snapshot_responsibility(self, responsibility_state: Optional[Any]) -> ResponsibilitySnapshot:
        """Create read-only snapshot of responsibility state."""
        if responsibility_state is None:
            return ResponsibilitySnapshot()

        return ResponsibilitySnapshot(
            total_weight=getattr(responsibility_state, "total_weight", 0.0),
            active_count=getattr(responsibility_state, "active_count", 0),
            avg_distance=getattr(responsibility_state, "avg_distance", 0.0),
            caution_level=getattr(responsibility_state, "caution", 0.0),
            anxiety_baseline=getattr(responsibility_state, "anxiety_baseline", 0.0),
        )

    def _snapshot_value_orientation(self, value_orientation: Optional[Any]) -> ValueOrientationSnapshot:
        """Create read-only snapshot of value orientation."""
        if value_orientation is None:
            return ValueOrientationSnapshot()

        return ValueOrientationSnapshot(
            dim_a=getattr(value_orientation, "dim_a", 0.0),
            dim_b=getattr(value_orientation, "dim_b", 0.0),
            dim_c=getattr(value_orientation, "dim_c", 0.0),
            dim_d=getattr(value_orientation, "dim_d", 0.0),
            dim_e=getattr(value_orientation, "dim_e", 0.0),
            overall_stability=getattr(value_orientation, "get_overall_stability", lambda: 0.0)()
            if callable(getattr(value_orientation, "get_overall_stability", None))
            else 0.0,
            update_count=getattr(value_orientation, "update_count", 0),
        )

    def _snapshot_decision(self, decision_outcome: Optional[dict[str, Any]]) -> DecisionSnapshot:
        """Create read-only snapshot of decision."""
        if decision_outcome is None:
            return DecisionSnapshot()

        policy_label = decision_outcome.get("policy_label", "")
        is_silence = decision_outcome.get("_is_silence", False)
        tone = decision_outcome.get("_tone", "neutral")

        # Determine outcome type
        if is_silence:
            outcome_type = OutcomeType.SILENCE
        elif tone == "light":
            outcome_type = OutcomeType.LIGHT_TONE
        elif tone == "serious":
            outcome_type = OutcomeType.SERIOUS_TONE
        elif policy_label:
            outcome_type = OutcomeType.SPEECH
        else:
            outcome_type = OutcomeType.UNKNOWN

        return DecisionSnapshot(
            policy_label=policy_label,
            score=decision_outcome.get("_score", 0.0),
            outcome_type=outcome_type,
            tone=tone,
            is_silence=is_silence,
            candidate_count=decision_outcome.get("_candidate_count", 0),
        )

    def _extract_factors(
        self,
        emotion: EmotionSnapshot,
        responsibility: ResponsibilitySnapshot,
        value: ValueOrientationSnapshot,
        decision: DecisionSnapshot,
        fear_index: Optional[Any],
    ) -> list[ContributingFactor]:
        """
        Extract contributing factors from snapshots.

        NOTE: These are POSSIBLE influences, not definitive causes.
        """
        factors: list[ContributingFactor] = []
        cfg = self.config

        # ── Emotion factors ──
        emotion_map = {
            "joy": emotion.joy,
            "anger": emotion.anger,
            "sadness": emotion.sadness,
            "fear": emotion.fear,
            "disgust": emotion.disgust,
            "surprise": emotion.surprise,
        }

        for emotion_name, value_float in emotion_map.items():
            if value_float >= cfg.emotion_threshold:
                direction = self._infer_emotion_direction(emotion_name, decision)
                factors.append(ContributingFactor(
                    category=FactorCategory.EMOTION,
                    name=emotion_name,
                    observed_value=value_float,
                    threshold=cfg.emotion_threshold,
                    direction=direction,
                    contribution_strength=min(1.0, value_float / 0.5),
                    description=self._describe_emotion_influence(emotion_name, value_float, decision),
                ))

        # ── Fear factor ──
        if fear_index is not None:
            fear_value = getattr(fear_index, "value", 0.0)
            if fear_value >= cfg.fear_threshold:
                factors.append(ContributingFactor(
                    category=FactorCategory.FEAR,
                    name="fear_index",
                    observed_value=fear_value,
                    threshold=cfg.fear_threshold,
                    direction=self._infer_fear_direction(decision),
                    contribution_strength=min(1.0, fear_value),
                    description=self._describe_fear_influence(fear_value, decision),
                ))

        # ── Responsibility factors ──
        if responsibility.total_weight >= cfg.responsibility_threshold:
            factors.append(ContributingFactor(
                category=FactorCategory.RESPONSIBILITY,
                name="total_weight",
                observed_value=responsibility.total_weight,
                threshold=cfg.responsibility_threshold,
                direction=InfluenceDirection.DAMPENING if decision.is_silence else InfluenceDirection.NEUTRAL,
                contribution_strength=min(1.0, responsibility.total_weight),
                description=f"Responsibility burden ({responsibility.total_weight:.2f}) may have influenced caution",
            ))

        if responsibility.caution_level >= 0.3:
            factors.append(ContributingFactor(
                category=FactorCategory.RESPONSIBILITY,
                name="caution_level",
                observed_value=responsibility.caution_level,
                threshold=0.3,
                direction=InfluenceDirection.POSITIVE if decision.is_silence else InfluenceDirection.DAMPENING,
                contribution_strength=responsibility.caution_level,
                description=f"Caution level ({responsibility.caution_level:.2f}) may have moderated response",
            ))

        # ── Value orientation factors ──
        value_dims = {
            "dim_a": value.dim_a,
            "dim_b": value.dim_b,
            "dim_c": value.dim_c,
            "dim_d": value.dim_d,
            "dim_e": value.dim_e,
        }

        for dim_name, dim_value in value_dims.items():
            if abs(dim_value) >= cfg.value_orientation_threshold:
                direction = InfluenceDirection.POSITIVE if dim_value > 0 else InfluenceDirection.NEGATIVE
                factors.append(ContributingFactor(
                    category=FactorCategory.VALUE_ORIENTATION,
                    name=dim_name,
                    observed_value=dim_value,
                    threshold=cfg.value_orientation_threshold,
                    direction=direction,
                    contribution_strength=abs(dim_value),
                    description=f"Value dimension {dim_name} ({dim_value:+.2f}) may have biased selection",
                ))

        # ── Tone factor ──
        if decision.tone and decision.tone != "neutral":
            factors.append(ContributingFactor(
                category=FactorCategory.TONE,
                name="selected_tone",
                observed_value=1.0 if decision.tone == "light" else -1.0 if decision.tone == "serious" else 0.0,
                threshold=0.0,
                direction=InfluenceDirection.POSITIVE,
                contribution_strength=0.5,
                description=f"Tone '{decision.tone}' was selected for response",
            ))

        # Sort by contribution strength and limit
        factors.sort(key=lambda f: f.contribution_strength, reverse=True)
        return factors[:cfg.max_factors]

    def _infer_emotion_direction(
        self,
        emotion_name: str,
        decision: DecisionSnapshot,
    ) -> InfluenceDirection:
        """Infer how an emotion may have influenced the decision."""
        if decision.is_silence:
            # Silence is often associated with fear, sadness, or caution
            if emotion_name in ["fear", "sadness"]:
                return InfluenceDirection.POSITIVE
            elif emotion_name == "joy":
                return InfluenceDirection.NEGATIVE
        else:
            if emotion_name == "joy":
                return InfluenceDirection.POSITIVE
            elif emotion_name in ["fear", "sadness"]:
                return InfluenceDirection.DAMPENING

        return InfluenceDirection.NEUTRAL

    def _infer_fear_direction(self, decision: DecisionSnapshot) -> InfluenceDirection:
        """Infer how fear may have influenced the decision."""
        if decision.is_silence:
            return InfluenceDirection.POSITIVE  # Fear may have contributed to silence
        elif decision.tone == "serious":
            return InfluenceDirection.POSITIVE  # Fear may have made tone serious
        else:
            return InfluenceDirection.DAMPENING  # Fear may have dampened boldness

    def _describe_emotion_influence(
        self,
        emotion_name: str,
        value: float,
        decision: DecisionSnapshot,
    ) -> str:
        """Generate description of emotion influence."""
        intensity = "High" if value > 0.5 else "Moderate"

        if decision.is_silence:
            if emotion_name == "fear":
                return f"{intensity} fear ({value:.2f}) may have contributed to choosing silence"
            elif emotion_name == "sadness":
                return f"{intensity} sadness ({value:.2f}) may have contributed to hesitation"
            else:
                return f"{intensity} {emotion_name} ({value:.2f}) was present during silence decision"
        else:
            return f"{intensity} {emotion_name} ({value:.2f}) may have influenced response style"

    def _describe_fear_influence(self, value: float, decision: DecisionSnapshot) -> str:
        """Generate description of fear influence."""
        if decision.is_silence:
            return f"Fear index ({value:.2f}) may have contributed to choosing silence"
        elif decision.tone == "serious":
            return f"Fear index ({value:.2f}) may have influenced serious tone selection"
        else:
            return f"Fear index ({value:.2f}) may have moderated response boldness"

    def get_history(self) -> list[TraceLog]:
        """Get trace history (read-only copy)."""
        return list(self._trace_history)

    def get_recent_traces(self, n: int = 5) -> list[TraceLog]:
        """Get N most recent traces."""
        history = list(self._trace_history)
        return history[-n:]

    def clear_history(self) -> None:
        """Clear trace history."""
        self._trace_history.clear()

    def get_generation_count(self) -> int:
        """Get total number of traces generated."""
        return self._generation_counter


# ── Convenience Functions ───────────────────────────────────────────


def generate_trace(
    emotion_state: Optional[Any] = None,
    responsibility_state: Optional[Any] = None,
    value_orientation: Optional[Any] = None,
    decision_outcome: Optional[dict[str, Any]] = None,
    fear_index: Optional[Any] = None,
    context: Optional[dict[str, Any]] = None,
    config: Optional[IntrospectionConfig] = None,
) -> TraceLog:
    """
    Convenience function to generate a single trace.

    Creates a temporary IntrospectionSystem and generates one trace.
    """
    system = IntrospectionSystem(config)
    return system.generate_trace(
        emotion_state=emotion_state,
        responsibility_state=responsibility_state,
        value_orientation=value_orientation,
        decision_outcome=decision_outcome,
        fear_index=fear_index,
        context=context,
    )


def create_introspection_system(
    config: Optional[IntrospectionConfig] = None,
) -> IntrospectionSystem:
    """Create a new introspection system."""
    return IntrospectionSystem(config)


def create_config(
    emotion_threshold: float = 0.2,
    responsibility_threshold: float = 0.3,
    max_factors: int = 10,
) -> IntrospectionConfig:
    """Create an introspection config with custom parameters."""
    return IntrospectionConfig(
        emotion_threshold=emotion_threshold,
        responsibility_threshold=responsibility_threshold,
        max_factors=max_factors,
    )


def get_trace_summary(trace: TraceLog) -> str:
    """Get a brief summary of a trace."""
    ds = trace.decision_snapshot
    top_factors = trace.get_top_factors(2)

    factor_str = ", ".join(f"{f.name}" for f in top_factors) if top_factors else "none"

    return (
        f"[{trace.trace_id[:8]}] {ds.policy_label or 'unknown'} "
        f"(type={ds.outcome_type.value}) <- {factor_str}"
    )


def traces_to_json(traces: list[TraceLog]) -> list[dict[str, Any]]:
    """Convert list of traces to JSON-compatible format."""
    return [t.to_dict() for t in traces]


def traces_from_json(data: list[dict[str, Any]]) -> list[TraceLog]:
    """Load traces from JSON-compatible format."""
    return [TraceLog.from_dict(d) for d in data]
