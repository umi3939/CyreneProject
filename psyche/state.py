"""
psyche/state.py - Core psychological state models (Pydantic).

All internal state is immutable. ``PsycheState`` is the root aggregate
holding emotions, drives, mood, optional pillar states, and fear_index.

Required methods: ``decay``, ``clamp_values``, ``to_dict``, ``from_dict``.

Usage::

    state = PsycheState()
    decayed = state.decay(delta_seconds=5.0)
    clamped = decayed.clamp_values()
    d = clamped.to_dict()
    restored = PsycheState.from_dict(d)
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .pillars import (
    AttachmentState,
    ContinuityState,
    FearIndex,
    IdentityState,
    ProjectionState,
)

# ── Emotion Vector (7-dim) ────────────────────────────────────

class EmotionVector(BaseModel):
    """Seven-dimensional emotion space.  Each value ranges 0.0–1.0."""

    joy: float = Field(default=0.0, ge=0.0, le=1.0)
    anger: float = Field(default=0.0, ge=0.0, le=1.0)
    sorrow: float = Field(default=0.0, ge=0.0, le=1.0)
    fear: float = Field(default=0.0, ge=0.0, le=1.0)
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    love: float = Field(default=0.0, ge=0.0, le=1.0)
    fun: float = Field(default=0.0, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()

    def to_loss_5d(self) -> dict[str, float]:
        """Convert 7-dim → design_loss 5-dim (joy,sad,fear,anger,calm)."""
        calm = max(0.0, 1.0 - max(self.joy, self.sorrow, self.fear, self.anger))
        return {
            "joy": self.joy,
            "sad": self.sorrow,
            "fear": self.fear,
            "anger": self.anger,
            "calm": calm,
        }


# ── Drive Vector (3-dim) ──────────────────────────────────────

class DriveVector(BaseModel):
    """Three motivational drives.  Each value ranges 0.0–1.0."""

    social: float = Field(default=0.5, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    expression: float = Field(default=0.5, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


# ── Mood ──────────────────────────────────────────────────────

class Mood(BaseModel):
    """Affective tone: valence (-1..1) and arousal (0..1)."""

    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.3, ge=0.0, le=1.0)

    @property
    def valence_label(self) -> str:
        """Return a label describing the valence."""
        if self.valence > 0.5:
            return "positive_high"
        if self.valence > 0.2:
            return "positive"
        if self.valence > -0.2:
            return "neutral"
        if self.valence > -0.5:
            return "negative"
        return "negative_high"


# ── Percept ───────────────────────────────────────────────────

class Percept(BaseModel):
    """Structured interpretation of one input stimulus."""

    text: str = ""
    meaning: str = ""
    emotion: str = "neutral"
    intent: str = "unknown"
    topics: list[str] = Field(default_factory=list)
    sentiment: float = 0.0
    emotion_valence: float = Field(default=0.0, ge=-1.0, le=1.0)


# ── Constants ─────────────────────────────────────────────────

DECAY_RATE: float = 0.95  # per-second exponential decay


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── PsycheState (root aggregate) ──────────────────────────────

class PsycheState(BaseModel):
    """Complete psychological state of the AI persona.

    Immutable: all mutation methods return a new instance.
    """

    emotions: EmotionVector = Field(default_factory=EmotionVector)
    drives: DriveVector = Field(default_factory=DriveVector)
    mood: Mood = Field(default_factory=Mood)

    # Loss-fear pillar states (Optional for backward compat)
    identity: Optional[IdentityState] = None
    attachment: Optional[AttachmentState] = None
    continuity: Optional[ContinuityState] = None
    projection: Optional[ProjectionState] = None
    fear_index: Optional[FearIndex] = None

    loss_aversion: float = 0.3
    last_updated: str = "2026-01-01T00:00:00"

    model_config = {"arbitrary_types_allowed": True}

    # ── required methods ──────────────────────────────────────

    def decay(self, delta_seconds: float) -> "PsycheState":
        """Return new state with emotions decayed over *delta_seconds*.

        Each emotion decays toward 0 at ``DECAY_RATE ** dt``.
        ``love`` decays more slowly (rate ** 0.5) for lasting bonds.
        """
        dt = min(delta_seconds, 3600.0)
        factor = DECAY_RATE ** dt
        slow_factor = DECAY_RATE ** (dt * 0.5)
        emo = self.emotions.as_dict()
        new_emo: dict[str, float] = {}
        for k, v in emo.items():
            if k == "love":
                new_emo[k] = _clamp(v * slow_factor)
            else:
                new_emo[k] = _clamp(v * factor)
        return self.model_copy(update={"emotions": EmotionVector(**new_emo)})

    def clamp_values(self) -> "PsycheState":
        """Return new state with all values clamped to valid ranges."""
        emo = {k: _clamp(v) for k, v in self.emotions.as_dict().items()}
        drv = {k: _clamp(v) for k, v in self.drives.as_dict().items()}
        mood = Mood(
            valence=_clamp(self.mood.valence, -1.0, 1.0),
            arousal=_clamp(self.mood.arousal, 0.0, 1.0),
        )
        fi_val = _clamp(self.fear_index.value if self.fear_index else 0.0)
        return self.model_copy(update={
            "emotions": EmotionVector(**emo),
            "drives": DriveVector(**drv),
            "mood": mood,
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON persistence."""
        if self.fear_index is not None:
            fi = self.fear_index
            fear_data: Any = {
                "identity_risk": round(fi.identity_risk, 4),
                "attachment_risk": round(fi.attachment_risk, 4),
                "continuity_risk": round(fi.continuity_risk, 4),
                "projection_risk": round(fi.projection_risk, 4),
                "value": round(fi.value, 4),
            }
        else:
            fear_data = 0.0
        d: dict[str, Any] = {
            "emotions": self.emotions.as_dict(),
            "drives": self.drives.as_dict(),
            "mood": {"valence": self.mood.valence, "arousal": self.mood.arousal},
            "fear_index": fear_data,
            "loss_aversion": self.loss_aversion,
            "last_updated": self.last_updated,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PsycheState":
        """Reconstruct a ``PsycheState`` from a plain dict."""
        emotions = EmotionVector(**d.get("emotions", {}))
        drives = DriveVector(**d.get("drives", {}))
        mood_raw = d.get("mood", {})
        if isinstance(mood_raw, dict):
            mood = Mood(**mood_raw)
        else:
            mood = Mood(valence=float(mood_raw), arousal=0.3)
        fear_val = d.get("fear_index", 0.0)
        if isinstance(fear_val, dict):
            fear_index = FearIndex(
                identity_risk=fear_val.get("identity_risk", 0.0),
                attachment_risk=fear_val.get("attachment_risk", 0.0),
                continuity_risk=fear_val.get("continuity_risk", 0.0),
                projection_risk=fear_val.get("projection_risk", 0.0),
            )
        elif isinstance(fear_val, (int, float)):
            fear_index = FearIndex(
                identity_risk=0.0, attachment_risk=0.0,
                continuity_risk=0.0, projection_risk=0.0,
            )
        else:
            fear_index = None
        return cls(
            emotions=emotions,
            drives=drives,
            mood=mood,
            fear_index=fear_index,
            loss_aversion=d.get("loss_aversion", 0.3),
            last_updated=d.get("last_updated", "2026-01-01T00:00:00"),
        )

    # ── convenience properties ────────────────────────────────

    @property
    def dominant_emotion(self) -> str:
        d = self.emotions.as_dict()
        return max(d, key=d.get)

    @property
    def dominant_emotion_value(self) -> float:
        return getattr(self.emotions, self.dominant_emotion)

    @property
    def fear_level(self) -> float:
        if self.fear_index is None:
            return 0.0
        return self.fear_index.value

    @property
    def dominant_fear(self) -> str:
        if self.fear_index is None:
            return ""
        return self.fear_index.dominant_fear

    def fear_summary(self) -> str:
        if self.fear_index is None:
            return "喪失恐怖: なし"
        fi = self.fear_index
        parts = [
            f"identity={fi.identity_risk:.2f}",
            f"attachment={fi.attachment_risk:.2f}",
            f"continuity={fi.continuity_risk:.2f}",
            f"projection={fi.projection_risk:.2f}",
        ]
        return (
            f"喪失恐怖: level={fi.value:.2f}, "
            f"dominant={fi.dominant_fear}, "
            + ", ".join(parts)
        )

    def emotion_summary(self) -> str:
        emo = self.emotions.as_dict()
        active = {k: v for k, v in emo.items() if v > 0.1}
        if not active:
            return "calm / neutral"
        parts = [f"{k}={v:.2f}" for k, v in sorted(active.items(), key=lambda x: -x[1])]
        return ", ".join(parts)
