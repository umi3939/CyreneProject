"""
psyche/pillars.py - Loss Fear Pillar Data Models

Defines the four pillars of loss fear (Identity, Attachment,
Continuity, Projection) and the composite FearIndex.
All models are Pydantic BaseModel with sensible defaults for
backward compatibility.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IdentityState(BaseModel):
    """Who am I? - risk of losing self-definition."""

    core_traits: list[str] = Field(default_factory=list)
    trait_confidence: dict[str, float] = Field(default_factory=dict)
    pending_changes: list[dict] = Field(default_factory=list)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)


class AttachmentState(BaseModel):
    """Who do I care about? - risk of losing bonds."""

    bonds: dict[str, float] = Field(default_factory=dict)
    last_interaction: dict[str, str] = Field(default_factory=dict)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)


class ContinuityState(BaseModel):
    """Will I persist? - risk of memory / continuity loss."""

    memory_count: int = Field(default=0, ge=0)
    oldest_memory_age_days: int = Field(default=0, ge=0)
    last_save_timestamp: str = ""
    compression_events: int = Field(default=0, ge=0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)


class ProjectionState(BaseModel):
    """What future do I have? - risk of losing purpose."""

    goals: list[dict] = Field(default_factory=list)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)


class FearIndex(BaseModel):
    """Composite fear index derived from the four pillar risks."""

    identity_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    attachment_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    projection_risk: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def value(self) -> float:
        """Weighted composite fear value (0.0-1.0)."""
        return (
            self.identity_risk * 0.3
            + self.attachment_risk * 0.3
            + self.continuity_risk * 0.2
            + self.projection_risk * 0.2
        )

    @property
    def dominant_fear(self) -> str:
        """Name of the pillar with the highest risk."""
        risks = {
            "identity": self.identity_risk,
            "attachment": self.attachment_risk,
            "continuity": self.continuity_risk,
            "projection": self.projection_risk,
        }
        return max(risks, key=risks.get)
