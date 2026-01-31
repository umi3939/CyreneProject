"""
psyche/fear.py - Fear Index Computation & Emotion/Drive Boost

Computes the composite FearIndex from pillar risks and applies
fear-driven boosts to emotions and drives.
"""

from __future__ import annotations

from .pillars import FearIndex
from .state import DriveVector, EmotionVector


def compute_fear_index(
    identity_risk: float = 0.0,
    attachment_risk: float = 0.0,
    continuity_risk: float = 0.0,
    projection_risk: float = 0.0,
) -> FearIndex:
    """Build a FearIndex from the four pillar risk values.

    Weights: identity=0.3, attachment=0.3, continuity=0.2, projection=0.2
    (Weighting is embedded in FearIndex.value property.)
    """
    return FearIndex(
        identity_risk=identity_risk,
        attachment_risk=attachment_risk,
        continuity_risk=continuity_risk,
        projection_risk=projection_risk,
    )


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def fear_emotion_boost(
    fear_index: FearIndex, emotions: EmotionVector
) -> EmotionVector:
    """Amplify fear and sorrow emotions when fear_index.value > 0.2.

    Returns a new EmotionVector (immutable pattern).
    """
    if fear_index.value <= 0.2:
        return emotions

    emo = emotions.as_dict()
    boost = (fear_index.value - 0.2) * 0.5  # scale: 0..0.4 range
    emo["fear"] = _clamp01(emo["fear"] + boost)
    emo["sorrow"] = _clamp01(emo["sorrow"] + boost * 0.5)
    return EmotionVector(**emo)


def fear_drive_boost(
    fear_index: FearIndex, drives: DriveVector
) -> DriveVector:
    """Boost drives in response to specific pillar fears.

    - attachment_risk high → social drive increases
    - identity_risk high → expression drive increases

    Returns a new DriveVector (immutable pattern).
    """
    drv = drives.model_dump()

    if fear_index.attachment_risk > 0.3:
        drv["social"] = _clamp01(
            drv["social"] + (fear_index.attachment_risk - 0.3) * 0.3
        )

    if fear_index.identity_risk > 0.3:
        drv["expression"] = _clamp01(
            drv["expression"] + (fear_index.identity_risk - 0.3) * 0.3
        )

    return DriveVector(**drv)
