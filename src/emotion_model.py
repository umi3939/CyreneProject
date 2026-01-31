"""
src/emotion_model.py - Emotion vector computation with decay.

Converts raw events (text sentiment, keywords) into a 5-dimensional
emotion vector ``{joy, sad, fear, anger, calm}`` and applies time-based
decay so emotions fade naturally.

Usage::

    vec = event_to_emotion({"sentiment": 0.8, "keywords": ["嬉しい"]})
    decayed = apply_decay(vec, delta_seconds=5.0)
"""

from __future__ import annotations

import copy
import math

# Default emotion vector (all neutral)
NEUTRAL: dict[str, float] = {
    "joy": 0.0,
    "sad": 0.0,
    "fear": 0.0,
    "anger": 0.0,
    "calm": 0.5,
}

# Keyword → primary emotion mapping
_KEYWORD_MAP: dict[str, str] = {
    "嬉しい": "joy", "楽しい": "joy", "好き": "joy", "幸せ": "joy",
    "happy": "joy", "love": "joy", "fun": "joy",
    "悲しい": "sad", "辛い": "sad", "寂しい": "sad",
    "sad": "sad", "lonely": "sad", "pain": "sad",
    "怖い": "fear", "不安": "fear", "緊張": "fear",
    "fear": "fear", "scared": "fear", "worry": "fear",
    "怒り": "anger", "イライラ": "anger", "ムカつく": "anger",
    "angry": "anger", "hate": "anger", "annoyed": "anger",
}

DECAY_RATE = 0.93  # per-second exponential decay


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def event_to_emotion(
    event: dict,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    """Convert an event dict into a 5-dim emotion vector.

    Parameters
    ----------
    event : dict
        Must contain at least ``sentiment`` (float, -1..1).
        Optionally ``keywords`` (list[str]).
    base : dict, optional
        Starting vector to add deltas to.  Defaults to ``NEUTRAL``.

    Returns
    -------
    dict[str, float]
        Updated emotion vector.
    """
    vec = copy.deepcopy(base or NEUTRAL)
    sentiment: float = event.get("sentiment", 0.0)

    # Sentiment-driven broad adjustments
    if sentiment > 0:
        vec["joy"] = _clamp(vec["joy"] + sentiment * 0.3)
        vec["calm"] = _clamp(vec["calm"] + sentiment * 0.1)
    else:
        vec["sad"] = _clamp(vec["sad"] + abs(sentiment) * 0.2)
        vec["anger"] = _clamp(vec["anger"] + abs(sentiment) * 0.1)
        vec["fear"] = _clamp(vec["fear"] + abs(sentiment) * 0.1)
        vec["calm"] = _clamp(vec["calm"] - abs(sentiment) * 0.15)

    # Keyword-driven specific boosts
    for kw in event.get("keywords", []):
        target = _KEYWORD_MAP.get(kw)
        if target and target in vec:
            vec[target] = _clamp(vec[target] + 0.15)

    return vec


def apply_decay(
    vec: dict[str, float],
    delta_seconds: float = 1.0,
) -> dict[str, float]:
    """Decay emotion intensities toward neutral over time.

    ``calm`` drifts back toward 0.5; all others decay toward 0.

    Parameters
    ----------
    vec : dict
        Current emotion vector.
    delta_seconds : float
        Seconds elapsed since last update.

    Returns
    -------
    dict[str, float]
        Decayed emotion vector (new dict).
    """
    factor = DECAY_RATE ** delta_seconds
    out: dict[str, float] = {}
    for k, v in vec.items():
        if k == "calm":
            # Drift toward 0.5
            out[k] = _clamp(0.5 + (v - 0.5) * factor)
        else:
            out[k] = _clamp(v * factor)
    return out


def compute_leftover(
    vec: dict[str, float],
    threshold: float = 0.05,
) -> dict[str, float]:
    """Zero out emotions below *threshold* (emotional leftover cleanup)."""
    return {k: (v if v >= threshold else 0.0) for k, v in vec.items()}
