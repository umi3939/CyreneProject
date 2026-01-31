"""
src/thinker.py - Thin wrapper around psyche/thought.py (LOCAL ONLY).

**Architecture**: All thinking/policy logic is LOCAL. No LLM calls.
This module wraps psyche.thought for backward compatibility with
existing code that imports from src.thinker.

Usage::

    from src.thinker import generate_candidates, score_candidates

    candidates = generate_candidates(percept, state, recalled)
    best = score_candidates(candidates, state)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from psyche import (
    PsycheState,
    Percept,
    generate_thought_candidates,
    select_policy,
)
from psyche.state import EmotionVector, DriveVector, Mood

logger = logging.getLogger(__name__)

# Re-export policy labels for backward compat
POLICY_LABELS = [
    ("empathize", "共感する"),
    ("challenge", "挑発・からかう"),
    ("ask_question", "質問で広げる"),
    ("share_thoughts", "感想を述べる"),
    ("encourage", "励ます"),
    ("change_topic", "話題を変える"),
]


async def generate_candidates(
    user_input: str,
    emotions: dict[str, float],
    state: dict[str, Any],
    recalled: list[dict],
    llm_call: Callable[..., Awaitable[str]],
) -> list[dict]:
    """Generate thought candidates (LOCAL ONLY — llm_call is IGNORED).

    This function maintains the old signature for backward compatibility
    but internally uses psyche.thought which is purely local.

    Args:
        user_input: Raw user text (used to build Percept).
        emotions: Current emotions dict (5-dim or 7-dim).
        state: Current state dict.
        recalled: Retrieved memories.
        llm_call: IGNORED — kept for signature compat only.

    Returns:
        List of candidate policy dicts.
    """
    # Convert to PsycheState
    psyche_state = _dict_to_psyche_state(state, emotions)

    # Build a simple Percept from user_input
    percept = Percept(
        text=user_input,
        meaning=user_input,
        emotion="neutral",
        intent="unknown",
        emotion_valence=0.0,
    )

    # Call psyche LOCAL logic
    candidates = generate_thought_candidates(psyche_state, percept, recalled)

    # Map to old format for compatibility
    return [
        {
            "policy_label": c.get("policy_label", "empathize"),
            "rationale": c.get("rationale", ""),
            "expected_drive_change": c.get("expected_drive_change", {}),
            "text": c.get("text", "..."),
        }
        for c in candidates
    ]


def score_candidates(
    candidates: list[dict],
    identity: dict[str, Any],
    attachment: dict[str, Any],
    projection: dict[str, Any],
    state: dict[str, Any],
) -> dict:
    """Score candidates and return the best one (LOCAL ONLY).

    Args:
        candidates: List of candidate dicts from generate_candidates.
        identity: Identity state (for risk consideration).
        attachment: Attachment state (for risk consideration).
        projection: Projection state (for risk consideration).
        state: Current state dict.

    Returns:
        Best candidate dict.
    """
    if not candidates:
        return {
            "policy_label": "empathize",
            "rationale": "デフォルト",
            "expected_drive_change": {"social": -0.05, "curiosity": -0.02},
            "text": "...あなたの気持ち、わかるわ",
        }

    # Convert to PsycheState for scoring
    psyche_state = _dict_to_psyche_state(state)

    # Use psyche select_policy (already sorted by score)
    best = select_policy(candidates, psyche_state)
    return best


def _dict_to_psyche_state(
    state: dict[str, Any],
    emotions: dict[str, float] | None = None,
) -> PsycheState:
    """Convert state dict to PsycheState for internal use."""
    emo_dict = emotions or state.get("emotions", {})

    # Handle 5-dim → 7-dim conversion
    if "sad" in emo_dict and "sorrow" not in emo_dict:
        emo_dict = {
            "joy": emo_dict.get("joy", 0.0),
            "anger": emo_dict.get("anger", 0.0),
            "sorrow": emo_dict.get("sad", 0.0),
            "fear": emo_dict.get("fear", 0.0),
            "surprise": 0.0,
            "love": 0.0,
            "fun": 0.0,
        }

    emotions_vec = EmotionVector(**{
        k: emo_dict.get(k, 0.0) for k in ["joy", "anger", "sorrow", "fear", "surprise", "love", "fun"]
    })

    drives_dict = state.get("drives", {"social": 0.5, "curiosity": 0.5})
    drives_vec = DriveVector(
        social=drives_dict.get("social", 0.5),
        curiosity=drives_dict.get("curiosity", 0.5),
        expression=drives_dict.get("expression", 0.5),
    )

    mood_val = state.get("mood", 0.0)
    if isinstance(mood_val, dict):
        mood = Mood(valence=mood_val.get("valence", 0.0), arousal=mood_val.get("arousal", 0.3))
    else:
        mood = Mood(valence=float(mood_val), arousal=0.3)

    return PsycheState(
        emotions=emotions_vec,
        drives=drives_vec,
        mood=mood,
        loss_aversion=state.get("loss_aversion", 0.3),
    )
