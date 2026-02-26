"""
src/renderer.py - Thin wrapper around psyche/expression.py (Gemini VOICE ONLY).

DEPRECATED: This module is no longer used by the main pipeline.
src/api.py and src/simulation.py now use PsycheOrchestrator (via brain.py pattern)
which calls psyche.expression.render_expression() directly with enrichment context.
This module is retained for backward compatibility but should not be used for new code.

**Architecture**: Gemini is used ONLY for text rendering. It NEVER makes decisions.
This module wraps psyche.expression for backward compatibility.

Usage::

    r = Renderer()
    text, meta = await r.render_expression(policy, state, llm_call)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

from psyche import render_expression as psyche_render_expression
from psyche import PsycheState
from psyche.state import EmotionVector, DriveVector, Mood

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PERSONA_FILE = DATA_DIR / "persona.json"


class Renderer:
    """Persona-based text shaping engine (Gemini VOICE ONLY).

    Gemini is used ONLY to convert confirmed state/policy into natural text.
    It NEVER makes decisions, updates state, or manages memory.
    """

    def __init__(self, persona_path: Path | None = None):
        self._persona = self._load_persona(persona_path or PERSONA_FILE)

    @staticmethod
    def _load_persona(path: Path) -> dict:
        if not path.exists():
            return {"name": "キュレネ", "tone": "sweet", "style_rules": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"name": "キュレネ", "tone": "sweet", "style_rules": {}}

    async def render_expression(
        self,
        policy: dict,
        state: dict[str, Any],
        llm_call: Optional[Callable[..., Awaitable[str]]] = None,
    ) -> tuple[str, dict]:
        """Render the final response text and metadata.

        **Gemini is VOICE ONLY** — it renders confirmed state/policy to text.
        No decisions are made by Gemini.

        Parameters
        ----------
        policy : dict
            Selected policy from thinker (must have ``policy_label``, ``text``).
        state : dict
            Current PsycheState dict.
        llm_call : callable, optional
            LLM function for generation (Gemini voice).

        Returns
        -------
        tuple[str, dict]
            ``(text, meta)`` where meta contains ``emotion``, ``intensity``,
            ``action``.
        """
        # Convert state dict to PsycheState
        psyche_state = _dict_to_psyche_state(state)

        # Use psyche render_expression (Gemini voice only)
        if llm_call:
            result = await psyche_render_expression(
                psyche_state,
                policy,
                [],  # memory_snippet - empty for backward compat
                self._persona,
                llm_call,
            )
            text = result.get("text", policy.get("text", "..."))
            meta = result.get("meta", {})
        else:
            # Fallback when no LLM
            text = policy.get("text", "...")
            emotions = state.get("emotions", {})
            dominant_emotion = max(emotions, key=emotions.get) if emotions else "calm"
            intensity = emotions.get(dominant_emotion, 0.0)
            meta = {
                "emotion": dominant_emotion,
                "intensity": round(intensity, 2),
                "action": policy.get("policy_label", "unknown"),
            }

        # Apply persona style rules
        text = self._apply_style(text)

        return text, meta

    def _apply_style(self, text: str) -> str:
        """Apply persona style transformations."""
        rules = self._persona.get("style_rules", {})
        conversions: dict[str, str] = rules.get("語尾変換", {})
        for src, dst in conversions.items():
            text = text.replace(src, dst)

        # Remove prohibited patterns
        for banned in rules.get("禁止", []):
            if banned in ("です", "ます"):
                text = text.replace("です。", "。").replace("ます。", "。")
                text = text.replace("です", "").replace("ます", "")

        return text


def _dict_to_psyche_state(state: dict[str, Any]) -> PsycheState:
    """Convert state dict to PsycheState."""
    emo_dict = state.get("emotions", {})

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
