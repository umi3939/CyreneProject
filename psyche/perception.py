"""
psyche/perception.py - Input Interpretation (auxiliary LLM).

Parses user text into a ``Percept`` using local heuristics first,
with optional LLM enrichment via ``llm_wrapper``.  Gemini is used
**only** for structured extraction — never for judgment or state changes.

Usage::

    percept = await parse_percept("こんにちは！", llm_fn, state)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional

from .state import Percept, PsycheState

logger = logging.getLogger(__name__)

# ── Heuristic keyword maps ─────────────────────────────────────

_EMOTION_KEYWORDS: dict[str, tuple[str, float]] = {
    "嬉しい": ("happy", 0.7), "楽しい": ("happy", 0.6), "好き": ("loving", 0.7),
    "幸せ": ("happy", 0.8), "ありがとう": ("happy", 0.5), "笑": ("happy", 0.4),
    "悲しい": ("sad", -0.6), "辛い": ("sad", -0.5), "寂しい": ("sad", -0.5),
    "怖い": ("scared", -0.6), "不安": ("scared", -0.4),
    "怒": ("angry", -0.5), "ムカ": ("angry", -0.5), "イライラ": ("angry", -0.5),
    "驚": ("surprised", 0.3), "びっくり": ("surprised", 0.3),
}

_INTENT_KEYWORDS: dict[str, str] = {
    "こんにちは": "greeting", "おはよう": "greeting", "やあ": "greeting",
    "？": "question", "?": "question",
    "教えて": "question", "何": "question",
    "ありがとう": "greeting", "バイバイ": "farewell", "さようなら": "farewell",
}


# ── Public API ─────────────────────────────────────────────────

async def parse_percept(
    user_text: str,
    llm_call_fn: Optional[Callable[..., Awaitable[str]]] = None,
    state: Optional[PsycheState] = None,
) -> Percept:
    """Parse user text into a Percept.

    1. Apply local heuristics for fast baseline.
    2. If *llm_call_fn* is available, enrich via LLM (auxiliary only).
    3. Merge results.

    Args:
        user_text: Raw user input.
        llm_call_fn: Optional async LLM function for enrichment.
        state: Optional current PsycheState (biases perception).

    Returns:
        Percept with structured interpretation.
    """
    # 1. Local heuristic baseline
    percept = _heuristic_parse(user_text)

    # 2. LLM enrichment (optional, auxiliary)
    if llm_call_fn:
        percept = await _llm_enrich(user_text, percept, llm_call_fn, state)

    return percept


# ── Local heuristic ────────────────────────────────────────────

def _heuristic_parse(text: str) -> Percept:
    """Fast local keyword-based parse.  No LLM."""
    emotion = "neutral"
    valence = 0.0
    intent = "unknown"
    topics: list[str] = []

    for kw, (emo, val) in _EMOTION_KEYWORDS.items():
        if kw in text:
            emotion = emo
            valence = val
            topics.append(kw)
            break

    for kw, nt in _INTENT_KEYWORDS.items():
        if kw in text:
            intent = nt
            break

    sentiment = valence  # simple mapping

    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent=intent,
        topics=topics,
        sentiment=sentiment,
        emotion_valence=max(-1.0, min(1.0, valence)),
    )


# ── LLM enrichment ────────────────────────────────────────────

async def _llm_enrich(
    text: str,
    baseline: Percept,
    llm_call_fn: Callable[..., Awaitable[str]],
    state: Optional[PsycheState],
) -> Percept:
    """Use LLM for richer extraction.  Falls back to baseline on failure."""
    emotion_context = ""
    if state:
        emotion_context = f"\n現在の感情状態: {state.emotion_summary()}\n気分: valence={state.mood.valence:.2f}"

    prompt = (
        f"以下のユーザー発話を解析してJSONで返してください。{emotion_context}\n\n"
        f"発話: 「{text}」\n\n"
        f"出力形式 (JSONのみ):\n"
        f'{{"meaning": "...", "emotion": "happy|sad|angry|surprised|scared|loving|teasing|neutral", '
        f'"intent": "greeting|question|sharing|request|joke|complaint|farewell|unknown", '
        f'"emotion_valence": float, "topics": ["..."]}}'
    )

    try:
        try:
            from src.llm_wrapper import PERCEPTION_SYSTEM_PROMPT, llm_call_with_system
            raw = await llm_call_with_system(PERCEPTION_SYSTEM_PROMPT, prompt)
        except ImportError:
            raw = await llm_call_fn(prompt)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict) or "emotion" not in data:
            return baseline

        return Percept(
            text=text,
            meaning=data.get("meaning", text),
            emotion=data.get("emotion", baseline.emotion),
            intent=data.get("intent", baseline.intent),
            topics=data.get("topics", baseline.topics),
            sentiment=float(data.get("emotion_valence", baseline.sentiment)),
            emotion_valence=max(-1.0, min(1.0, float(data.get("emotion_valence", baseline.emotion_valence)))),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("Perception LLM enrichment failed: %s", e)
        return baseline
