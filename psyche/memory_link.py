"""
psyche/memory_link.py - Mood-Congruent Memory Recall (LOCAL ONLY).

Wraps ``MemoryManager.recall()`` and re-scores results based on
current emotional state.  **No LLM calls** — pure local re-ranking.

Usage::

    recalled = recall_by_mood(percept, state, memory_mgr, top_k=3)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .state import Percept, PsycheState

if TYPE_CHECKING:
    pass  # MemoryManager duck-typed

logger = logging.getLogger(__name__)

# Keywords associated with positive/negative valence for scoring
# English keywords use word-boundary matching to avoid false positives
# (e.g., "fun" matching "function"). Japanese keywords use substring matching
# since Japanese text doesn't have word boundaries.
_POSITIVE_KEYWORDS_JA = ["嬉しい", "楽しい", "好き", "幸せ", "笑", "感謝", "素敵"]
_NEGATIVE_KEYWORDS_JA = ["悲しい", "怒り", "辛い", "寂しい", "怖い", "不安", "嫌"]
_POSITIVE_RE = re.compile(
    "|".join(_POSITIVE_KEYWORDS_JA)
    + r"|\bhappy\b|\bjoy\b|\blove\b|\bfun\b|\bgood\b|\bgreat\b|\bthank\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    "|".join(_NEGATIVE_KEYWORDS_JA)
    + r"|\bsad\b|\bangry\b|\bfear\b|\bbad\b|\bhate\b|\bworry\b|\bpain\b",
    re.IGNORECASE,
)


def _estimate_memory_valence(mem: dict) -> float:
    """Estimate a memory's emotional valence from text content.  Returns -1.0 to 1.0."""
    text = (mem.get("summary", "") + " " + " ".join(mem.get("keywords", []))).lower()
    pos_hits = len(_POSITIVE_RE.findall(text))
    neg_hits = len(_NEGATIVE_RE.findall(text))
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / total


async def recall_with_mood(
    percept: Percept,
    state: PsycheState,
    memory: Any,
    top_k: int = 3,
) -> list[dict]:
    """Recall memories with mood-congruent bias.  LOCAL ONLY — no LLM.

    1. Fetches 2× candidates from ``memory.recall()``.
    2. Re-ranks by mood congruence (matching valence direction).
    3. Applies attachment partner bonus.

    Args:
        percept: Current input (used as query text).
        state: Current PsycheState.
        memory: Object with ``.recall(query, top_k=N)`` async method.
        top_k: Number of results to return.

    Returns:
        List of memory dicts, re-ranked by mood congruence.
    """
    fetch_k = max(top_k * 2, 6)
    candidates = await memory.recall(percept.text, top_k=fetch_k)

    if not candidates:
        return []

    current_valence = state.mood.valence

    # Collect top partner names for attachment bonus
    top_partners: list[str] = []
    if state.attachment is not None:
        from .attachment_manager import get_top_partners
        top_partners = [name for name, _score in get_top_partners(state.attachment, n=5)]

    scored: list[tuple[float, dict]] = []
    for idx, mem in enumerate(candidates):
        mem_valence = _estimate_memory_valence(mem)
        # Mood-congruent bonus
        congruence_bonus = current_valence * mem_valence * 2.0
        # Base score from position
        base_score = fetch_k - idx
        final_score = base_score + congruence_bonus

        # Attachment bonus
        if top_partners:
            mem_text = (mem.get("summary", "") + " " + " ".join(mem.get("keywords", []))).lower()
            for partner in top_partners:
                if partner.lower() in mem_text:
                    final_score += 2.0
                    break

        scored.append((final_score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:top_k]]
