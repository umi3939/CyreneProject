"""
psyche/continuity_manager.py - Continuity Pillar Manager

Manages memory persistence, compression, and continuity risk.
All functions are pure where possible; memory_mgr interactions
are side-effectful but clearly scoped.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .pillars import ContinuityState

if TYPE_CHECKING:
    from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def maybe_save(
    event_text: str,
    response: str,
    state: ContinuityState,
    memory_mgr: "MemoryManager",
    importance: int = 3,
    is_attachment_event: bool = False,
) -> bool:
    """Decide whether to save an event to long-term memory.

    Saves when importance >= 3 or when the event involves attachment.

    Returns:
        True if saved, False otherwise.
    """
    should_save = importance >= 3 or is_attachment_event
    if should_save:
        logger.debug("Continuity: saving event to memory")
        # Actual save is delegated to memory_mgr by the caller
        return True
    return False


def compress_and_cleanup(
    memory_mgr: "MemoryManager",
    max_age_days: int = 90,
) -> int:
    """Compress old low-importance memories.

    Returns:
        Number of memories compressed.
    """
    # This is a stub; actual compression logic depends on MemoryManager API
    logger.debug(f"Continuity: compress_and_cleanup (max_age={max_age_days})")
    return 0


def calc_continuity_risk(
    memory_count: int = 0,
    recent_compressions: int = 0,
) -> float:
    """Compute continuity risk (0.0 - 1.0).

    High risk when:
    - Very few memories (< 5)
    - Many recent compression events (information loss)
    """
    # Few memories → high risk
    if memory_count < 5:
        count_risk = 0.6
    elif memory_count < 20:
        count_risk = 0.3
    else:
        count_risk = 0.1

    # Recent compressions → additional risk
    compression_risk = min(recent_compressions * 0.1, 0.4)

    return min(count_risk + compression_risk, 1.0)


def audit_memory_health(state: ContinuityState) -> dict:
    """Return a diagnostic dict of memory health indicators."""
    return {
        "memory_count": state.memory_count,
        "oldest_memory_age_days": state.oldest_memory_age_days,
        "compression_events": state.compression_events,
        "risk": state.risk,
        "status": (
            "healthy" if state.risk < 0.3
            else "warning" if state.risk < 0.6
            else "critical"
        ),
    }
