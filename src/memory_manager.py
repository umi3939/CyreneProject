"""
src/memory_manager.py - Long-term memory persistence with scoring.

Atomic read/write to ``data/example_memories.json``.

Scoring formula for recall::

    S = w1*(importance/5) + w2*recency_factor + w3*keyword_match + w4*fuzzy_sim

Usage::

    mgr = MemoryManager()
    results = mgr.recall("ゲームの話", top_k=3)
    mgr.maybe_save("新しいイベント", "応答", state_dict)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MEMORIES_FILE = DATA_DIR / "example_memories.json"

# Scoring weights
W_IMPORTANCE = 0.4
W_RECENCY = 0.3
W_KEYWORD = 0.2
W_FUZZY = 0.1


class MemoryManager:
    """File-backed memory store with recall scoring and lifecycle management."""

    def __init__(
        self,
        filepath: Path | None = None,
        llm_call: Optional[Callable[..., Awaitable[str]]] = None,
    ):
        self.filepath = filepath or MEMORIES_FILE
        self._llm_call = llm_call
        self._memories: list[dict] = self._load()
        logger.info("MemoryManager loaded %d memories from %s", len(self._memories), self.filepath)

    # ── persistence ────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not self.filepath.exists():
            return []
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load memories: %s", e)
            return []

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.filepath.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Atomic rename (best-effort on Windows)
        shutil.move(str(tmp), str(self.filepath))

    # ── public API ─────────────────────────────────────────────

    @property
    def memories(self) -> list[dict]:
        """Return a shallow copy of all memories."""
        return list(self._memories)

    @property
    def count(self) -> int:
        return len(self._memories)

    async def recall(self, user_input: str, top_k: int = 3) -> list[dict]:
        """Retrieve most relevant memories using weighted scoring.

        S = w1*(importance/5) + w2*recency + w3*keyword_match + w4*fuzzy
        """
        if not self._memories or not user_input:
            return []

        now = datetime.now()
        scored: list[tuple[float, dict]] = []

        for mem in self._memories:
            imp = mem.get("importance", 1) / 5.0

            try:
                age_days = (now - datetime.fromisoformat(mem["date"])).days
                recency = max(0.0, 1.0 - age_days / 365.0)
            except (ValueError, KeyError):
                recency = 0.0

            keywords = mem.get("keywords", [])
            kw_hits = sum(1 for k in keywords if k.lower() in user_input.lower())
            keyword_score = kw_hits / max(len(keywords), 1)

            summary = mem.get("summary", "")
            common = sum(1 for c in set(user_input) if c in summary)
            fuzzy = common / max(len(set(user_input)), 1)

            score = (
                W_IMPORTANCE * imp
                + W_RECENCY * recency
                + W_KEYWORD * keyword_score
                + W_FUZZY * fuzzy
            )
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:top_k]]

        # Mark as recalled
        now_str = now.isoformat(timespec="seconds")
        for m in results:
            m["last_recalled"] = now_str
        if results:
            self._save()
        return results

    def maybe_save(
        self,
        event: str,
        candidate_response: str,
        state: dict[str, Any],
        *,
        explicit_request: bool = False,
        importance: int = 3,
        involves_attachment: bool = False,
    ) -> bool:
        """Decide whether to persist an event as a new memory.

        Saves when ``explicit_request``, ``importance >= 3``, or
        ``involves_attachment`` is true.
        """
        should_save = explicit_request or importance >= 3 or involves_attachment
        if not should_save:
            return False

        entry: dict[str, Any] = {
            "id": max((m.get("id", 0) for m in self._memories), default=0) + 1,
            "summary": event[:300],
            "keywords": _extract_keywords(event),
            "importance": min(5, max(1, importance)),
            "date": datetime.now().isoformat(timespec="seconds"),
            "protected": importance >= 5,
            "last_recalled": None,
        }
        self._memories.append(entry)
        self._save()
        logger.info("Memory saved (id=%d): %s", entry["id"], entry["summary"][:60])
        return True

    async def compress_and_cleanup(self, max_age_days: int = 90) -> int:
        """Compress old, low-importance memories.

        Protected memories are never removed.
        """
        now = datetime.now()
        compressed = 0
        kept: list[dict] = []

        for mem in self._memories:
            if mem.get("protected"):
                kept.append(mem)
                continue
            try:
                age = (now - datetime.fromisoformat(mem["date"])).days
            except (ValueError, KeyError):
                age = 999

            if age > max_age_days and mem.get("importance", 1) < 3:
                if self._llm_call:
                    try:
                        summary = await self._llm_call(
                            f"以下を一行で要約して:\n{mem.get('summary', '')}",
                            {},
                        )
                        mem["summary"] = summary[:200]
                        mem["importance"] = max(1, mem.get("importance", 1) - 1)
                        kept.append(mem)
                    except Exception:
                        pass
                compressed += 1
            else:
                kept.append(mem)

        self._memories = kept
        self._save()
        return compressed

    def backup(self, backup_path: Path | None = None) -> Path:
        """Create a backup copy of the memories file."""
        if backup_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.filepath.with_suffix(f".backup_{ts}.json")
        if self.filepath.exists():
            shutil.copy2(str(self.filepath), str(backup_path))
        return backup_path


def _extract_keywords(text: str) -> list[str]:
    """Simple keyword extraction (non-trivial tokens)."""
    import re
    words = re.findall(r"[\w\u3040-\u9fff]+", text)
    return list(set(w for w in words if len(w) > 1))[:10]
