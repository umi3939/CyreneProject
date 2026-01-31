"""
src/projection_manager.py - Projection (goals/purpose) persistence.

CRUD for ``data/projections.json``.

Usage::

    mgr = ProjectionManager()
    mgr.add_goal("新しい歌を覚える")
    mgr.simulate_progress_change("entertain", 0.1)
    mgr.reset("entertain")  # for testing
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PROJECTIONS_FILE = DATA_DIR / "projections.json"


class ProjectionManager:
    """File-backed goal/projection manager."""

    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or PROJECTIONS_FILE
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.filepath.exists():
            return {"goals": [], "risk": 0.7}
        try:
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"goals": [], "risk": 0.7}

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._data["risk"] = self.calc_risk()
        self.filepath.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def state(self) -> dict:
        return dict(self._data)

    @property
    def goals(self) -> list[dict]:
        return list(self._data.get("goals", []))

    def add_goal(self, description: str) -> dict:
        """Add a new active goal."""
        goal = {
            "id": uuid.uuid4().hex[:8],
            "description": description,
            "progress": 0.0,
            "status": "active",
        }
        self._data.setdefault("goals", []).append(goal)
        self._save()
        return goal

    def simulate_progress_change(self, goal_id: str, delta: float):
        """Update progress on a goal by *delta* (clamped 0–1)."""
        for g in self._data.get("goals", []):
            if g.get("id") == goal_id:
                g["progress"] = max(0.0, min(1.0, g.get("progress", 0.0) + delta))
                if g["progress"] >= 1.0:
                    g["status"] = "completed"
                self._save()
                return
        logger.warning("Goal %s not found", goal_id)

    def remove_goal(self, goal_id: str):
        """Remove a goal by ID."""
        self._data["goals"] = [
            g for g in self._data.get("goals", []) if g.get("id") != goal_id
        ]
        self._save()

    def reset(self, goal_id: str):
        """Reset a goal's progress to 0 (for testing)."""
        for g in self._data.get("goals", []):
            if g.get("id") == goal_id:
                g["progress"] = 0.0
                g["status"] = "active"
                self._save()
                return

    def calc_risk(self) -> float:
        """Compute projection risk (0.0 – 1.0)."""
        goals = self._data.get("goals", [])
        if not goals:
            return 0.7
        active = [g for g in goals if g.get("status") == "active"]
        if not active:
            return 0.5
        progressing = [g for g in active if g.get("progress", 0.0) > 0.0]
        if not progressing:
            return 0.6
        return 0.1

    def get_risk(self) -> float:
        return self._data.get("risk", self.calc_risk())
