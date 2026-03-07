"""
src/state_manager.py - Per-user state persistence and fear index.

Stores and retrieves per-user ``PsycheState`` dicts from
``data/state.json``.  Computes the composite ``fear_index`` from the
four pillar risks.

Usage::

    mgr = StateManager()
    state = mgr.get_state("user_1")
    mgr.update_state_on_event("user_1", event_dict, feedback_dict)
    fear = StateManager.calc_fear_index(0.1, 0.3, 0.5, 0.2)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.emotion_model import event_to_emotion, apply_decay

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "state.json"

_DEFAULT_STATE: dict[str, Any] = {
    "emotions": {"joy": 0.0, "sad": 0.0, "fear": 0.0, "anger": 0.0, "calm": 0.5},
    "drives": {"social": 0.5, "curiosity": 0.5},
    "mood": 0.0,
    "last_updated": "2025-01-01T00:00:00",
    "loss_aversion": 0.3,
    "fear_index": 0.0,
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class StateManager:
    """File-backed per-user psychological state manager."""

    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or STATE_FILE
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.filepath.exists():
            return {}
        try:
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.filepath.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.filepath)

    # ── public API ─────────────────────────────────────────────

    def get_state(self, user_id: str) -> dict[str, Any]:
        """Return a copy of the state for *user_id*, creating a default if needed."""
        if user_id not in self._data:
            self._data[user_id] = _make_default_state()
            self._save()
        return dict(self._data[user_id])

    def set_state(self, user_id: str, state: dict[str, Any]):
        """Overwrite the state for *user_id*."""
        self._data[user_id] = state
        self._save()

    def update_state_on_event(
        self,
        user_id: str,
        event: dict[str, Any],
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update the user's state based on an incoming event.

        1. Compute emotion delta from the event.
        2. Apply time-based decay.
        3. Update drives (social satisfied by interaction, curiosity by info).
        4. Recalculate mood as average valence.
        5. Persist.

        Returns the updated state dict.
        """
        state = self.get_state(user_id)

        # Time delta (capped at 1 hour to prevent extreme decay)
        try:
            last = datetime.fromisoformat(state["last_updated"])
            delta_seconds = min(3600.0, max(0.0, (datetime.now() - last).total_seconds()))
        except (ValueError, KeyError):
            delta_seconds = 1.0

        # 1. Decay existing emotions over elapsed time
        decayed_base = apply_decay(state.get("emotions", {}), delta_seconds)
        # 2. Apply new event stimulus on top of decayed base
        new_emotions = event_to_emotion(event, base=decayed_base)

        # Drive update
        drives = dict(state.get("drives", {"social": 0.5, "curiosity": 0.5}))
        drives["social"] = _clamp(drives["social"] - 0.05)       # talking satisfies
        drives["social"] = _clamp(drives["social"] + 0.01 * min(delta_seconds, 60.0))  # lonely over time
        if event.get("intent") in ("question", "sharing"):
            drives["curiosity"] = _clamp(drives["curiosity"] - 0.05)
        drives["curiosity"] = _clamp(drives["curiosity"] + 0.005 * min(delta_seconds, 60.0))

        # Mood = net valence of emotions
        pos = new_emotions.get("joy", 0) + new_emotions.get("calm", 0)
        neg = new_emotions.get("sad", 0) + new_emotions.get("anger", 0) + new_emotions.get("fear", 0)
        new_mood = _clamp((pos - neg) / 2.0, -1.0, 1.0)

        state.update({
            "emotions": new_emotions,
            "drives": drives,
            "mood": round(new_mood, 4),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        })

        # Apply feedback-based fear if provided
        if feedback and "fear_index" in feedback:
            state["fear_index"] = feedback["fear_index"]

        self._data[user_id] = state
        self._save()
        return state

    # ── fear index ─────────────────────────────────────────────

    @staticmethod
    def calc_fear_index(
        identity_risk: float = 0.0,
        attachment_risk: float = 0.0,
        continuity_risk: float = 0.0,
        projection_risk: float = 0.0,
    ) -> float:
        """Weighted composite fear index (0.0 – 1.0).

        Weights: identity 0.3, attachment 0.3, continuity 0.2, projection 0.2.
        """
        return _clamp(
            identity_risk * 0.3
            + attachment_risk * 0.3
            + continuity_risk * 0.2
            + projection_risk * 0.2
        )


def _make_default_state() -> dict[str, Any]:
    import copy
    s = copy.deepcopy(_DEFAULT_STATE)
    s["last_updated"] = datetime.now().isoformat(timespec="seconds")
    return s
