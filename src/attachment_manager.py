"""
src/attachment_manager.py - Attachment bond persistence.

Manages ``data/example_attachments.json``.
Bond update rules, daily decay, and partner ranking.

Usage::

    mgr = AttachmentManager()
    mgr.update_bond("default_user", "user_A", importance=4, positive=True)
    top = mgr.get_top_partners("default_user", n=3)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
ATTACHMENTS_FILE = DATA_DIR / "example_attachments.json"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class AttachmentManager:
    """File-backed attachment bond manager."""

    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or ATTACHMENTS_FILE
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.filepath.exists():
            return {}
        try:
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_user(self, user_id: str):
        if user_id not in self._data:
            self._data[user_id] = {
                "bonds": {},
                "last_interaction": {},
                "risk": 0.7,
            }

    # ── bond updates ───────────────────────────────────────────

    def update_bond(
        self,
        user_id: str,
        partner_id: str,
        importance: int = 3,
        positive: bool = True,
    ):
        """Strengthen or weaken a bond based on an event.

        bond += importance * 0.02  (positive)
        bond -= importance * 0.03  (negative)
        """
        self._ensure_user(user_id)
        bonds = self._data[user_id]["bonds"]
        current = bonds.get(partner_id, 0.0)

        if positive:
            bonds[partner_id] = _clamp01(current + importance * 0.02)
        else:
            bonds[partner_id] = _clamp01(current - importance * 0.03)

        self._data[user_id]["last_interaction"][partner_id] = (
            datetime.now().isoformat(timespec="seconds")
        )
        self._data[user_id]["risk"] = self._calc_risk(bonds)
        self._save()

    def apply_daily_decay(self, user_id: str, days: float = 1.0):
        """Decay all bonds for a user by time passage (0.98^days)."""
        self._ensure_user(user_id)
        bonds = self._data[user_id]["bonds"]
        factor = 0.98 ** days
        for pid in bonds:
            bonds[pid] = _clamp01(bonds[pid] * factor)
        self._data[user_id]["risk"] = self._calc_risk(bonds)
        self._save()

    def get_top_partners(self, user_id: str, n: int = 3) -> list[tuple[str, float]]:
        """Return top-n partners by bond strength."""
        self._ensure_user(user_id)
        bonds = self._data[user_id].get("bonds", {})
        return sorted(bonds.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_risk(self, user_id: str) -> float:
        """Return current attachment risk for a user."""
        self._ensure_user(user_id)
        return self._data[user_id].get("risk", 0.7)

    def get_state(self, user_id: str) -> dict:
        """Return full attachment state for a user."""
        self._ensure_user(user_id)
        return dict(self._data[user_id])

    @staticmethod
    def _calc_risk(bonds: dict[str, float]) -> float:
        if not bonds:
            return 0.7
        max_bond = max(bonds.values())
        if max_bond < 0.3:
            return 0.6
        if max_bond < 0.5:
            return 0.3
        return max(0.0, 0.2 - max_bond * 0.1)
