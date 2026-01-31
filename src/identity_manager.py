"""
src/identity_manager.py - Identity pillar persistence.

Manages ``data/identity.json``.  Supports proposing and applying
identity changes with a two-step confirmation flow for core traits.

Usage::

    mgr = IdentityManager()
    result = mgr.propose_identity_change({"trait": "romantic", "new_value": "cold"})
    if not result["requires_confirmation"]:
        mgr.apply_change(result["change"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
IDENTITY_FILE = DATA_DIR / "identity.json"


class IdentityManager:
    """File-backed identity manager."""

    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or IDENTITY_FILE
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.filepath.exists():
            return {
                "core_traits": [],
                "trait_confidence": {},
                "pending_changes": [],
                "risk": 0.3,
            }
        try:
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"core_traits": [], "trait_confidence": {}, "pending_changes": [], "risk": 0.3}

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

    def propose_identity_change(self, change: dict) -> dict:
        """Evaluate a proposed identity change.

        Returns ``requires_confirmation=True`` when the change targets
        a core trait (magnitude > threshold).
        """
        trait = change.get("trait", "")
        conflicts = trait in self._data.get("core_traits", [])

        if conflicts:
            self._data.setdefault("pending_changes", []).append(change)
            self._save()

        return {
            "change": change,
            "requires_confirmation": conflicts,
            "reason": (
                f"'{trait}' is a core trait — change needs confirmation"
                if conflicts
                else "no conflict with core traits"
            ),
        }

    def apply_change(self, change: dict):
        """Apply a confirmed identity change."""
        trait = change.get("trait", "")
        if trait and trait not in self._data.get("core_traits", []):
            self._data["core_traits"].append(trait)
        self._data["trait_confidence"][trait] = change.get("confidence", 0.5)
        pending = self._data.get("pending_changes", [])
        self._data["pending_changes"] = [p for p in pending if p != change]
        self._save()

    def calc_risk(self) -> float:
        """Compute identity risk (0.0 – 1.0)."""
        pending = self._data.get("pending_changes", [])
        confidence = self._data.get("trait_confidence", {})

        pending_risk = min(len(pending) * 0.15, 0.5)

        if confidence:
            avg_conf = sum(confidence.values()) / len(confidence)
            conf_risk = max(0.0, 0.5 - avg_conf)
        else:
            conf_risk = 0.3

        return min(pending_risk + conf_risk, 1.0)

    def get_risk(self) -> float:
        return self._data.get("risk", self.calc_risk())
