"""
psyche/persistence.py - Atomic State Persistence Manager

Handles safe saving and loading of psychological state snapshots:
- Atomic writes (temp file → rename) to prevent corruption
- Structural validation on load
- Safe fallback to defaults on corruption
- Separation from emotion/decision logic

Design principles:
- IO operations are isolated from state logic
- All-or-nothing persistence (no partial saves)
- No automatic value correction
- Single restore at startup only
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .snapshot import (
    Snapshot,
    create_default_snapshot,
    validate_snapshot,
    SNAPSHOT_VERSION,
)

logger = logging.getLogger(__name__)

# Default persistence location
DEFAULT_SNAPSHOT_DIR = Path(__file__).parent.parent / "data"
DEFAULT_SNAPSHOT_FILE = "psyche_snapshot.json"


class PersistenceManager:
    """
    Manages atomic persistence of psychological state snapshots.

    Usage:
        # At startup (once only)
        mgr = PersistenceManager()
        snapshot = mgr.load_or_create("user_1")

        # After each turn / periodically
        mgr.save(snapshot)

    The manager ensures:
    - Atomic writes that won't corrupt on crash
    - Structural validation on load
    - Safe defaults when data is corrupted
    - Complete isolation from state logic
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        filename: str = DEFAULT_SNAPSHOT_FILE,
    ):
        """
        Initialize persistence manager.

        Args:
            directory: Directory for snapshot files. Defaults to data/.
            filename: Snapshot filename. Defaults to psyche_snapshot.json.
        """
        self.directory = Path(directory) if directory else DEFAULT_SNAPSHOT_DIR
        self.filename = filename
        self._filepath = self.directory / self.filename

        # Ensure directory exists
        self.directory.mkdir(parents=True, exist_ok=True)

        # Track if restore has been performed (should only happen once)
        self._restored = False

    @property
    def filepath(self) -> Path:
        """Full path to the snapshot file."""
        return self._filepath

    def exists(self) -> bool:
        """Check if a snapshot file exists."""
        return self._filepath.exists()

    # ── Save Operations ────────────────────────────────────────

    def save(self, snapshot: Snapshot) -> bool:
        """
        Atomically save a snapshot to disk.

        Uses temp file + rename pattern to ensure atomic writes:
        1. Write to temporary file in same directory
        2. Flush and sync to ensure data is on disk
        3. Atomic rename to target filename

        This prevents corruption if the process crashes mid-write.

        Args:
            snapshot: The snapshot to save.

        Returns:
            True if save succeeded, False otherwise.
        """
        # Update timestamp
        snapshot = snapshot.update_timestamp()

        # Serialize
        try:
            data = snapshot.to_dict()
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to serialize snapshot: %s", e)
            return False

        # Atomic write: temp file → rename
        try:
            # Create temp file in same directory (required for atomic rename)
            fd, temp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix="snapshot_",
                dir=self.directory,
            )

            try:
                # Write data
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json_str)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (on POSIX, this is atomic; on Windows, mostly atomic)
                # On Windows, we need to handle the case where target exists
                if os.name == "nt":  # Windows
                    # Remove target if exists, then rename
                    if self._filepath.exists():
                        self._filepath.unlink()
                    os.rename(temp_path, self._filepath)
                else:
                    # POSIX: rename is atomic even if target exists
                    os.rename(temp_path, self._filepath)

                logger.debug("Snapshot saved: %s", self._filepath)
                return True

            except Exception as e:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise e

        except Exception as e:
            logger.error("Failed to save snapshot: %s", e)
            return False

    def save_if_changed(
        self,
        snapshot: Snapshot,
        last_saved: Optional[Snapshot] = None,
    ) -> bool:
        """
        Save snapshot only if it differs from the last saved version.

        Compares serialized forms to detect changes.

        Args:
            snapshot: Current snapshot to potentially save.
            last_saved: Previously saved snapshot for comparison.

        Returns:
            True if saved (or no change needed), False on error.
        """
        if last_saved is None:
            return self.save(snapshot)

        # Compare serialized forms
        try:
            current_data = snapshot.to_dict()
            last_data = last_saved.to_dict()

            # Remove timestamps for comparison
            current_data.pop("updated_at", None)
            last_data.pop("updated_at", None)

            if current_data == last_data:
                logger.debug("Snapshot unchanged, skipping save")
                return True

            return self.save(snapshot)

        except Exception as e:
            logger.error("Error comparing snapshots: %s", e)
            return self.save(snapshot)

    # ── Load Operations ────────────────────────────────────────

    def load(self) -> Optional[Snapshot]:
        """
        Load snapshot from disk with validation.

        Returns:
            Loaded snapshot if valid, None if file doesn't exist or is invalid.
        """
        if not self._filepath.exists():
            logger.info("No snapshot file found: %s", self._filepath)
            return None

        try:
            # Read file
            content = self._filepath.read_text(encoding="utf-8")
            data = json.loads(content)

            # Reconstruct snapshot
            snapshot = Snapshot.from_dict(data)
            if snapshot is None:
                logger.warning("Failed to reconstruct snapshot from data")
                return None

            # Validate structure
            is_valid, issues = validate_snapshot(snapshot)
            if not is_valid:
                logger.warning("Snapshot validation failed: %s", issues)
                return None

            logger.info(
                "Snapshot loaded: user=%s, version=%d, updated=%s",
                snapshot.user_id,
                snapshot.version,
                snapshot.updated_at,
            )
            return snapshot

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in snapshot file: %s", e)
            return None
        except Exception as e:
            logger.error("Failed to load snapshot: %s", e)
            return None

    def load_or_create(self, user_id: str = "default") -> Snapshot:
        """
        Load existing snapshot or create a new default.

        This is the primary method for startup initialization.
        Should only be called once per session.

        Args:
            user_id: User identifier for new snapshots.

        Returns:
            Loaded snapshot if valid, otherwise a fresh default snapshot.
        """
        if self._restored:
            logger.warning("load_or_create called multiple times - returning fresh default")

        self._restored = True

        # Try to load existing
        snapshot = self.load()
        if snapshot is not None:
            logger.info("Restored snapshot from disk (continuity preserved)")
            return snapshot

        # Create new default
        logger.info("Creating fresh default snapshot")
        snapshot = create_default_snapshot(user_id)

        # Save the new default
        self.save(snapshot)

        return snapshot

    # ── Maintenance Operations ─────────────────────────────────

    def backup(self, suffix: str = "") -> bool:
        """
        Create a backup of the current snapshot file.

        Args:
            suffix: Optional suffix for backup filename.

        Returns:
            True if backup succeeded or no file to backup.
        """
        if not self._filepath.exists():
            return True

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self._filepath.stem}_backup_{timestamp}{suffix}{self._filepath.suffix}"
            backup_path = self.directory / backup_name

            # Copy file content
            content = self._filepath.read_text(encoding="utf-8")
            backup_path.write_text(content, encoding="utf-8")

            logger.info("Backup created: %s", backup_path)
            return True

        except Exception as e:
            logger.error("Failed to create backup: %s", e)
            return False

    def delete(self) -> bool:
        """
        Delete the snapshot file (for testing or reset).

        Returns:
            True if deleted or didn't exist.
        """
        try:
            if self._filepath.exists():
                self._filepath.unlink()
                logger.info("Snapshot deleted: %s", self._filepath)
            return True
        except Exception as e:
            logger.error("Failed to delete snapshot: %s", e)
            return False


# ── Convenience Functions ──────────────────────────────────────

_default_manager: Optional[PersistenceManager] = None


def get_default_manager() -> PersistenceManager:
    """Get or create the default persistence manager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PersistenceManager()
    return _default_manager


def save_snapshot(snapshot: Snapshot) -> bool:
    """Save snapshot using the default manager."""
    return get_default_manager().save(snapshot)


def load_snapshot() -> Optional[Snapshot]:
    """Load snapshot using the default manager."""
    return get_default_manager().load()


def restore_or_create(user_id: str = "default") -> Snapshot:
    """Load or create snapshot using the default manager."""
    return get_default_manager().load_or_create(user_id)


# ── Integration Helper ─────────────────────────────────────────

def create_persistence_hooks(
    manager: Optional[PersistenceManager] = None,
) -> dict[str, Callable]:
    """
    Create callback hooks for integrating persistence into the main loop.

    Returns dict with:
    - on_startup: Call once at startup to restore state
    - on_turn_end: Call after each turn to save state
    - on_shutdown: Call before shutdown for final save

    Usage:
        hooks = create_persistence_hooks()
        snapshot = hooks["on_startup"]("user_1")
        # ... process turn, update snapshot ...
        hooks["on_turn_end"](snapshot)
        # ... on exit ...
        hooks["on_shutdown"](snapshot)
    """
    mgr = manager or get_default_manager()

    def on_startup(user_id: str = "default") -> Snapshot:
        """Restore or create snapshot at startup."""
        return mgr.load_or_create(user_id)

    def on_turn_end(snapshot: Snapshot) -> bool:
        """Save snapshot after turn completion."""
        return mgr.save(snapshot)

    def on_shutdown(snapshot: Snapshot) -> bool:
        """Final save before shutdown."""
        return mgr.save(snapshot)

    return {
        "on_startup": on_startup,
        "on_turn_end": on_turn_end,
        "on_shutdown": on_shutdown,
    }
