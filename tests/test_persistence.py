"""
tests/test_persistence.py - Tests for state persistence system

Verifies:
1. Snapshot structure and serialization
2. Atomic save operations
3. Load and validation
4. Corruption handling and safe defaults
5. Continuity across save/restore cycles
"""

import json
import time
from pathlib import Path

import pytest

from psyche.state import PsycheState, EmotionVector, Mood
from psyche.short_term_memory import ShortTermMemory
from psyche.short_term_loop import LoopState, LoopConfig
from psyche.responsibility import ResponsibilityState
from psyche.snapshot import (
    Snapshot,
    create_default_snapshot,
    validate_snapshot,
    SNAPSHOT_VERSION,
)
from psyche.persistence import (
    PersistenceManager,
    create_persistence_hooks,
)


class TestSnapshot:
    """Tests for the Snapshot structure."""

    def test_create_default_snapshot(self):
        """Default snapshot has all required components."""
        snapshot = create_default_snapshot("test_user")

        assert snapshot.user_id == "test_user"
        assert snapshot.version == SNAPSHOT_VERSION
        assert snapshot.psyche is not None
        assert snapshot.loop is not None
        assert snapshot.responsibility is not None

    def test_snapshot_to_dict(self):
        """Snapshot serializes to dict correctly."""
        snapshot = create_default_snapshot()
        data = snapshot.to_dict()

        assert "version" in data
        assert "psyche" in data
        assert "loop" in data
        assert "responsibility" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_snapshot_from_dict_roundtrip(self):
        """Snapshot survives serialization roundtrip."""
        original = create_default_snapshot("roundtrip_user")

        # Modify some values
        original.psyche = PsycheState(
            emotions=EmotionVector(joy=0.7, sorrow=0.2),
            mood=Mood(valence=0.5, arousal=0.6),
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = Snapshot.from_dict(data)

        assert restored is not None
        assert restored.user_id == "roundtrip_user"
        assert restored.psyche.emotions.joy == 0.7
        assert restored.psyche.emotions.sorrow == 0.2
        assert restored.psyche.mood.valence == 0.5

    def test_snapshot_from_dict_invalid_version(self):
        """Future version returns None."""
        data = create_default_snapshot().to_dict()
        data["version"] = SNAPSHOT_VERSION + 100

        result = Snapshot.from_dict(data)
        assert result is None

    def test_snapshot_from_dict_missing_fields(self):
        """Missing required fields return None."""
        # Missing psyche
        data = {"version": 1, "loop": {}, "responsibility": {}}
        assert Snapshot.from_dict(data) is None

        # Missing version
        data = {"psyche": {}, "loop": {}, "responsibility": {}}
        assert Snapshot.from_dict(data) is None

    def test_validate_snapshot_valid(self):
        """Valid snapshot passes validation."""
        snapshot = create_default_snapshot()
        is_valid, issues = validate_snapshot(snapshot)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_snapshot_detects_issues(self):
        """Validation detects structural issues."""
        snapshot = create_default_snapshot()

        # Test with future version (which validation should catch)
        snapshot.version = SNAPSHOT_VERSION + 100

        is_valid, issues = validate_snapshot(snapshot)
        assert is_valid is False
        assert any("version" in issue.lower() for issue in issues)


class TestPersistenceManager:
    """Tests for the PersistenceManager."""

    def test_save_and_load(self, tmp_path: Path):
        """Basic save and load cycle works."""
        mgr = PersistenceManager(directory=tmp_path, filename="test.json")

        snapshot = create_default_snapshot("save_test")
        snapshot.psyche = PsycheState(
            emotions=EmotionVector(joy=0.8, fear=0.1)
        )

        # Save
        assert mgr.save(snapshot) is True
        assert mgr.exists() is True

        # Load
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.user_id == "save_test"
        assert loaded.psyche.emotions.joy == 0.8

    def test_load_nonexistent(self, tmp_path: Path):
        """Loading nonexistent file returns None."""
        mgr = PersistenceManager(directory=tmp_path, filename="nonexistent.json")

        assert mgr.exists() is False
        assert mgr.load() is None

    def test_load_or_create_new(self, tmp_path: Path):
        """load_or_create creates new snapshot when none exists."""
        mgr = PersistenceManager(directory=tmp_path, filename="new.json")

        snapshot = mgr.load_or_create("new_user")

        assert snapshot is not None
        assert snapshot.user_id == "new_user"
        assert mgr.exists() is True  # Should have saved the new snapshot

    def test_load_or_create_existing(self, tmp_path: Path):
        """load_or_create restores existing snapshot."""
        mgr = PersistenceManager(directory=tmp_path, filename="existing.json")

        # Create and save initial
        initial = create_default_snapshot("existing_user")
        initial.psyche = PsycheState(emotions=EmotionVector(love=0.9))
        mgr.save(initial)

        # Create new manager and restore
        mgr2 = PersistenceManager(directory=tmp_path, filename="existing.json")
        restored = mgr2.load_or_create("different_user")

        # Should restore existing, not create new
        assert restored.user_id == "existing_user"
        assert restored.psyche.emotions.love == 0.9

    def test_atomic_save_creates_file(self, tmp_path: Path):
        """Atomic save creates the target file."""
        mgr = PersistenceManager(directory=tmp_path, filename="atomic.json")
        snapshot = create_default_snapshot()

        mgr.save(snapshot)

        # File should exist and be valid JSON
        content = (tmp_path / "atomic.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["version"] == SNAPSHOT_VERSION

    def test_load_corrupted_json(self, tmp_path: Path):
        """Corrupted JSON returns None."""
        filepath = tmp_path / "corrupted.json"
        filepath.write_text("{ invalid json }", encoding="utf-8")

        mgr = PersistenceManager(directory=tmp_path, filename="corrupted.json")
        assert mgr.load() is None

    def test_load_invalid_structure(self, tmp_path: Path):
        """Invalid structure returns None."""
        filepath = tmp_path / "invalid.json"
        filepath.write_text('{"foo": "bar"}', encoding="utf-8")

        mgr = PersistenceManager(directory=tmp_path, filename="invalid.json")
        assert mgr.load() is None

    def test_backup(self, tmp_path: Path):
        """Backup creates a copy of the snapshot file."""
        mgr = PersistenceManager(directory=tmp_path, filename="backup_test.json")

        snapshot = create_default_snapshot()
        mgr.save(snapshot)

        assert mgr.backup() is True

        # Should have at least 2 files now
        files = list(tmp_path.glob("*.json"))
        assert len(files) >= 2

    def test_delete(self, tmp_path: Path):
        """Delete removes the snapshot file."""
        mgr = PersistenceManager(directory=tmp_path, filename="delete_test.json")

        snapshot = create_default_snapshot()
        mgr.save(snapshot)
        assert mgr.exists() is True

        mgr.delete()
        assert mgr.exists() is False


class TestContinuity:
    """Tests verifying psychological continuity across restarts."""

    def test_emotion_continuity(self, tmp_path: Path):
        """Emotions persist across save/restore cycle."""
        mgr = PersistenceManager(directory=tmp_path)

        # Session 1: Build up emotions
        snapshot = create_default_snapshot()
        snapshot.psyche = PsycheState(
            emotions=EmotionVector(joy=0.8, sorrow=0.3, fear=0.1),
            mood=Mood(valence=0.6, arousal=0.7),
        )
        mgr.save(snapshot)

        # Session 2: Restore
        mgr2 = PersistenceManager(directory=tmp_path)
        restored = mgr2.load_or_create()

        assert restored.psyche.emotions.joy == 0.8
        assert restored.psyche.emotions.sorrow == 0.3
        assert restored.psyche.mood.valence == 0.6

    def test_short_term_memory_continuity(self, tmp_path: Path):
        """Short-term memory persists across save/restore cycle."""
        mgr = PersistenceManager(directory=tmp_path)

        # Session 1: Add stimuli to short-term memory
        snapshot = create_default_snapshot()
        snapshot.loop.memory = snapshot.loop.memory.add_stimulus(
            source_text="Hello!",
            topics=["greeting"],
            emotion_label="happy",
            intent="greeting",
            raw_intensity=0.5,
            valence=0.5,
        )
        snapshot.loop.memory = snapshot.loop.memory.add_stimulus(
            source_text="Great game!",
            topics=["game"],
            emotion_label="happy",
            intent="praise",
            raw_intensity=0.8,
            valence=0.8,
        )
        mgr.save(snapshot)

        # Session 2: Restore
        mgr2 = PersistenceManager(directory=tmp_path)
        restored = mgr2.load_or_create()

        assert len(restored.loop.memory.entries) == 2
        assert restored.loop.memory.entries[0].source_text == "Hello!"
        assert restored.loop.memory.entries[1].valence == 0.8

    def test_responsibility_continuity(self, tmp_path: Path):
        """Responsibility burden persists across save/restore cycle."""
        mgr = PersistenceManager(directory=tmp_path)

        # Session 1: Accumulate responsibility
        snapshot = create_default_snapshot()
        snapshot.responsibility = ResponsibilityState(
            total_weight=0.5,
            accumulated_harm=0.3,
            accumulated_confidence=0.2,
        )
        mgr.save(snapshot)

        # Session 2: Restore
        mgr2 = PersistenceManager(directory=tmp_path)
        restored = mgr2.load_or_create()

        assert restored.responsibility.total_weight == 0.5
        assert restored.responsibility.accumulated_harm == 0.3


class TestPersistenceHooks:
    """Tests for the integration hooks."""

    def test_hooks_lifecycle(self, tmp_path: Path):
        """Hooks provide correct lifecycle callbacks."""
        mgr = PersistenceManager(directory=tmp_path)
        hooks = create_persistence_hooks(mgr)

        # Startup
        snapshot = hooks["on_startup"]("hook_user")
        assert snapshot is not None
        assert snapshot.user_id == "hook_user"

        # Modify state
        snapshot.psyche = PsycheState(emotions=EmotionVector(fun=0.9))

        # Turn end
        assert hooks["on_turn_end"](snapshot) is True

        # Shutdown
        assert hooks["on_shutdown"](snapshot) is True

        # Verify persisted
        loaded = mgr.load()
        assert loaded.psyche.emotions.fun == 0.9


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_short_term_memory(self, tmp_path: Path):
        """Empty short-term memory saves and restores correctly."""
        mgr = PersistenceManager(directory=tmp_path)

        snapshot = create_default_snapshot()
        assert len(snapshot.loop.memory.entries) == 0

        mgr.save(snapshot)
        restored = mgr.load()

        assert len(restored.loop.memory.entries) == 0

    def test_timestamp_present_after_save(self, tmp_path: Path):
        """updated_at timestamp is present after save."""
        mgr = PersistenceManager(directory=tmp_path)

        snapshot = create_default_snapshot()
        mgr.save(snapshot)

        loaded = mgr.load()
        # Timestamp should be a valid ISO format string
        assert loaded.updated_at is not None
        assert "T" in loaded.updated_at  # ISO format contains T separator

    def test_version_preserved(self, tmp_path: Path):
        """Version number is preserved through save/load."""
        mgr = PersistenceManager(directory=tmp_path)

        snapshot = create_default_snapshot()
        assert snapshot.version == SNAPSHOT_VERSION

        mgr.save(snapshot)
        loaded = mgr.load()

        assert loaded.version == SNAPSHOT_VERSION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
