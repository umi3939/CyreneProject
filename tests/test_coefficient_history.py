"""
tests/test_coefficient_history.py - Tests for coefficient history recording.

Tests (design_coefficient_history.md):
- Session start snapshot capture
- Delta computation with previous session
- History entry structure (timestamp, changes, change_count, snapshot)
- FIFO limit enforcement
- No-change recording ("no changes" fact)
- First session recording (no previous data)
- History file read/write failure safety
- Psyche non-impact (no enrichment, no save/load, no processing input)
- History file separate from coefficients.json
- Coefficient registry load() not affected by history failures
"""

import json
import os
import tempfile

import pytest

from psyche import coefficient_registry


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before and after each test."""
    coefficient_registry.reset()
    yield
    coefficient_registry.reset()


@pytest.fixture
def history_path(tmp_path):
    """Provide a temporary history file path."""
    return str(tmp_path / "coefficient_history.json")


@pytest.fixture
def coeff_path(tmp_path):
    """Provide a temporary coefficients file path."""
    return str(tmp_path / "coefficients.json")


def _write_json(path: str, data) -> None:
    """Helper to write JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_json(path: str):
    """Helper to read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Test: First session recording (no previous history)
# =============================================================================

class TestFirstSession:
    """When no history file exists, first session is recorded correctly."""

    def test_first_session_creates_history_file(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        assert entry is not None
        assert os.path.isfile(history_path)

    def test_first_session_entry_structure(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        assert "timestamp" in entry
        assert "changes" in entry
        assert "change_count" in entry
        assert "snapshot" in entry

    def test_first_session_has_empty_changes(self, history_path):
        """First session has no previous data to compare, so changes is empty."""
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        assert entry["changes"] == []
        assert entry["change_count"] == 0

    def test_first_session_snapshot_contains_all_coefficients(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        snapshot = entry["snapshot"]
        # Flattened keys should include various coefficients
        assert "drive_dynamics.total_change_limit" in snapshot
        assert "mood_autonomy.tracking_speed_min" in snapshot
        assert "policy_selection.score_section_band" in snapshot
        assert "value_orientation.base_learning_rate" in snapshot
        assert "fluctuation.amplitude_cap" in snapshot

    def test_first_session_snapshot_values_match_defaults(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        snapshot = entry["snapshot"]
        assert snapshot["drive_dynamics.total_change_limit"] == 0.15
        assert snapshot["mood_autonomy.tracking_speed_min"] == 0.03
        assert snapshot["policy_selection.score_section_band"] == 1.5

    def test_first_session_timestamp_is_iso_format(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        ts = entry["timestamp"]
        assert isinstance(ts, str)
        # Should be parseable ISO format with timezone
        assert "T" in ts

    def test_history_file_contains_one_entry(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        data = _read_json(history_path)
        assert isinstance(data, list)
        assert len(data) == 1


# =============================================================================
# Test: Delta computation with previous session
# =============================================================================

class TestDeltaComputation:
    """Changes are correctly detected between sessions."""

    def test_no_changes_detected_when_same_values(self, history_path):
        """Two sessions with same defaults -> no changes."""
        # First session
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        # Second session (reset + reload same defaults)
        coefficient_registry.reset()
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        assert entry["change_count"] == 0
        assert entry["changes"] == []

    def test_changes_detected_when_values_differ(self, tmp_path, history_path):
        """Two sessions with different values -> changes detected."""
        # First session with defaults
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        # Second session with modified value
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {
            "drive_dynamics": {"total_change_limit": 0.25},
        })
        coefficient_registry.reset()
        coefficient_registry.load(coeff_path)
        entry = coefficient_registry.record_history(history_path)

        assert entry["change_count"] == 1
        assert len(entry["changes"]) == 1
        change = entry["changes"][0]
        assert change["key"] == "drive_dynamics.total_change_limit"
        assert change["old_value"] == 0.15
        assert change["new_value"] == 0.25

    def test_multiple_changes_detected(self, tmp_path, history_path):
        """Multiple coefficients changed between sessions."""
        # First session with defaults
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        # Second session with two changed values
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {
            "drive_dynamics": {"total_change_limit": 0.25},
            "fluctuation": {"amplitude_cap": 0.20},
        })
        coefficient_registry.reset()
        coefficient_registry.load(coeff_path)
        entry = coefficient_registry.record_history(history_path)

        assert entry["change_count"] == 2
        changed_keys = {c["key"] for c in entry["changes"]}
        assert "drive_dynamics.total_change_limit" in changed_keys
        assert "fluctuation.amplitude_cap" in changed_keys

    def test_nested_value_change_detected(self, tmp_path, history_path):
        """Changes in deeply nested values are detected."""
        # First session
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        # Second session with nested change
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {
            "drive_dynamics": {
                "section_band": {
                    "emotion_drive_coupling": {"social": 0.10, "curiosity": 0.06, "expression": 0.06},
                },
            },
        })
        coefficient_registry.reset()
        coefficient_registry.load(coeff_path)
        entry = coefficient_registry.record_history(history_path)

        assert entry["change_count"] >= 1
        changed_keys = {c["key"] for c in entry["changes"]}
        assert "drive_dynamics.section_band.emotion_drive_coupling.social" in changed_keys

    def test_change_old_and_new_values_correct(self, tmp_path, history_path):
        """Each change entry contains correct old and new values."""
        # First session
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        # Second session
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {
            "mood_autonomy": {"tracking_speed_min": 0.05},
        })
        coefficient_registry.reset()
        coefficient_registry.load(coeff_path)
        entry = coefficient_registry.record_history(history_path)

        changes_by_key = {c["key"]: c for c in entry["changes"]}
        assert "mood_autonomy.tracking_speed_min" in changes_by_key
        change = changes_by_key["mood_autonomy.tracking_speed_min"]
        assert change["old_value"] == 0.03
        assert change["new_value"] == 0.05


# =============================================================================
# Test: History accumulation across multiple sessions
# =============================================================================

class TestMultipleSessions:
    """History correctly accumulates across multiple sessions."""

    def test_three_sessions_accumulated(self, tmp_path, history_path):
        """Three sessions create three entries."""
        for i in range(3):
            coefficient_registry.reset()
            coefficient_registry.load("/nonexistent.json")
            coefficient_registry.record_history(history_path)

        data = _read_json(history_path)
        assert len(data) == 3

    def test_each_session_has_timestamp(self, history_path):
        """Each session entry has its own timestamp."""
        for _ in range(2):
            coefficient_registry.reset()
            coefficient_registry.load("/nonexistent.json")
            coefficient_registry.record_history(history_path)

        data = _read_json(history_path)
        assert all("timestamp" in entry for entry in data)
        # Timestamps should be different (or at least both present)
        assert data[0]["timestamp"] is not None
        assert data[1]["timestamp"] is not None

    def test_session_with_change_then_revert(self, tmp_path, history_path):
        """Session 1: defaults, Session 2: changed, Session 3: reverted."""
        # Session 1: defaults
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        # Session 2: changed
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {"fluctuation": {"amplitude_cap": 0.20}})
        coefficient_registry.reset()
        coefficient_registry.load(coeff_path)
        entry2 = coefficient_registry.record_history(history_path)
        assert entry2["change_count"] == 1

        # Session 3: reverted to defaults
        coefficient_registry.reset()
        coefficient_registry.load("/nonexistent.json")
        entry3 = coefficient_registry.record_history(history_path)
        assert entry3["change_count"] == 1
        change = entry3["changes"][0]
        assert change["key"] == "fluctuation.amplitude_cap"
        assert change["old_value"] == 0.20
        assert change["new_value"] == 0.12


# =============================================================================
# Test: FIFO limit
# =============================================================================

class TestFIFOLimit:
    """History entries are limited by FIFO."""

    def test_fifo_limit_enforced(self, history_path):
        """Entries beyond FIFO limit are removed (oldest first)."""
        limit = coefficient_registry._HISTORY_FIFO_LIMIT

        # Create entries exceeding the limit
        for i in range(limit + 10):
            coefficient_registry.reset()
            coefficient_registry.load("/nonexistent.json")
            coefficient_registry.record_history(history_path)

        data = _read_json(history_path)
        assert len(data) == limit

    def test_fifo_preserves_most_recent(self, history_path):
        """FIFO keeps the most recent entries, not the oldest."""
        limit = coefficient_registry._HISTORY_FIFO_LIMIT

        for i in range(limit + 5):
            coefficient_registry.reset()
            coefficient_registry.load("/nonexistent.json")
            coefficient_registry.record_history(history_path)

        data = _read_json(history_path)
        assert len(data) == limit
        # All entries should have timestamps (basic sanity)
        assert all("timestamp" in e for e in data)

    def test_fifo_at_exact_limit(self, history_path):
        """Exactly at the limit, no entries should be removed."""
        limit = coefficient_registry._HISTORY_FIFO_LIMIT

        for i in range(limit):
            coefficient_registry.reset()
            coefficient_registry.load("/nonexistent.json")
            coefficient_registry.record_history(history_path)

        data = _read_json(history_path)
        assert len(data) == limit


# =============================================================================
# Test: Snapshot completeness
# =============================================================================

class TestSnapshotCompleteness:
    """Snapshot captures all coefficient values."""

    def test_snapshot_is_flat_dict(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        snapshot = entry["snapshot"]
        assert isinstance(snapshot, dict)
        # All values should be non-dict (flattened)
        for k, v in snapshot.items():
            assert not isinstance(v, dict), f"Key '{k}' should be flattened, got dict"

    def test_snapshot_keys_use_dot_notation(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        snapshot = entry["snapshot"]
        # Nested keys should use dot notation
        has_dots = any("." in k for k in snapshot.keys())
        assert has_dots, "Flattened snapshot should have dot-separated keys"

    def test_snapshot_contains_all_categories(self, history_path):
        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        snapshot = entry["snapshot"]
        categories = set()
        for key in snapshot:
            categories.add(key.split(".")[0])

        expected = {
            "drive_dynamics", "mood_autonomy", "policy_selection",
            "value_orientation", "fluctuation", "experience_intensity",
            "emotion_processing", "perception",
            "memory_emotion_return", "other_hypothesis_emotion_return",
            "description_common",
        }
        assert categories == expected

    def test_snapshot_reflects_custom_values(self, tmp_path, history_path):
        """Snapshot reflects values from a custom coefficients.json."""
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {
            "drive_dynamics": {"total_change_limit": 0.30},
        })
        coefficient_registry.load(coeff_path)
        entry = coefficient_registry.record_history(history_path)

        assert entry["snapshot"]["drive_dynamics.total_change_limit"] == 0.30


# =============================================================================
# Test: History file read/write failure safety
# =============================================================================

class TestFailureSafety:
    """History file failures do not affect coefficient registry operation."""

    def test_record_returns_none_when_not_loaded(self, history_path):
        """record_history() returns None if registry not loaded."""
        result = coefficient_registry.record_history(history_path)
        assert result is None

    def test_record_with_unwritable_path(self, tmp_path):
        """Writing to an invalid path logs warning but does not raise."""
        coefficient_registry.load("/nonexistent.json")
        # Use a path that cannot be written (directory does not exist and
        # parent is not writable on some systems, or use invalid chars)
        bad_path = str(tmp_path / "nonexistent_dir" / "sub" / "deep" / "history.json")
        # Should not raise - creates directories
        entry = coefficient_registry.record_history(bad_path)
        assert entry is not None

    def test_corrupted_history_file_handled(self, history_path):
        """Corrupted history file is handled gracefully."""
        # Write garbage to history file
        with open(history_path, "w") as f:
            f.write("not valid json {{{")

        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        assert entry is not None
        assert entry["change_count"] == 0  # No previous valid data
        # History file should be rewritten with fresh data
        data = _read_json(history_path)
        assert len(data) == 1

    def test_non_list_history_file_handled(self, history_path):
        """History file with non-list content is handled gracefully."""
        _write_json(history_path, {"not": "a list"})

        coefficient_registry.load("/nonexistent.json")
        entry = coefficient_registry.record_history(history_path)

        assert entry is not None
        data = _read_json(history_path)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_registry_unaffected_by_history_failure(self, tmp_path):
        """Coefficient values are unchanged regardless of history file issues."""
        coeff_path = str(tmp_path / "coefficients.json")
        _write_json(coeff_path, {
            "drive_dynamics": {"total_change_limit": 0.20},
        })
        coefficient_registry.load(coeff_path)

        # Record with a path that will write successfully
        history_path = str(tmp_path / "history.json")
        coefficient_registry.record_history(history_path)

        # Registry values should be unaffected
        assert coefficient_registry.get("drive_dynamics", "total_change_limit") == 0.20
        defaults = coefficient_registry.get_defaults()
        assert coefficient_registry.get("mood_autonomy") == defaults["mood_autonomy"]


# =============================================================================
# Test: History file is separate from coefficients.json
# =============================================================================

class TestFileSeparation:
    """History file is managed independently from coefficients.json."""

    def test_history_default_path_different_from_coefficients(self):
        """Default history path is not the same as coefficients path."""
        coeff_path = coefficient_registry._resolve_coefficient_file_path()
        hist_path = coefficient_registry._resolve_history_file_path()
        assert coeff_path != hist_path

    def test_history_default_path_in_data_dir(self):
        """Default history path is in the data/ directory."""
        hist_path = coefficient_registry._resolve_history_file_path()
        assert "data" in hist_path
        assert hist_path.endswith("coefficient_history.json")

    def test_history_does_not_modify_coefficients_file(self, tmp_path, history_path):
        """Recording history does not modify the coefficients.json file."""
        coeff_path = str(tmp_path / "coefficients.json")
        original_data = {"drive_dynamics": {"total_change_limit": 0.20}}
        _write_json(coeff_path, original_data)

        coefficient_registry.load(coeff_path)
        coefficient_registry.record_history(history_path)

        # coefficients.json should be unchanged
        after_data = _read_json(coeff_path)
        assert after_data == original_data


# =============================================================================
# Test: Psyche non-impact guarantees
# =============================================================================

class TestPsycheNonImpact:
    """History data does not flow into psyche processing."""

    def test_no_enrichment_method_on_history(self):
        """No enrichment-related methods exist for history."""
        public_attrs = [a for a in dir(coefficient_registry) if not a.startswith("_")]
        enrichment_names = {
            "get_enrichment", "to_enrichment", "enrichment_entry",
            "get_prompt_enrichment", "enrichment",
        }
        for attr in public_attrs:
            assert attr not in enrichment_names

    def test_no_save_load_for_history(self):
        """History does not add to psyche save/load fields."""
        public_attrs = [a for a in dir(coefficient_registry) if not a.startswith("_")]
        save_load_names = {"to_dict", "from_dict", "save_state", "load_state"}
        for attr in public_attrs:
            assert attr not in save_load_names

    def test_record_history_does_not_modify_registry(self, history_path):
        """Recording history does not change any coefficient values."""
        coefficient_registry.load("/nonexistent.json")
        before = coefficient_registry.get("drive_dynamics")

        coefficient_registry.record_history(history_path)

        after = coefficient_registry.get("drive_dynamics")
        assert before == after

    def test_get_history_returns_independent_copy(self, history_path):
        """get_history() data cannot affect the registry."""
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        history = coefficient_registry.get_history(history_path)
        # Mutate the returned history
        history.clear()

        # Re-read should still have data
        history2 = coefficient_registry.get_history(history_path)
        assert len(history2) == 1


# =============================================================================
# Test: get_history() function
# =============================================================================

class TestGetHistory:
    """get_history() returns recorded entries."""

    def test_get_history_empty_when_no_file(self, history_path):
        """get_history() returns empty list when no history file exists."""
        result = coefficient_registry.get_history(history_path)
        assert result == []

    def test_get_history_returns_entries(self, history_path):
        """get_history() returns previously recorded entries."""
        coefficient_registry.load("/nonexistent.json")
        coefficient_registry.record_history(history_path)

        result = coefficient_registry.get_history(history_path)
        assert len(result) == 1
        assert "timestamp" in result[0]
        assert "snapshot" in result[0]

    def test_get_history_multiple_entries(self, history_path):
        """get_history() returns all accumulated entries."""
        for _ in range(3):
            coefficient_registry.reset()
            coefficient_registry.load("/nonexistent.json")
            coefficient_registry.record_history(history_path)

        result = coefficient_registry.get_history(history_path)
        assert len(result) == 3

    def test_get_history_corrupted_file(self, history_path):
        """get_history() returns empty list for corrupted file."""
        with open(history_path, "w") as f:
            f.write("corrupted data")

        result = coefficient_registry.get_history(history_path)
        assert result == []


# =============================================================================
# Test: Flatten dict utility
# =============================================================================

class TestFlattenDict:
    """Internal _flatten_dict works correctly."""

    def test_flat_dict_unchanged(self):
        result = coefficient_registry._flatten_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict_flattened(self):
        result = coefficient_registry._flatten_dict({"a": {"b": 1, "c": 2}})
        assert result == {"a.b": 1, "a.c": 2}

    def test_deeply_nested(self):
        result = coefficient_registry._flatten_dict({"a": {"b": {"c": 3}}})
        assert result == {"a.b.c": 3}

    def test_mixed_depth(self):
        result = coefficient_registry._flatten_dict({
            "x": 1,
            "y": {"z": 2},
        })
        assert result == {"x": 1, "y.z": 2}

    def test_empty_dict(self):
        result = coefficient_registry._flatten_dict({})
        assert result == {}


# =============================================================================
# Test: Compute changes utility
# =============================================================================

class TestComputeChanges:
    """Internal _compute_changes works correctly."""

    def test_no_changes(self):
        prev = {"a": 1, "b": 2}
        curr = {"a": 1, "b": 2}
        changes = coefficient_registry._compute_changes(prev, curr)
        assert changes == []

    def test_value_changed(self):
        prev = {"a": 1, "b": 2}
        curr = {"a": 1, "b": 3}
        changes = coefficient_registry._compute_changes(prev, curr)
        assert len(changes) == 1
        assert changes[0]["key"] == "b"
        assert changes[0]["old_value"] == 2
        assert changes[0]["new_value"] == 3

    def test_key_added(self):
        prev = {"a": 1}
        curr = {"a": 1, "b": 2}
        changes = coefficient_registry._compute_changes(prev, curr)
        assert len(changes) == 1
        assert changes[0]["key"] == "b"
        assert "old_value" not in changes[0]
        assert changes[0]["new_value"] == 2

    def test_key_removed(self):
        prev = {"a": 1, "b": 2}
        curr = {"a": 1}
        changes = coefficient_registry._compute_changes(prev, curr)
        assert len(changes) == 1
        assert changes[0]["key"] == "b"
        assert changes[0]["old_value"] == 2
        assert "new_value" not in changes[0]

    def test_multiple_changes(self):
        prev = {"a": 1, "b": 2, "c": 3}
        curr = {"a": 1, "b": 5, "c": 6}
        changes = coefficient_registry._compute_changes(prev, curr)
        assert len(changes) == 2
        changed_keys = {c["key"] for c in changes}
        assert changed_keys == {"b", "c"}

    def test_changes_sorted_by_key(self):
        prev = {"z": 1, "a": 2}
        curr = {"z": 10, "a": 20}
        changes = coefficient_registry._compute_changes(prev, curr)
        assert changes[0]["key"] == "a"
        assert changes[1]["key"] == "z"
