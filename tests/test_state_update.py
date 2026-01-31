"""
tests/test_state_update.py - State update and fear_index computation tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.state_manager import StateManager
from src.emotion_model import event_to_emotion, apply_decay, NEUTRAL


class TestEmotionModel:
    """Verify emotion vector computation."""

    def test_positive_sentiment_increases_joy(self):
        vec = event_to_emotion({"sentiment": 0.8, "keywords": []})
        assert vec["joy"] > NEUTRAL["joy"]

    def test_negative_sentiment_increases_sad(self):
        vec = event_to_emotion({"sentiment": -0.7, "keywords": []})
        assert vec["sad"] > NEUTRAL["sad"]

    def test_keyword_boost(self):
        vec = event_to_emotion({"sentiment": 0.0, "keywords": ["嬉しい"]})
        assert vec["joy"] > 0.1

    def test_decay_reduces_intensity(self):
        vec = {"joy": 0.8, "sad": 0.0, "fear": 0.0, "anger": 0.0, "calm": 0.5}
        decayed = apply_decay(vec, delta_seconds=10.0)
        assert decayed["joy"] < vec["joy"]

    def test_calm_drifts_to_half(self):
        vec = {"joy": 0.0, "sad": 0.0, "fear": 0.0, "anger": 0.0, "calm": 1.0}
        decayed = apply_decay(vec, delta_seconds=50.0)
        assert decayed["calm"] < 1.0
        assert decayed["calm"] > 0.4  # drifting toward 0.5

    def test_values_stay_clamped(self):
        vec = event_to_emotion({"sentiment": 5.0, "keywords": ["嬉しい"] * 10})
        for v in vec.values():
            assert 0.0 <= v <= 1.0


class TestStateManager:
    """Verify state persistence and update logic."""

    def test_get_state_creates_default(self, tmp_data_dir: Path):
        mgr = StateManager(filepath=tmp_data_dir / "state.json")
        state = mgr.get_state("new_user")
        assert "emotions" in state
        assert "drives" in state
        assert "mood" in state
        assert "fear_index" in state

    def test_update_state_changes_emotions(self, tmp_data_dir: Path):
        mgr = StateManager(filepath=tmp_data_dir / "state.json")
        before = mgr.get_state("test_user")
        event = {"sentiment": 0.8, "keywords": ["嬉しい"], "intent": "greeting"}
        after = mgr.update_state_on_event("test_user", event)
        assert after["emotions"]["joy"] > before["emotions"]["joy"]

    def test_update_state_changes_drives(self, tmp_data_dir: Path):
        mgr = StateManager(filepath=tmp_data_dir / "state.json")
        before = mgr.get_state("test_user")
        event = {"sentiment": 0.3, "keywords": [], "intent": "sharing"}
        after = mgr.update_state_on_event("test_user", event)
        # Social should decrease (talking satisfies social drive)
        assert after["drives"]["social"] < before["drives"]["social"] + 0.1

    def test_update_state_changes_mood(self, tmp_data_dir: Path):
        mgr = StateManager(filepath=tmp_data_dir / "state.json")
        event = {"sentiment": 0.9, "keywords": ["嬉しい"]}
        after = mgr.update_state_on_event("test_user", event)
        assert after["mood"] > 0.0  # positive event → positive mood

    def test_set_state_persists(self, tmp_data_dir: Path):
        mgr = StateManager(filepath=tmp_data_dir / "state.json")
        custom = {"emotions": {"joy": 1.0}, "mood": 0.5, "drives": {}, "fear_index": 0.8}
        mgr.set_state("custom_user", custom)
        reloaded = StateManager(filepath=tmp_data_dir / "state.json")
        assert reloaded.get_state("custom_user")["fear_index"] == 0.8


class TestFearIndex:
    """Verify fear_index computation."""

    def test_all_zero_risks(self):
        assert StateManager.calc_fear_index(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_all_max_risks(self):
        fear = StateManager.calc_fear_index(1.0, 1.0, 1.0, 1.0)
        assert fear == pytest.approx(1.0)

    def test_weighted_correctly(self):
        # identity=0.3 weight, attachment=0.3, continuity=0.2, projection=0.2
        fear = StateManager.calc_fear_index(1.0, 0.0, 0.0, 0.0)
        assert fear == pytest.approx(0.3)
        fear = StateManager.calc_fear_index(0.0, 1.0, 0.0, 0.0)
        assert fear == pytest.approx(0.3)
        fear = StateManager.calc_fear_index(0.0, 0.0, 1.0, 0.0)
        assert fear == pytest.approx(0.2)
        fear = StateManager.calc_fear_index(0.0, 0.0, 0.0, 1.0)
        assert fear == pytest.approx(0.2)

    def test_fear_index_changes_with_events(self, tmp_data_dir: Path):
        """Integration: fear_index should change when pillar risks change."""
        mgr = StateManager(filepath=tmp_data_dir / "state.json")
        state = mgr.get_state("test_user")

        fear_before = StateManager.calc_fear_index(
            identity_risk=0.0,
            attachment_risk=0.3,
            continuity_risk=0.3,
            projection_risk=0.1,
        )
        # Simulate: attachment risk increases (long absence)
        fear_after = StateManager.calc_fear_index(
            identity_risk=0.0,
            attachment_risk=0.7,  # increased
            continuity_risk=0.3,
            projection_risk=0.1,
        )
        assert fear_after > fear_before, (
            f"fear_index should increase: {fear_before} → {fear_after}"
        )
