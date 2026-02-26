"""
tests/test_state.py - Tests for PsycheState serialization (to_dict / from_dict).

Verifies:
1. FearIndex 4-pillar values survive save/load roundtrip
2. Backward compatibility with old format (fear_index as float)
3. FearIndex=None handling
"""

from __future__ import annotations

import pytest

from psyche.state import PsycheState, EmotionVector, DriveVector, Mood
from psyche.pillars import FearIndex


class TestFearIndexSerialization:
    """Verify FearIndex pillar values are preserved through to_dict/from_dict."""

    def test_roundtrip_preserves_pillar_risks(self):
        """4 pillar risk values must survive a save/load cycle."""
        fi = FearIndex(
            identity_risk=0.4,
            attachment_risk=0.6,
            continuity_risk=0.3,
            projection_risk=0.8,
        )
        state = PsycheState(fear_index=fi)

        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        assert restored.fear_index is not None
        assert restored.fear_index.identity_risk == pytest.approx(0.4, abs=1e-3)
        assert restored.fear_index.attachment_risk == pytest.approx(0.6, abs=1e-3)
        assert restored.fear_index.continuity_risk == pytest.approx(0.3, abs=1e-3)
        assert restored.fear_index.projection_risk == pytest.approx(0.8, abs=1e-3)

    def test_roundtrip_preserves_composite_value(self):
        """The composite fear value must match after roundtrip."""
        fi = FearIndex(
            identity_risk=0.4,
            attachment_risk=0.6,
            continuity_risk=0.3,
            projection_risk=0.8,
        )
        state = PsycheState(fear_index=fi)
        original_value = fi.value

        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        assert restored.fear_level == pytest.approx(original_value, abs=1e-3)

    def test_to_dict_fear_index_is_dict(self):
        """to_dict should serialize FearIndex as a dict with pillar risks."""
        fi = FearIndex(
            identity_risk=0.1,
            attachment_risk=0.2,
            continuity_risk=0.3,
            projection_risk=0.4,
        )
        state = PsycheState(fear_index=fi)

        d = state.to_dict()

        assert isinstance(d["fear_index"], dict)
        assert "identity_risk" in d["fear_index"]
        assert "attachment_risk" in d["fear_index"]
        assert "continuity_risk" in d["fear_index"]
        assert "projection_risk" in d["fear_index"]
        assert "value" in d["fear_index"]

    def test_backward_compat_float_fear_index(self):
        """Old format with fear_index as float should still load."""
        d = {
            "emotions": {},
            "drives": {},
            "mood": {"valence": 0.0, "arousal": 0.3},
            "fear_index": 0.42,
            "loss_aversion": 0.3,
            "last_updated": "2026-01-01T00:00:00",
        }
        restored = PsycheState.from_dict(d)

        # Should create a FearIndex with all zeros (old behavior)
        assert restored.fear_index is not None
        assert restored.fear_index.identity_risk == 0.0
        assert restored.fear_index.attachment_risk == 0.0
        assert restored.fear_index.continuity_risk == 0.0
        assert restored.fear_index.projection_risk == 0.0

    def test_backward_compat_int_fear_index(self):
        """Old format with fear_index as int (0) should still load."""
        d = {
            "emotions": {},
            "drives": {},
            "mood": {"valence": 0.0, "arousal": 0.3},
            "fear_index": 0,
            "loss_aversion": 0.3,
            "last_updated": "2026-01-01T00:00:00",
        }
        restored = PsycheState.from_dict(d)

        assert restored.fear_index is not None
        assert restored.fear_index.identity_risk == 0.0

    def test_none_fear_index_to_dict(self):
        """PsycheState with fear_index=None serializes as 0.0."""
        state = PsycheState(fear_index=None)
        d = state.to_dict()

        assert d["fear_index"] == 0.0

    def test_none_fear_index_roundtrip(self):
        """PsycheState with fear_index=None survives roundtrip."""
        state = PsycheState(fear_index=None)
        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        # from_dict creates a FearIndex with zeros from numeric value
        assert restored.fear_index is not None
        assert restored.fear_level == 0.0

    def test_all_zero_risks_roundtrip(self):
        """All-zero FearIndex survives roundtrip correctly."""
        fi = FearIndex(
            identity_risk=0.0,
            attachment_risk=0.0,
            continuity_risk=0.0,
            projection_risk=0.0,
        )
        state = PsycheState(fear_index=fi)

        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        assert restored.fear_index is not None
        assert restored.fear_index.identity_risk == 0.0
        assert restored.fear_index.attachment_risk == 0.0
        assert restored.fear_index.continuity_risk == 0.0
        assert restored.fear_index.projection_risk == 0.0
        assert restored.fear_level == 0.0

    def test_asymmetric_risks_roundtrip(self):
        """Asymmetric risk values (only one pillar high) survive roundtrip."""
        fi = FearIndex(
            identity_risk=0.9,
            attachment_risk=0.0,
            continuity_risk=0.0,
            projection_risk=0.0,
        )
        state = PsycheState(fear_index=fi)

        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        assert restored.fear_index.identity_risk == pytest.approx(0.9, abs=1e-3)
        assert restored.fear_index.attachment_risk == 0.0
        assert restored.fear_index.continuity_risk == 0.0
        assert restored.fear_index.projection_risk == 0.0
        assert restored.fear_index.dominant_fear == "identity"

    def test_partial_dict_fear_index(self):
        """Dict with partial pillar keys uses defaults for missing ones."""
        d = {
            "emotions": {},
            "drives": {},
            "mood": {"valence": 0.0, "arousal": 0.3},
            "fear_index": {"identity_risk": 0.5, "value": 0.15},
            "loss_aversion": 0.3,
            "last_updated": "2026-01-01T00:00:00",
        }
        restored = PsycheState.from_dict(d)

        assert restored.fear_index is not None
        assert restored.fear_index.identity_risk == 0.5
        assert restored.fear_index.attachment_risk == 0.0
        assert restored.fear_index.continuity_risk == 0.0
        assert restored.fear_index.projection_risk == 0.0


class TestPsycheStateToFromDict:
    """General PsycheState serialization tests."""

    def test_full_roundtrip(self):
        """Full PsycheState roundtrip preserves all fields."""
        fi = FearIndex(
            identity_risk=0.3,
            attachment_risk=0.5,
            continuity_risk=0.2,
            projection_risk=0.7,
        )
        state = PsycheState(
            emotions=EmotionVector(joy=0.8, sorrow=0.2, fear=0.1),
            drives=DriveVector(social=0.7, curiosity=0.3, expression=0.9),
            mood=Mood(valence=0.4, arousal=0.6),
            fear_index=fi,
            loss_aversion=0.5,
            last_updated="2026-02-27T12:00:00",
        )

        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        assert restored.emotions.joy == 0.8
        assert restored.emotions.sorrow == 0.2
        assert restored.drives.social == 0.7
        assert restored.mood.valence == 0.4
        assert restored.mood.arousal == 0.6
        assert restored.fear_index.identity_risk == pytest.approx(0.3, abs=1e-3)
        assert restored.fear_index.attachment_risk == pytest.approx(0.5, abs=1e-3)
        assert restored.fear_index.continuity_risk == pytest.approx(0.2, abs=1e-3)
        assert restored.fear_index.projection_risk == pytest.approx(0.7, abs=1e-3)
        assert restored.loss_aversion == 0.5
        assert restored.last_updated == "2026-02-27T12:00:00"

    def test_default_state_roundtrip(self):
        """Default PsycheState roundtrip works correctly."""
        state = PsycheState()
        d = state.to_dict()
        restored = PsycheState.from_dict(d)

        assert restored.emotions.joy == 0.0
        assert restored.drives.social == 0.5
        assert restored.mood.valence == 0.0
        assert restored.loss_aversion == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
