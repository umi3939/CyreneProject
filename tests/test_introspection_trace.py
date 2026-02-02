"""
Tests for psyche/introspection_trace.py - Introspective Trace Generation (内省ログ生成)

Verifies:
- READ-ONLY access to all states (no modifications)
- Structured JSON-compatible trace logs
- Contributing factors are "possible influences" not "definitive reasons"
- Append-only log history
"""

import pytest
import time
import json

from psyche.introspection_trace import (
    InfluenceDirection,
    FactorCategory,
    OutcomeType,
    ContributingFactor,
    EmotionSnapshot,
    ResponsibilitySnapshot,
    ValueOrientationSnapshot,
    DecisionSnapshot,
    TraceLog,
    IntrospectionConfig,
    IntrospectionSystem,
    generate_trace,
    create_introspection_system,
    create_config,
    get_trace_summary,
    traces_to_json,
    traces_from_json,
)

from psyche.state import EmotionVector
from psyche.value_orientation import ValueOrientation
from psyche.responsibility import ResponsibilityInfluence


# ── Snapshot Tests ──────────────────────────────────────────────────


class TestEmotionSnapshot:
    """Tests for EmotionSnapshot."""

    def test_default_values(self):
        """Default snapshot has zero values."""
        snapshot = EmotionSnapshot()
        assert snapshot.joy == 0.0
        assert snapshot.fear == 0.0
        assert snapshot.dominant_emotion == ""

    def test_serialization(self):
        """Snapshot can be serialized to dict."""
        snapshot = EmotionSnapshot(joy=0.5, fear=0.3, dominant_emotion="joy")
        data = snapshot.to_dict()

        assert data["joy"] == 0.5
        assert data["fear"] == 0.3
        assert data["dominant_emotion"] == "joy"

    def test_deserialization(self):
        """Snapshot can be restored from dict."""
        data = {"joy": 0.5, "fear": 0.3, "dominant_emotion": "joy"}
        snapshot = EmotionSnapshot.from_dict(data)

        assert snapshot.joy == 0.5
        assert snapshot.fear == 0.3


class TestResponsibilitySnapshot:
    """Tests for ResponsibilitySnapshot."""

    def test_serialization_roundtrip(self):
        """Snapshot survives serialization roundtrip."""
        original = ResponsibilitySnapshot(
            total_weight=0.7,
            active_count=3,
            caution_level=0.5,
        )
        data = original.to_dict()
        restored = ResponsibilitySnapshot.from_dict(data)

        assert restored.total_weight == original.total_weight
        assert restored.active_count == original.active_count
        assert restored.caution_level == original.caution_level


class TestValueOrientationSnapshot:
    """Tests for ValueOrientationSnapshot."""

    def test_captures_all_dimensions(self):
        """Snapshot captures all 5 dimensions."""
        snapshot = ValueOrientationSnapshot(
            dim_a=0.1, dim_b=0.2, dim_c=0.3, dim_d=0.4, dim_e=0.5
        )
        data = snapshot.to_dict()

        assert data["dim_a"] == 0.1
        assert data["dim_e"] == 0.5


class TestDecisionSnapshot:
    """Tests for DecisionSnapshot."""

    def test_captures_decision_details(self):
        """Snapshot captures decision details."""
        snapshot = DecisionSnapshot(
            policy_label="共感する",
            score=0.75,
            outcome_type=OutcomeType.SPEECH,
            tone="warm",
            is_silence=False,
        )

        assert snapshot.policy_label == "共感する"
        assert snapshot.outcome_type == OutcomeType.SPEECH

    def test_serialization(self):
        """Snapshot serializes outcome_type as string."""
        snapshot = DecisionSnapshot(outcome_type=OutcomeType.SILENCE)
        data = snapshot.to_dict()

        assert data["outcome_type"] == "silence"


# ── ContributingFactor Tests ────────────────────────────────────────


class TestContributingFactor:
    """Tests for ContributingFactor."""

    def test_creates_with_defaults(self):
        """Factor can be created with defaults."""
        factor = ContributingFactor()
        assert factor.category == FactorCategory.OTHER
        assert factor.direction == InfluenceDirection.NEUTRAL
        assert factor.contribution_strength == 0.0

    def test_creates_with_values(self):
        """Factor can be created with specific values."""
        factor = ContributingFactor(
            category=FactorCategory.EMOTION,
            name="fear",
            observed_value=0.7,
            direction=InfluenceDirection.POSITIVE,
            contribution_strength=0.8,
            description="High fear contributed to silence",
        )

        assert factor.category == FactorCategory.EMOTION
        assert factor.name == "fear"
        assert factor.observed_value == 0.7
        assert factor.direction == InfluenceDirection.POSITIVE

    def test_serialization_roundtrip(self):
        """Factor survives serialization roundtrip."""
        original = ContributingFactor(
            category=FactorCategory.FEAR,
            name="fear_index",
            observed_value=0.6,
            direction=InfluenceDirection.POSITIVE,
            contribution_strength=0.7,
        )
        data = original.to_dict()
        restored = ContributingFactor.from_dict(data)

        assert restored.category == original.category
        assert restored.name == original.name
        assert restored.observed_value == original.observed_value

    def test_description_is_possibility(self):
        """Description uses 'may have' language."""
        factor = ContributingFactor(
            description="High fear may have contributed to choosing silence"
        )
        # Should use possibility language, not definitive
        assert "may have" in factor.description or "contributed" in factor.description


# ── TraceLog Tests ──────────────────────────────────────────────────


class TestTraceLog:
    """Tests for TraceLog."""

    def test_creates_with_unique_id(self):
        """Each trace has a unique ID."""
        trace1 = TraceLog()
        trace2 = TraceLog()
        assert trace1.trace_id != trace2.trace_id

    def test_has_timestamp(self):
        """Trace has timestamp."""
        before = time.time()
        trace = TraceLog()
        after = time.time()

        assert before <= trace.timestamp <= after

    def test_serialization_is_json_compatible(self):
        """Trace serializes to JSON-compatible dict."""
        trace = TraceLog(
            generation=5,
            emotion_snapshot=EmotionSnapshot(joy=0.5),
            decision_snapshot=DecisionSnapshot(policy_label="test"),
            contributing_factors=[
                ContributingFactor(name="fear", observed_value=0.3)
            ],
        )

        data = trace.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        assert restored_data["generation"] == 5
        assert restored_data["emotion_snapshot"]["joy"] == 0.5

    def test_deserialization(self):
        """Trace can be restored from dict."""
        original = TraceLog(
            generation=10,
            emotion_snapshot=EmotionSnapshot(fear=0.7),
            contributing_factors=[
                ContributingFactor(name="fear", contribution_strength=0.8)
            ],
        )

        data = original.to_dict()
        restored = TraceLog.from_dict(data)

        assert restored.generation == 10
        assert restored.emotion_snapshot.fear == 0.7
        assert len(restored.contributing_factors) == 1

    def test_to_readable_output(self):
        """Trace produces human-readable output."""
        trace = TraceLog(
            decision_snapshot=DecisionSnapshot(
                policy_label="沈黙する",
                is_silence=True,
                outcome_type=OutcomeType.SILENCE,
            ),
            contributing_factors=[
                ContributingFactor(
                    category=FactorCategory.FEAR,
                    name="fear_index",
                    observed_value=0.7,
                    contribution_strength=0.8,
                    direction=InfluenceDirection.POSITIVE,
                )
            ],
        )

        readable = trace.to_readable()

        assert "Introspection Trace" in readable
        assert "沈黙する" in readable
        assert "fear" in readable.lower()
        # Should include possibility disclaimer
        assert "possibilities" in readable.lower() or "possible" in readable.lower()

    def test_get_top_factors(self):
        """Can get top factors by contribution strength."""
        trace = TraceLog(
            contributing_factors=[
                ContributingFactor(name="low", contribution_strength=0.2),
                ContributingFactor(name="high", contribution_strength=0.9),
                ContributingFactor(name="mid", contribution_strength=0.5),
            ]
        )

        top = trace.get_top_factors(2)

        assert len(top) == 2
        assert top[0].name == "high"
        assert top[1].name == "mid"

    def test_has_factor_category(self):
        """Can check for factor category presence."""
        trace = TraceLog(
            contributing_factors=[
                ContributingFactor(category=FactorCategory.EMOTION),
                ContributingFactor(category=FactorCategory.FEAR),
            ]
        )

        assert trace.has_factor_category(FactorCategory.EMOTION)
        assert trace.has_factor_category(FactorCategory.FEAR)
        assert not trace.has_factor_category(FactorCategory.RESPONSIBILITY)


# ── IntrospectionSystem Tests ───────────────────────────────────────


class TestIntrospectionSystem:
    """Tests for IntrospectionSystem."""

    def test_creates_with_default_config(self):
        """System creates with default config."""
        system = IntrospectionSystem()
        assert system.config is not None
        assert system.get_generation_count() == 0

    def test_generates_trace(self):
        """System generates trace from states."""
        system = IntrospectionSystem()

        trace = system.generate_trace(
            decision_outcome={"policy_label": "共感する", "_score": 0.7}
        )

        assert trace is not None
        assert trace.generation == 1
        assert trace.decision_snapshot.policy_label == "共感する"

    def test_increments_generation(self):
        """Generation counter increments with each trace."""
        system = IntrospectionSystem()

        trace1 = system.generate_trace()
        trace2 = system.generate_trace()
        trace3 = system.generate_trace()

        assert trace1.generation == 1
        assert trace2.generation == 2
        assert trace3.generation == 3

    def test_stores_history(self):
        """System stores trace history."""
        system = IntrospectionSystem()

        system.generate_trace()
        system.generate_trace()
        system.generate_trace()

        history = system.get_history()
        assert len(history) == 3

    def test_history_is_append_only(self):
        """History cannot be modified externally."""
        system = IntrospectionSystem()
        system.generate_trace()

        # Get history and modify it
        history = system.get_history()
        history.clear()

        # Internal history should be unchanged
        assert len(system.get_history()) == 1

    def test_get_recent_traces(self):
        """Can get recent traces."""
        system = IntrospectionSystem()

        for i in range(10):
            system.generate_trace()

        recent = system.get_recent_traces(3)
        assert len(recent) == 3
        assert recent[0].generation == 8
        assert recent[2].generation == 10


class TestReadOnlyAccess:
    """Tests verifying READ-ONLY access to states."""

    def test_emotion_state_not_modified(self):
        """Emotion state is not modified by trace generation."""
        emotion = EmotionVector(joy=0.5, fear=0.3)
        original_joy = emotion.joy
        original_fear = emotion.fear

        system = IntrospectionSystem()
        system.generate_trace(emotion_state=emotion)

        # Original state unchanged
        assert emotion.joy == original_joy
        assert emotion.fear == original_fear

    def test_value_orientation_not_modified(self):
        """Value orientation is not modified by trace generation."""
        orientation = ValueOrientation(dim_a=0.5, dim_b=-0.3)
        original_a = orientation.dim_a
        original_b = orientation.dim_b

        system = IntrospectionSystem()
        system.generate_trace(value_orientation=orientation)

        # Original state unchanged
        assert orientation.dim_a == original_a
        assert orientation.dim_b == original_b

    def test_decision_dict_not_modified(self):
        """Decision dict is not modified by trace generation."""
        decision = {
            "policy_label": "共感する",
            "_score": 0.7,
            "_tone": "warm",
        }
        original_decision = decision.copy()

        system = IntrospectionSystem()
        system.generate_trace(decision_outcome=decision)

        # Original dict unchanged
        assert decision == original_decision

    def test_snapshots_are_copies(self):
        """Snapshots are copies, not references."""
        emotion = EmotionVector(joy=0.5)

        system = IntrospectionSystem()
        trace = system.generate_trace(emotion_state=emotion)

        # Modify original
        emotion.joy = 0.9

        # Snapshot should still have original value
        assert trace.emotion_snapshot.joy == 0.5


# ── Factor Extraction Tests ─────────────────────────────────────────


class TestFactorExtraction:
    """Tests for factor extraction logic."""

    def test_extracts_high_emotion_factors(self):
        """Extracts factors for high emotions."""
        emotion = EmotionVector(fear=0.6)
        config = IntrospectionConfig(emotion_threshold=0.2)

        system = IntrospectionSystem(config)
        trace = system.generate_trace(
            emotion_state=emotion,
            decision_outcome={"policy_label": "沈黙する", "_is_silence": True},
        )

        fear_factors = [f for f in trace.contributing_factors if f.name == "fear"]
        assert len(fear_factors) > 0
        assert fear_factors[0].observed_value == 0.6

    def test_extracts_value_orientation_factors(self):
        """Extracts factors for significant value dimensions."""
        orientation = ValueOrientation(dim_a=0.5)
        config = IntrospectionConfig(value_orientation_threshold=0.15)

        system = IntrospectionSystem(config)
        trace = system.generate_trace(
            value_orientation=orientation,
            decision_outcome={"policy_label": "test"},
        )

        dim_factors = [f for f in trace.contributing_factors if f.name == "dim_a"]
        assert len(dim_factors) > 0

    def test_respects_threshold(self):
        """Does not extract factors below threshold."""
        emotion = EmotionVector(joy=0.1)  # Below threshold
        config = IntrospectionConfig(emotion_threshold=0.2)

        system = IntrospectionSystem(config)
        trace = system.generate_trace(emotion_state=emotion)

        joy_factors = [f for f in trace.contributing_factors if f.name == "joy"]
        assert len(joy_factors) == 0

    def test_builds_factor_summary(self):
        """Builds factor summary by category."""
        emotion = EmotionVector(fear=0.5, sadness=0.4)
        orientation = ValueOrientation(dim_a=0.3)

        system = IntrospectionSystem()
        trace = system.generate_trace(
            emotion_state=emotion,
            value_orientation=orientation,
            decision_outcome={"policy_label": "test"},
        )

        # Should have counts by category
        assert "emotion" in trace.factor_summary
        assert trace.factor_summary["emotion"] >= 1


class TestDescriptionLanguage:
    """Tests verifying description uses possibility language."""

    def test_descriptions_use_possibility(self):
        """Factor descriptions use 'may have' language."""
        emotion = EmotionVector(fear=0.7)

        system = IntrospectionSystem()
        trace = system.generate_trace(
            emotion_state=emotion,
            decision_outcome={"policy_label": "沈黙する", "_is_silence": True},
        )

        for factor in trace.contributing_factors:
            if factor.description:
                # Should NOT use definitive language
                assert "definitely" not in factor.description.lower()
                assert "certainly" not in factor.description.lower()
                # Should use possibility language
                assert ("may" in factor.description.lower() or
                        "possible" in factor.description.lower() or
                        "could" in factor.description.lower() or
                        factor.description == "")


# ── Convenience Function Tests ──────────────────────────────────────


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_trace_function(self):
        """generate_trace convenience function works."""
        trace = generate_trace(
            decision_outcome={"policy_label": "test", "_score": 0.5}
        )

        assert trace is not None
        assert trace.decision_snapshot.policy_label == "test"

    def test_create_introspection_system(self):
        """create_introspection_system function works."""
        system = create_introspection_system()
        assert isinstance(system, IntrospectionSystem)

    def test_create_config(self):
        """create_config function works."""
        config = create_config(
            emotion_threshold=0.3,
            max_factors=5,
        )

        assert config.emotion_threshold == 0.3
        assert config.max_factors == 5

    def test_get_trace_summary(self):
        """get_trace_summary returns brief summary."""
        trace = TraceLog(
            decision_snapshot=DecisionSnapshot(
                policy_label="共感する",
                outcome_type=OutcomeType.SPEECH,
            ),
            contributing_factors=[
                ContributingFactor(name="fear"),
                ContributingFactor(name="sadness"),
            ],
        )

        summary = get_trace_summary(trace)

        assert "共感する" in summary
        assert "speech" in summary
        assert "fear" in summary

    def test_traces_to_json(self):
        """traces_to_json converts list to JSON format."""
        traces = [
            TraceLog(generation=1),
            TraceLog(generation=2),
        ]

        data = traces_to_json(traces)

        assert len(data) == 2
        assert data[0]["generation"] == 1
        assert data[1]["generation"] == 2

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None

    def test_traces_from_json(self):
        """traces_from_json restores list from JSON format."""
        data = [
            {"generation": 1, "trace_id": "test1"},
            {"generation": 2, "trace_id": "test2"},
        ]

        traces = traces_from_json(data)

        assert len(traces) == 2
        assert traces[0].generation == 1
        assert traces[1].generation == 2


# ── Integration Tests ───────────────────────────────────────────────


class TestIntegration:
    """Integration tests."""

    def test_full_trace_workflow(self):
        """Complete workflow from states to trace."""
        # Setup states
        emotion = EmotionVector(joy=0.3, fear=0.6)
        orientation = ValueOrientation(dim_a=-0.3, confidence_a=0.4)
        decision = {
            "policy_label": "沈黙する",
            "_score": 0.7,
            "_is_silence": True,
            "_tone": "neutral",
        }

        # Generate trace
        system = IntrospectionSystem()
        trace = system.generate_trace(
            emotion_state=emotion,
            value_orientation=orientation,
            decision_outcome=decision,
        )

        # Verify snapshots
        assert trace.emotion_snapshot.joy == 0.3
        assert trace.emotion_snapshot.fear == 0.6
        assert trace.value_orientation_snapshot.dim_a == -0.3
        assert trace.decision_snapshot.is_silence is True

        # Verify factors extracted
        assert len(trace.contributing_factors) > 0

        # Verify JSON serializable
        data = trace.to_dict()
        json_str = json.dumps(data)
        restored = TraceLog.from_dict(json.loads(json_str))
        assert restored.generation == trace.generation

    def test_multiple_traces_build_history(self):
        """Multiple traces build analyzable history."""
        system = IntrospectionSystem()

        # Generate several traces
        for i in range(5):
            emotion = EmotionVector(fear=0.1 * i)
            system.generate_trace(
                emotion_state=emotion,
                decision_outcome={"policy_label": f"decision_{i}"},
            )

        # Analyze history
        history = system.get_history()
        assert len(history) == 5

        # Can export to JSON
        json_data = traces_to_json(history)
        assert len(json_data) == 5

        # Can restore from JSON
        restored = traces_from_json(json_data)
        assert len(restored) == 5
        assert restored[4].decision_snapshot.policy_label == "decision_4"
