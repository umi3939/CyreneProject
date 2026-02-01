"""
tests/test_self_reference.py - Tests for self-reference loop structure

Verifies:
1. State acquisition (read-only)
2. State summarization (coarse compression)
3. Self-tag generation
4. Self-reference execution (circulation)
5. Integration with decision bias
6. No state modification
"""

import pytest

from psyche.self_reference import (
    SelfTag,
    SelfTagCategory,
    SelfReferenceConfig,
    SelfReferenceState,
    acquire_self_reference_targets,
    summarize_state,
    generate_self_tags,
    execute_self_reference,
    apply_self_tags_to_bias,
    get_self_reference_summary,
    create_self_reference_state,
)
from psyche.state import PsycheState, EmotionVector, Mood
from psyche.responsibility import ResponsibilityState
from psyche.short_term_memory import ShortTermMemory
from psyche.dynamics import DynamicsState, DynamicsPhase, create_dynamics_state, enter_peak
from psyche.decision_bias import DecisionBias, create_neutral_bias


class TestSelfTag:
    """Tests for SelfTag structure."""

    def test_create_self_tag(self):
        """Self-tag can be created with basic fields."""
        tag = SelfTag(
            category=SelfTagCategory.EMOTION,
            label="test_tag",
            source_value=0.5,
            weight=1.0,
        )

        assert tag.category == SelfTagCategory.EMOTION
        assert tag.label == "test_tag"
        assert tag.source_value == 0.5
        assert tag.weight == 1.0

    def test_tag_serialization(self):
        """Tag survives serialization roundtrip."""
        tag = SelfTag(
            category=SelfTagCategory.FEAR,
            label="fear_present",
            source_value=0.7,
            weight=0.8,
        )

        data = tag.to_dict()
        restored = SelfTag.from_dict(data)

        assert restored.category == tag.category
        assert restored.label == tag.label
        assert restored.source_value == tag.source_value
        assert restored.weight == tag.weight


class TestSelfReferenceState:
    """Tests for SelfReferenceState structure."""

    def test_create_empty_state(self):
        """Empty state can be created."""
        state = create_self_reference_state()

        assert len(state.tags) == 0
        assert state.reference_count == 0

    def test_get_tags_by_category(self):
        """Tags can be filtered by category."""
        tags = [
            SelfTag(category=SelfTagCategory.EMOTION, label="emo1"),
            SelfTag(category=SelfTagCategory.FEAR, label="fear1"),
            SelfTag(category=SelfTagCategory.EMOTION, label="emo2"),
        ]
        state = SelfReferenceState(tags=tags)

        emotion_tags = state.get_tags_by_category(SelfTagCategory.EMOTION)
        assert len(emotion_tags) == 2

        fear_tags = state.get_tags_by_category(SelfTagCategory.FEAR)
        assert len(fear_tags) == 1

    def test_has_tag(self):
        """Tag presence can be checked by label."""
        tags = [
            SelfTag(category=SelfTagCategory.EMOTION, label="test_label"),
        ]
        state = SelfReferenceState(tags=tags)

        assert state.has_tag("test_label") is True
        assert state.has_tag("nonexistent") is False

    def test_state_serialization(self):
        """State survives serialization roundtrip."""
        tags = [
            SelfTag(category=SelfTagCategory.EMOTION, label="test"),
        ]
        state = SelfReferenceState(tags=tags, reference_count=5)

        data = state.to_dict()
        restored = SelfReferenceState.from_dict(data)

        assert len(restored.tags) == 1
        assert restored.reference_count == 5


class TestAcquireTargets:
    """Tests for state acquisition (read-only)."""

    def test_acquire_from_empty(self):
        """Acquisition with no inputs returns empty dict."""
        targets = acquire_self_reference_targets()
        assert targets == {}

    def test_acquire_from_psyche_state(self):
        """Acquisition extracts PsycheState values."""
        state = PsycheState(
            emotions=EmotionVector(joy=0.5, anger=0.3),
            mood=Mood(valence=-0.2, arousal=0.6),
        )

        targets = acquire_self_reference_targets(psyche_state=state)

        assert "emotions" in targets
        assert targets["emotions"]["joy"] == 0.5
        assert "mood_valence" in targets
        assert targets["mood_valence"] == -0.2

    def test_acquire_from_responsibility(self):
        """Acquisition extracts ResponsibilityState values."""
        resp = ResponsibilityState(
            total_weight=0.8,
            accumulated_harm=0.5,
        )

        targets = acquire_self_reference_targets(responsibility_state=resp)

        assert "responsibility" in targets
        assert targets["responsibility"]["total_weight"] == 0.8
        assert targets["responsibility"]["accumulated_harm"] == 0.5

    def test_acquire_from_short_term_memory(self):
        """Acquisition extracts ShortTermMemory values."""
        memory = ShortTermMemory()
        memory = memory.add_stimulus(
            source_text="test",
            topics=["topic"],
            emotion_label="happy",
            intent="sharing",
            raw_intensity=0.6,
            valence=0.5,
        )

        targets = acquire_self_reference_targets(short_term_memory=memory)

        assert "short_term_memory" in targets
        assert targets["short_term_memory"]["entry_count"] == 1

    def test_acquire_from_dynamics(self):
        """Acquisition extracts DynamicsState values."""
        dynamics = create_dynamics_state()
        dynamics = enter_peak(dynamics, "joy", 0.8)

        targets = acquire_self_reference_targets(dynamics_state=dynamics)

        assert "dynamics" in targets
        assert targets["dynamics"]["phase"] == "peak"
        assert targets["dynamics"]["peak_emotion"] == "joy"

    def test_acquire_does_not_modify_state(self):
        """Acquisition is read-only - does not modify input."""
        state = PsycheState(
            emotions=EmotionVector(joy=0.5),
        )
        original_joy = state.emotions.joy

        acquire_self_reference_targets(psyche_state=state)

        assert state.emotions.joy == original_joy


class TestSummarizeState:
    """Tests for state summarization."""

    def test_summarize_empty(self):
        """Empty targets produce empty summary."""
        summary = summarize_state({})
        assert summary == {}

    def test_summarize_emotions(self):
        """Emotions are summarized to dominant emotion."""
        targets = {
            "emotions": {"joy": 0.3, "anger": 0.8, "sorrow": 0.1},
        }

        summary = summarize_state(targets)

        assert summary["dominant_emotion"] == "anger"
        assert summary["dominant_emotion_value"] == 0.8

    def test_summarize_mood(self):
        """Mood valence is preserved in summary."""
        targets = {
            "mood_valence": -0.5,
            "mood_arousal": 0.7,
        }

        summary = summarize_state(targets)

        assert summary["mood_valence"] == -0.5
        assert summary["mood_arousal"] == 0.7

    def test_custom_summarize_function(self):
        """Custom summarize function is used when provided."""
        def custom_summarize(targets):
            return {"custom": True}

        config = SelfReferenceConfig(summarize_function=custom_summarize)
        summary = summarize_state({"test": 1}, config)

        assert summary == {"custom": True}


class TestGenerateTags:
    """Tests for self-tag generation."""

    def test_generate_from_empty(self):
        """Empty summary produces no tags."""
        tags = generate_self_tags({})
        assert len(tags) == 0

    def test_generate_emotion_tag(self):
        """Dominant emotion generates a tag."""
        summary = {
            "dominant_emotion": "joy",
            "dominant_emotion_value": 0.7,
        }

        tags = generate_self_tags(summary)

        emotion_tags = [t for t in tags if t.category == SelfTagCategory.EMOTION]
        assert len(emotion_tags) >= 1
        assert any(t.label == "dominant_joy" for t in emotion_tags)

    def test_generate_mood_tags(self):
        """Mood generates positive/negative/neutral tag."""
        # Negative mood
        tags_neg = generate_self_tags({"mood_valence": -0.5})
        assert any(t.label == "negative_mood" for t in tags_neg)

        # Positive mood
        tags_pos = generate_self_tags({"mood_valence": 0.5})
        assert any(t.label == "positive_mood" for t in tags_pos)

        # Neutral mood
        tags_neu = generate_self_tags({"mood_valence": 0.0})
        assert any(t.label == "neutral_mood" for t in tags_neu)

    def test_generate_fear_tag(self):
        """Fear level generates a tag."""
        tags = generate_self_tags({"fear_level": 0.6})

        fear_tags = [t for t in tags if t.category == SelfTagCategory.FEAR]
        assert len(fear_tags) == 1
        assert fear_tags[0].label == "fear_present"

    def test_generate_responsibility_tags(self):
        """Responsibility generates tags."""
        summary = {
            "responsibility": {
                "total_weight": 0.5,
                "accumulated_harm": 0.3,
            }
        }

        tags = generate_self_tags(summary)

        resp_tags = [t for t in tags if t.category == SelfTagCategory.RESPONSIBILITY]
        assert len(resp_tags) == 2
        assert any(t.label == "responsibility_present" for t in resp_tags)
        assert any(t.label == "harm_accumulated" for t in resp_tags)

    def test_custom_tag_generator(self):
        """Custom tag generator is used when provided."""
        def custom_generator(summary):
            return [SelfTag(category=SelfTagCategory.EMOTION, label="custom")]

        config = SelfReferenceConfig(tag_generator=custom_generator)
        tags = generate_self_tags({"test": 1}, config)

        assert len(tags) == 1
        assert tags[0].label == "custom"


class TestExecuteSelfReference:
    """Tests for full self-reference loop execution."""

    def test_execute_with_all_inputs(self):
        """Full execution with all state inputs."""
        psyche = PsycheState(
            emotions=EmotionVector(anger=0.7),
            mood=Mood(valence=-0.4),
        )
        resp = ResponsibilityState(total_weight=0.5)
        memory = ShortTermMemory()
        dynamics = create_dynamics_state()

        result = execute_self_reference(
            psyche_state=psyche,
            responsibility_state=resp,
            short_term_memory=memory,
            dynamics_state=dynamics,
        )

        assert isinstance(result, SelfReferenceState)
        assert len(result.tags) > 0
        assert result.reference_count == 0

    def test_execute_tracks_circulation(self):
        """Reference count increments with circulation."""
        psyche = PsycheState()

        # First execution
        state1 = execute_self_reference(psyche_state=psyche)
        assert state1.reference_count == 0

        # Second execution (circulation)
        state2 = execute_self_reference(
            psyche_state=psyche,
            previous_state=state1,
        )
        assert state2.reference_count == 1

        # Third execution
        state3 = execute_self_reference(
            psyche_state=psyche,
            previous_state=state2,
        )
        assert state3.reference_count == 2

    def test_execute_does_not_modify_inputs(self):
        """Execution does not modify any input state."""
        psyche = PsycheState(emotions=EmotionVector(joy=0.5))
        original_joy = psyche.emotions.joy

        execute_self_reference(psyche_state=psyche)

        assert psyche.emotions.joy == original_joy


class TestApplyToDecisionBias:
    """Tests for applying self-tags to decision bias."""

    def test_apply_to_none_bias(self):
        """Applying to None returns None."""
        state = SelfReferenceState(tags=[
            SelfTag(category=SelfTagCategory.EMOTION, label="test"),
        ])

        result = apply_self_tags_to_bias(None, state)
        assert result is None

    def test_apply_with_no_tags(self):
        """No tags returns original bias unchanged."""
        bias = create_neutral_bias()
        state = SelfReferenceState(tags=[])

        result = apply_self_tags_to_bias(bias, state)
        assert result is bias

    def test_apply_negative_mood_tag(self):
        """Negative mood tag adjusts valence."""
        bias = DecisionBias(valence_bias=0.0)
        state = SelfReferenceState(tags=[
            SelfTag(
                category=SelfTagCategory.EMOTION,
                label="negative_mood",
                source_value=-0.5,
                weight=1.0,
            ),
        ])

        result = apply_self_tags_to_bias(bias, state)

        # Negative mood should have adjusted valence
        assert result.valence_bias != bias.valence_bias

    def test_apply_does_not_modify_original(self):
        """Application creates new bias, doesn't modify original."""
        original_valence = 0.3
        bias = DecisionBias(valence_bias=original_valence)
        state = SelfReferenceState(tags=[
            SelfTag(
                category=SelfTagCategory.EMOTION,
                label="negative_mood",
                source_value=-0.5,
                weight=1.0,
            ),
        ])

        apply_self_tags_to_bias(bias, state)

        # Original should be unchanged
        assert bias.valence_bias == original_valence


class TestConfigurableScales:
    """Tests for configurable scale factors."""

    def test_global_scale_zero(self):
        """Global scale 0 produces tags with zero weight."""
        config = SelfReferenceConfig(global_scale=0.0)
        summary = {"dominant_emotion": "joy", "dominant_emotion_value": 0.8}

        tags = generate_self_tags(summary, config)

        for tag in tags:
            assert tag.weight == 0.0

    def test_category_scale(self):
        """Category scale affects specific tag weights."""
        config = SelfReferenceConfig(
            category_scales={"emotion": 2.0, "fear": 0.5},
        )
        summary = {
            "dominant_emotion": "joy",
            "dominant_emotion_value": 0.8,
            "fear_level": 0.6,
        }

        tags = generate_self_tags(summary, config)

        emotion_tags = [t for t in tags if t.category == SelfTagCategory.EMOTION]
        fear_tags = [t for t in tags if t.category == SelfTagCategory.FEAR]

        assert all(t.weight == 2.0 for t in emotion_tags)
        assert all(t.weight == 0.5 for t in fear_tags)


class TestDiagnostics:
    """Tests for diagnostic functions."""

    def test_get_summary(self):
        """Summary provides useful diagnostic info."""
        tags = [
            SelfTag(category=SelfTagCategory.EMOTION, label="test1"),
            SelfTag(category=SelfTagCategory.FEAR, label="test2"),
        ]
        state = SelfReferenceState(tags=tags, reference_count=3)

        summary = state.get_summary()

        assert summary["tag_count"] == 2
        assert "emotion" in summary["categories"]
        assert "fear" in summary["categories"]
        assert summary["reference_count"] == 3

    def test_get_summary_string(self):
        """Human-readable summary is generated."""
        tags = [
            SelfTag(category=SelfTagCategory.EMOTION, label="negative_mood"),
        ]
        state = SelfReferenceState(tags=tags, reference_count=1)

        summary_str = get_self_reference_summary(state)

        assert "negative_mood" in summary_str
        assert "count=1" in summary_str

    def test_empty_summary_string(self):
        """Empty state produces appropriate summary."""
        state = SelfReferenceState(tags=[])

        summary_str = get_self_reference_summary(state)

        assert "no tags" in summary_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
