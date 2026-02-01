"""
tests/test_self_reference.py - Tests for self-reference loop structure

Verifies:
1. State acquisition (read-only)
2. State summarization (coarse compression)
3. Self-tag generation
4. Self-reference execution (circulation)
5. Integration with decision bias
6. No state modification
7. Responsibility distribution summary (from DispersionState)
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
    ResponsibilityDistributionSummary,
    summarize_responsibility_units,
    generate_responsibility_distribution_tags,
)
from psyche.state import PsycheState, EmotionVector, Mood
from psyche.responsibility import ResponsibilityState
from psyche.short_term_memory import ShortTermMemory
from psyche.dynamics import DynamicsState, DynamicsPhase, create_dynamics_state, enter_peak
from psyche.decision_bias import DecisionBias, create_neutral_bias
from psyche.responsibility_dispersion import (
    DispersionState,
    create_responsibility_unit,
    disperse_responsibility,
    DispersionPlan,
    create_dispersion_state,
)


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


class TestResponsibilityDistributionSummary:
    """Tests for ResponsibilityDistributionSummary structure."""

    def test_create_empty_summary(self):
        """Empty summary has default values."""
        summary = ResponsibilityDistributionSummary()

        assert summary.total_weight == 0.0
        assert summary.unit_count == 0
        assert summary.average_distance == 0.0
        assert summary.dominant_meaning == ""

    def test_summary_serialization(self):
        """Summary survives serialization roundtrip."""
        summary = ResponsibilityDistributionSummary(
            total_weight=0.5,
            unit_count=3,
            average_distance=1.5,
            near_weight_ratio=0.3,
            far_weight_ratio=0.7,
            dominant_meaning="caused_harm",
            meaning_count=2,
        )

        data = summary.to_dict()
        restored = ResponsibilityDistributionSummary.from_dict(data)

        assert restored.total_weight == summary.total_weight
        assert restored.unit_count == summary.unit_count
        assert restored.average_distance == summary.average_distance
        assert restored.dominant_meaning == summary.dominant_meaning


class TestSummarizeResponsibilityUnits:
    """Tests for summarize_responsibility_units function."""

    def test_summarize_empty_state(self):
        """Empty dispersion state produces empty summary."""
        state = create_dispersion_state()

        summary = summarize_responsibility_units(state)

        assert summary.total_weight == 0.0
        assert summary.unit_count == 0

    def test_summarize_none_state(self):
        """None dispersion state produces empty summary."""
        summary = summarize_responsibility_units(None)

        assert summary.total_weight == 0.0
        assert summary.unit_count == 0

    def test_summarize_single_unit(self):
        """Single unit produces correct summary."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            meaning="caused_harm",
            distance=0.5,
            time_slice="immediate",
        )

        summary = summarize_responsibility_units(state)

        assert summary.total_weight == 0.5
        assert summary.unit_count == 1
        assert summary.average_distance == 0.5
        assert summary.dominant_meaning == "caused_harm"
        assert summary.time_concentrated is True

    def test_summarize_multiple_units(self):
        """Multiple units produce correct summary."""
        unit1, state = create_responsibility_unit(
            weight=0.3,
            origin="a",
            meaning="meaning_a",
            distance=0.5,
        )
        unit2, state = create_responsibility_unit(
            weight=0.7,
            origin="b",
            meaning="meaning_b",
            distance=2.0,
            state=state,
        )

        summary = summarize_responsibility_units(state)

        assert summary.total_weight == pytest.approx(1.0)
        assert summary.unit_count == 2
        # Weighted average distance: (0.3*0.5 + 0.7*2.0) / 1.0 = 1.55
        assert summary.average_distance == pytest.approx(1.55)
        # Dominant meaning is meaning_b (weight 0.7 > 0.3)
        assert summary.dominant_meaning == "meaning_b"
        assert summary.meaning_count == 2

    def test_summarize_distance_distribution(self):
        """Distance distribution is correctly calculated."""
        # Near unit (distance < 1.0)
        unit1, state = create_responsibility_unit(
            weight=0.4,
            origin="a",
            distance=0.5,
        )
        # Far unit (distance >= 1.0)
        unit2, state = create_responsibility_unit(
            weight=0.6,
            origin="b",
            distance=2.0,
            state=state,
        )

        summary = summarize_responsibility_units(state)

        assert summary.near_weight_ratio == pytest.approx(0.4)
        assert summary.far_weight_ratio == pytest.approx(0.6)

    def test_summarize_time_distribution(self):
        """Time distribution is correctly calculated."""
        unit1, state = create_responsibility_unit(
            weight=0.3,
            origin="a",
            time_slice="immediate",
        )
        unit2, state = create_responsibility_unit(
            weight=0.4,
            origin="b",
            time_slice="near_future",
            state=state,
        )
        unit3, state = create_responsibility_unit(
            weight=0.3,
            origin="c",
            time_slice="distant_future",
            state=state,
        )

        summary = summarize_responsibility_units(state)

        assert summary.time_slice_count == 3
        assert summary.time_concentrated is False
        # Dominant is near_future (0.4)
        assert summary.dominant_time_slice == "near_future"

    def test_summarize_does_not_modify_state(self):
        """Summarization is read-only."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")
        original_unit_count = len(state.units)

        summarize_responsibility_units(state)

        assert len(state.units) == original_unit_count


class TestGenerateResponsibilityDistributionTags:
    """Tests for generate_responsibility_distribution_tags function."""

    def test_generate_from_empty_summary(self):
        """Empty summary produces no tags."""
        summary = ResponsibilityDistributionSummary()

        tags = generate_responsibility_distribution_tags(summary)

        assert len(tags) == 0

    def test_generate_weight_present_tag(self):
        """Weight > 0 generates weight_present tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=0.5,
            unit_count=1,
        )

        tags = generate_responsibility_distribution_tags(summary)

        weight_tags = [t for t in tags if t.label == "responsibility_weight_present"]
        assert len(weight_tags) == 1
        assert weight_tags[0].source_value == 0.5

    def test_generate_near_dominant_tag(self):
        """Near dominant generates near_dominant tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=1.0,
            unit_count=2,
            near_weight_ratio=0.7,
            far_weight_ratio=0.3,
        )

        tags = generate_responsibility_distribution_tags(summary)

        near_tags = [t for t in tags if t.label == "responsibility_near_dominant"]
        assert len(near_tags) == 1

    def test_generate_far_dominant_tag(self):
        """Far dominant generates far_dominant tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=1.0,
            unit_count=2,
            near_weight_ratio=0.3,
            far_weight_ratio=0.7,
        )

        tags = generate_responsibility_distribution_tags(summary)

        far_tags = [t for t in tags if t.label == "responsibility_far_dominant"]
        assert len(far_tags) == 1

    def test_generate_time_concentrated_tag(self):
        """Concentrated time generates time_concentrated tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=0.5,
            unit_count=1,
            time_slice_count=1,
            time_concentrated=True,
        )

        tags = generate_responsibility_distribution_tags(summary)

        time_tags = [t for t in tags if t.label == "responsibility_time_concentrated"]
        assert len(time_tags) == 1

    def test_generate_time_dispersed_tag(self):
        """Dispersed time generates time_dispersed tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=0.5,
            unit_count=3,
            time_slice_count=3,
            time_concentrated=False,
        )

        tags = generate_responsibility_distribution_tags(summary)

        time_tags = [t for t in tags if t.label == "responsibility_time_dispersed"]
        assert len(time_tags) == 1

    def test_generate_meaning_tag(self):
        """Dominant meaning generates meaning tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=0.5,
            unit_count=1,
            dominant_meaning="caused_harm",
            meaning_weights={"caused_harm": 0.5},
        )

        tags = generate_responsibility_distribution_tags(summary)

        meaning_tags = [t for t in tags if "meaning_caused_harm" in t.label]
        assert len(meaning_tags) == 1

    def test_generate_meaning_diverse_tag(self):
        """Multiple meanings generate meaning_diverse tag."""
        summary = ResponsibilityDistributionSummary(
            total_weight=1.0,
            unit_count=2,
            meaning_count=3,
        )

        tags = generate_responsibility_distribution_tags(summary)

        diverse_tags = [t for t in tags if t.label == "responsibility_meaning_diverse"]
        assert len(diverse_tags) == 1


class TestAcquireWithDispersionState:
    """Tests for acquire_self_reference_targets with dispersion state."""

    def test_acquire_with_dispersion_state(self):
        """Dispersion state is acquired correctly."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            meaning="test_meaning",
        )

        targets = acquire_self_reference_targets(dispersion_state=state)

        assert "responsibility_distribution" in targets
        assert targets["responsibility_distribution"]["total_weight"] == 0.5
        assert targets["responsibility_distribution"]["unit_count"] == 1

    def test_acquire_without_dispersion_state(self):
        """Missing dispersion state is handled."""
        targets = acquire_self_reference_targets()

        assert "responsibility_distribution" not in targets


class TestExecuteWithDispersionState:
    """Tests for execute_self_reference with dispersion state."""

    def test_execute_includes_dispersion_tags(self):
        """Execution with dispersion state generates distribution tags."""
        unit, state = create_responsibility_unit(
            weight=0.5,
            origin="test",
            meaning="test_meaning",
            distance=0.5,
        )

        result = execute_self_reference(dispersion_state=state)

        # Should have responsibility distribution tags
        dist_tags = [
            t for t in result.tags
            if t.category == SelfTagCategory.RESPONSIBILITY_DISTRIBUTION
        ]
        assert len(dist_tags) > 0

    def test_execute_does_not_modify_dispersion_state(self):
        """Execution does not modify dispersion state."""
        unit, state = create_responsibility_unit(weight=0.5, origin="test")
        original_unit_count = len(state.units)

        execute_self_reference(dispersion_state=state)

        assert len(state.units) == original_unit_count


class TestApplyResponsibilityDistributionTags:
    """Tests for apply_self_tags_to_bias with distribution tags."""

    def test_apply_near_dominant_tag(self):
        """Near dominant tag adjusts intensity."""
        bias = create_neutral_bias()
        state = SelfReferenceState(tags=[
            SelfTag(
                category=SelfTagCategory.RESPONSIBILITY_DISTRIBUTION,
                label="responsibility_near_dominant",
                source_value=0.7,
                weight=1.0,
            ),
        ])

        result = apply_self_tags_to_bias(bias, state)

        # Near dominant should increase intensity
        assert result.residue_intensity > bias.residue_intensity

    def test_apply_weight_present_tag(self):
        """Weight present tag adjusts intensity."""
        bias = create_neutral_bias()
        state = SelfReferenceState(tags=[
            SelfTag(
                category=SelfTagCategory.RESPONSIBILITY_DISTRIBUTION,
                label="responsibility_weight_present",
                source_value=0.8,
                weight=1.0,
            ),
        ])

        result = apply_self_tags_to_bias(bias, state)

        # Weight present should slightly increase intensity
        assert result.residue_intensity > bias.residue_intensity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
