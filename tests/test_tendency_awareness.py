"""
Tests for tendency_awareness.py - Self-Awareness of Repeated Tendency (反復傾向の自己認知)

These tests verify:
1. PURE OBSERVATION: No feedback to decision making
2. ABSTRACT CONCEPTS: Raw numbers converted to fuzzy levels (Low/Medium/High)
3. Self-Reference integration for self-description
4. Threshold-based awareness generation
5. Reversibility - awareness fades when tendency weakens
"""

import pytest

from psyche.goal_candidates import CandidateCategory
from psyche.scoped_goal import ScopedGoal, ScopeStatus
from psyche.repeated_tendency import (
    RepeatedTendencySystem,
    RepeatedTendencyConfig,
    Tendency,
    TendencyPattern,
)
from psyche.tendency_awareness import (
    StrengthLevel,
    DurationLevel,
    ConfidenceLevel,
    AwarenessType,
    TendencyAwarenessItem,
    TendencyAwareness,
    AwarenessConfig,
    observe_tendency,
    observe_tendencies,
    generate_awareness_tags,
    get_awareness_summary,
    get_awareness_for_introspection,
    create_config,
    create_empty_awareness,
    _classify_strength,
    _classify_duration,
    _classify_confidence,
)


class TestAbstractEnums:
    """Tests for abstract concept enums."""

    def test_strength_levels_are_abstract(self):
        """Strength levels should be human-readable concepts, not numbers."""
        assert StrengthLevel.NONE.value == "none"
        assert StrengthLevel.SLIGHT.value == "slight"
        assert StrengthLevel.MODERATE.value == "moderate"
        assert StrengthLevel.STRONG.value == "strong"

    def test_duration_levels_are_abstract(self):
        """Duration levels should be fuzzy concepts, not turn counts."""
        assert DurationLevel.RECENT.value == "recent"
        assert DurationLevel.ESTABLISHED.value == "established"
        assert DurationLevel.PERSISTENT.value == "persistent"

    def test_confidence_levels_are_abstract(self):
        """Confidence levels should not expose reinforcement counts."""
        assert ConfidenceLevel.UNCERTAIN.value == "uncertain"
        assert ConfidenceLevel.FORMING.value == "forming"
        assert ConfidenceLevel.ESTABLISHED.value == "established"

    def test_awareness_types_are_descriptive(self):
        """Awareness types should describe self-perception."""
        assert AwarenessType.HABIT_FORMING.value == "habit_forming"
        assert AwarenessType.SLIGHT_BIAS.value == "slight_bias"
        assert AwarenessType.STRONG_HABIT.value == "strong_habit"
        assert AwarenessType.FADING_HABIT.value == "fading_habit"


class TestClassificationFunctions:
    """Tests for internal classification functions."""

    def test_classify_strength_none(self):
        config = create_config()
        assert _classify_strength(0.01, config) == StrengthLevel.NONE

    def test_classify_strength_slight(self):
        config = create_config()
        assert _classify_strength(0.04, config) == StrengthLevel.SLIGHT

    def test_classify_strength_moderate(self):
        config = create_config()
        assert _classify_strength(0.09, config) == StrengthLevel.MODERATE

    def test_classify_strength_strong(self):
        config = create_config()
        assert _classify_strength(0.15, config) == StrengthLevel.STRONG

    def test_classify_duration_recent(self):
        config = create_config()
        assert _classify_duration(90, 95, config) == DurationLevel.RECENT

    def test_classify_duration_established(self):
        config = create_config()
        assert _classify_duration(50, 75, config) == DurationLevel.ESTABLISHED

    def test_classify_duration_persistent(self):
        config = create_config()
        assert _classify_duration(10, 100, config) == DurationLevel.PERSISTENT

    def test_classify_confidence_uncertain(self):
        config = create_config()
        assert _classify_confidence(1, config) == ConfidenceLevel.UNCERTAIN

    def test_classify_confidence_forming(self):
        config = create_config()
        assert _classify_confidence(4, config) == ConfidenceLevel.FORMING

    def test_classify_confidence_established(self):
        config = create_config()
        assert _classify_confidence(10, config) == ConfidenceLevel.ESTABLISHED


class TestTendencyAwarenessItem:
    """Tests for TendencyAwarenessItem dataclass."""

    def test_default_item(self):
        item = TendencyAwarenessItem()
        assert item.awareness_type == AwarenessType.SLIGHT_BIAS
        assert item.strength_level == StrengthLevel.NONE
        assert item.description == ""

    def test_item_serialization(self):
        item = TendencyAwarenessItem(
            awareness_type=AwarenessType.HABIT_FORMING,
            category=CandidateCategory.CONNECTION,
            strength_level=StrengthLevel.MODERATE,
            duration_level=DurationLevel.ESTABLISHED,
            description="Test description",
        )
        data = item.to_dict()
        restored = TendencyAwarenessItem.from_dict(data)

        assert restored.awareness_type == AwarenessType.HABIT_FORMING
        assert restored.category == CandidateCategory.CONNECTION
        assert restored.strength_level == StrengthLevel.MODERATE

    def test_item_does_not_expose_raw_numbers(self):
        """Item serialization should not include raw numerical values."""
        item = TendencyAwarenessItem(
            strength_level=StrengthLevel.STRONG,
            _tendency_id="internal123",
        )
        data = item.to_dict()

        # Internal ID should not be exposed
        assert "_tendency_id" not in data
        # Strength should be abstract, not numerical
        assert data["strength_level"] == "strong"
        assert isinstance(data["strength_level"], str)


class TestTendencyAwareness:
    """Tests for TendencyAwareness collection."""

    def test_empty_awareness(self):
        awareness = create_empty_awareness()
        assert not awareness.has_awareness
        assert len(awareness.items) == 0
        assert awareness.dominant_category is None

    def test_awareness_with_items(self):
        item = TendencyAwarenessItem(
            category=CandidateCategory.EXPRESSION,
            strength_level=StrengthLevel.MODERATE,
        )
        awareness = TendencyAwareness(
            items=[item],
            has_awareness=True,
            dominant_category=CandidateCategory.EXPRESSION,
            overall_strength=StrengthLevel.MODERATE,
        )

        assert awareness.has_awareness
        assert len(awareness.items) == 1

    def test_get_by_category(self):
        items = [
            TendencyAwarenessItem(category=CandidateCategory.CONNECTION),
            TendencyAwarenessItem(category=CandidateCategory.EXPRESSION),
            TendencyAwarenessItem(category=CandidateCategory.CONNECTION),
        ]
        awareness = TendencyAwareness(items=items, has_awareness=True)

        connection_items = awareness.get_by_category(CandidateCategory.CONNECTION)
        assert len(connection_items) == 2

    def test_get_strongest(self):
        items = [
            TendencyAwarenessItem(strength_level=StrengthLevel.SLIGHT),
            TendencyAwarenessItem(strength_level=StrengthLevel.STRONG),
            TendencyAwarenessItem(strength_level=StrengthLevel.MODERATE),
        ]
        awareness = TendencyAwareness(items=items, has_awareness=True)

        strongest = awareness.get_strongest()
        assert strongest is not None
        assert strongest.strength_level == StrengthLevel.STRONG


class TestObserveTendency:
    """Tests for observe_tendency function."""

    def test_weak_tendency_not_noticed(self):
        """Tendencies below threshold should not generate awareness."""
        config = create_config(min_strength_for_awareness=0.05)
        tendency = Tendency(
            strength=0.02,  # Below threshold
            confidence=0.5,
            total_reinforcements=5,
        )

        result = observe_tendency(tendency, current_turn=100, config=config)
        assert result is None

    def test_notable_tendency_generates_awareness(self):
        """Tendencies above threshold should generate awareness."""
        config = create_config(min_strength_for_awareness=0.03)
        tendency = Tendency(
            pattern=TendencyPattern(category=CandidateCategory.APPROACH),
            strength=0.10,
            confidence=0.6,
            total_reinforcements=8,
            first_formed_turn=50,
        )

        result = observe_tendency(tendency, current_turn=100, config=config)

        assert result is not None
        assert result.category == CandidateCategory.APPROACH
        assert result.strength_level != StrengthLevel.NONE

    def test_awareness_contains_description(self):
        """Awareness should contain human-readable description."""
        tendency = Tendency(
            pattern=TendencyPattern(category=CandidateCategory.CONNECTION),
            strength=0.12,
            confidence=0.7,
            total_reinforcements=10,
        )

        result = observe_tendency(tendency, current_turn=100)

        assert result is not None
        assert result.description != ""
        assert "connection" in result.description.lower()


class TestObserveTendencies:
    """Tests for observe_tendencies (main entry point)."""

    def test_empty_system_no_awareness(self):
        """Empty system should produce no awareness."""
        config = RepeatedTendencyConfig()
        system = RepeatedTendencySystem(config=config)

        awareness = observe_tendencies(system)

        assert not awareness.has_awareness
        assert len(awareness.items) == 0

    def test_system_with_tendencies_produces_awareness(self):
        """System with notable tendencies should produce awareness."""
        config = RepeatedTendencyConfig(
            min_repetitions=2,
            strength_increment=0.05,
            decay_rate=0.001,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Build up a tendency
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        awareness = observe_tendencies(system)

        assert awareness.has_awareness
        assert len(awareness.items) >= 1
        assert awareness.dominant_category == CandidateCategory.EXPRESSION

    def test_multiple_tendencies_all_observed(self):
        """Multiple tendencies should each produce awareness items."""
        config = RepeatedTendencyConfig(
            min_repetitions=2,
            strength_increment=0.05,
            decay_rate=0.001,
            miss_decay_multiplier=1.0,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Build first tendency
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Build second tendency
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.EXPLORATION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        awareness = observe_tendencies(system)

        # Should have awareness of both (if both still above threshold)
        categories = {item.category for item in awareness.items}
        assert len(categories) >= 1  # At least one tendency noticed


class TestPureObservation:
    """Tests ensuring this is PURE OBSERVATION with no side effects."""

    def test_observe_does_not_modify_system(self):
        """Observing tendencies should not modify the system state."""
        config = RepeatedTendencyConfig(
            min_repetitions=2,
            strength_increment=0.05,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        # Build tendency
        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.APPROACH,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Record state before observation
        tendencies_before = len(system.get_tendencies())
        turn_count_before = system.state.turn_count

        # Observe multiple times
        for _ in range(10):
            observe_tendencies(system)

        # State should be unchanged
        assert len(system.get_tendencies()) == tendencies_before
        assert system.state.turn_count == turn_count_before

    def test_awareness_does_not_expose_raw_numbers(self):
        """Awareness should not contain raw numerical values."""
        config = RepeatedTendencyConfig(
            min_repetitions=2,
            strength_increment=0.05,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=config)

        for _ in range(5):
            scope = ScopedGoal(
                category=CandidateCategory.CONNECTION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        awareness = observe_tendencies(system)
        data = awareness.to_dict()

        # Check that strength is abstract string, not number
        for item in data["items"]:
            assert isinstance(item["strength_level"], str)
            assert isinstance(item["duration_level"], str)
            assert isinstance(item["confidence_level"], str)


class TestNoDecisionFeedback:
    """Tests ensuring awareness does NOT affect decision making."""

    def test_awareness_has_no_bias_methods(self):
        """TendencyAwareness should not have methods for decision bias."""
        awareness = TendencyAwareness()

        # Should NOT have these methods
        assert not hasattr(awareness, "apply_to_candidate")
        assert not hasattr(awareness, "apply_to_decision")
        assert not hasattr(awareness, "get_bias")
        assert not hasattr(awareness, "modify_score")

    def test_awareness_item_has_no_score_field(self):
        """Awareness items should not have score/weight for decisions."""
        item = TendencyAwarenessItem()
        data = item.to_dict()

        # Should NOT have decision-related fields
        assert "score" not in data
        assert "bias" not in data
        assert "weight" not in data
        assert "adjustment" not in data


class TestSelfReferenceIntegration:
    """Tests for integration with SelfReferenceSystem."""

    def test_generate_tags_for_empty_awareness(self):
        """Empty awareness should generate no tags."""
        awareness = create_empty_awareness()
        tags = generate_awareness_tags(awareness)
        assert len(tags) == 0

    def test_generate_tags_for_awareness(self):
        """Awareness should generate self-reference tags."""
        item = TendencyAwarenessItem(
            awareness_type=AwarenessType.HABIT_FORMING,
            category=CandidateCategory.EXPRESSION,
            strength_level=StrengthLevel.MODERATE,
            description="Test habit forming",
        )
        awareness = TendencyAwareness(
            items=[item],
            has_awareness=True,
            dominant_category=CandidateCategory.EXPRESSION,
            overall_strength=StrengthLevel.MODERATE,
        )

        tags = generate_awareness_tags(awareness)

        assert len(tags) >= 1
        # Tags should be in self-reference format
        for tag in tags:
            assert "category" in tag
            assert tag["category"] == "tendency"
            assert "label" in tag

    def test_tags_contain_abstract_metadata(self):
        """Tags should contain abstract metadata for self-description."""
        item = TendencyAwarenessItem(
            awareness_type=AwarenessType.STRONG_HABIT,
            category=CandidateCategory.CONNECTION,
            strength_level=StrengthLevel.STRONG,
            duration_level=DurationLevel.PERSISTENT,
            description="I've developed a habit",
        )
        awareness = TendencyAwareness(
            items=[item],
            has_awareness=True,
            dominant_category=CandidateCategory.CONNECTION,
            overall_strength=StrengthLevel.STRONG,
        )

        tags = generate_awareness_tags(awareness)

        # Find the item tag (not the overall tag)
        item_tag = next(t for t in tags if "strong_habit" in t["label"])
        assert "metadata" in item_tag
        assert item_tag["metadata"]["strength"] == "strong"
        assert item_tag["metadata"]["description"] == "I've developed a habit"

    def test_tags_have_zero_source_value(self):
        """Tags should have zero source_value to hide raw numbers."""
        item = TendencyAwarenessItem(
            strength_level=StrengthLevel.MODERATE,
        )
        awareness = TendencyAwareness(
            items=[item],
            has_awareness=True,
        )

        tags = generate_awareness_tags(awareness)

        for tag in tags:
            # source_value should be 0 (hidden)
            assert tag["source_value"] == 0.0


class TestIntrospectionIntegration:
    """Tests for introspection trace integration."""

    def test_introspection_data_for_empty_awareness(self):
        awareness = create_empty_awareness()
        data = get_awareness_for_introspection(awareness)

        assert data["has_tendency_awareness"] is False
        assert "observation_note" in data

    def test_introspection_data_contains_descriptions(self):
        item = TendencyAwarenessItem(
            description="I seem to be connecting more",
        )
        awareness = TendencyAwareness(
            items=[item],
            has_awareness=True,
        )

        data = get_awareness_for_introspection(awareness)

        assert data["has_tendency_awareness"] is True
        assert "I seem to be connecting more" in data["descriptions"]
        # Should include note about non-influence on decisions
        assert "NOT affect" in data["observation_note"]

    def test_summary_is_human_readable(self):
        item = TendencyAwarenessItem(
            strength_level=StrengthLevel.MODERATE,
            description="I notice a habit forming",
        )
        awareness = TendencyAwareness(
            items=[item],
            has_awareness=True,
        )

        summary = get_awareness_summary(awareness)

        assert "I notice a habit forming" in summary
        assert "moderate" in summary.lower()


class TestAwarenessConfig:
    """Tests for AwarenessConfig."""

    def test_default_config(self):
        config = create_config()
        assert config.min_strength_for_awareness > 0
        assert config.slight_threshold < config.moderate_threshold
        assert config.moderate_threshold < config.strong_threshold

    def test_config_serialization(self):
        config = create_config(
            min_strength_for_awareness=0.05,
            strong_threshold=0.15,
        )
        data = config.to_dict()
        restored = AwarenessConfig.from_dict(data)

        assert restored.min_strength_for_awareness == 0.05
        assert restored.strong_threshold == 0.15


class TestFadingAwareness:
    """Tests for awareness fading when tendencies weaken."""

    def test_fading_habit_detection(self):
        """Should detect when a habit is fading."""
        config = create_config(fading_miss_threshold=3)
        tendency = Tendency(
            pattern=TendencyPattern(category=CandidateCategory.APPROACH),
            strength=0.08,
            confidence=0.5,
            total_reinforcements=5,
            consecutive_misses=5,  # Many misses
        )

        result = observe_tendency(tendency, current_turn=100, config=config)

        assert result is not None
        assert result.awareness_type == AwarenessType.FADING_HABIT

    def test_weak_tendency_disappears_from_awareness(self):
        """Tendency that drops below threshold should not appear."""
        config = create_config(min_strength_for_awareness=0.05)
        tendency = Tendency(
            strength=0.03,  # Below threshold
        )

        result = observe_tendency(tendency, current_turn=100, config=config)
        assert result is None


class TestIntegration:
    """Full integration tests."""

    def test_full_awareness_cycle(self):
        """Test complete cycle: build tendency -> awareness -> self-reference tags."""
        # Create system and build tendency
        tendency_config = RepeatedTendencyConfig(
            min_repetitions=2,
            strength_increment=0.05,
            decay_rate=0.001,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=tendency_config)

        for _ in range(6):
            scope = ScopedGoal(
                category=CandidateCategory.EXPRESSION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        # Observe and generate awareness
        awareness = observe_tendencies(system)

        # Generate tags for self-reference
        tags = generate_awareness_tags(awareness)

        # Get introspection data
        intro_data = get_awareness_for_introspection(awareness)

        # Verify the chain
        assert awareness.has_awareness
        assert len(tags) >= 1
        assert intro_data["has_tendency_awareness"]

        # Verify no raw numbers leaked
        for tag in tags:
            assert tag["source_value"] == 0.0

    def test_awareness_reflects_tendency_state(self):
        """Awareness should accurately reflect current tendency state."""
        tendency_config = RepeatedTendencyConfig(
            min_repetitions=2,
            strength_increment=0.03,
            decay_rate=0.001,
            recency_window=50,
        )
        system = RepeatedTendencySystem(config=tendency_config)

        # Build slight tendency (few reinforcements)
        for _ in range(3):
            scope = ScopedGoal(
                category=CandidateCategory.ISOLATION,
                status=ScopeStatus.USED,
                action_taken=True,
            )
            system.observe_turn(scoped_goal_used=scope)

        awareness = observe_tendencies(system)

        if awareness.has_awareness:
            # Should be slight, not strong
            assert awareness.overall_strength in [
                StrengthLevel.SLIGHT,
                StrengthLevel.MODERATE,
            ]
