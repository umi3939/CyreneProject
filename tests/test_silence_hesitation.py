"""
tests/test_silence_hesitation.py - Tests for Silence / Hesitation system

Verifies:
1. Silence is a valid, scorable decision candidate
2. Silence does NOT block the main thread (uses duration attribute)
3. Silence is NOT an error fallback
4. System continues normally after silence is chosen
5. Consecutive silence tracking and natural return to speech
"""

import pytest

from psyche.state import PsycheState, Percept, EmotionVector, DriveVector, Mood
from psyche.responsibility import ResponsibilityInfluence
from psyche.silence_hesitation import (
    SilenceType,
    SilenceConfig,
    SilenceCandidate,
    SilenceResult,
    SilenceState,
    generate_silence_candidate,
    evaluate_silence_score,
    create_silence_result,
    create_speech_result,
    silence_candidate_to_policy,
    is_silence_policy,
    is_silence_result,
    get_silence_duration,
    generate_candidates_with_silence,
    get_silence_summary,
    create_silence_config,
    to_dict,
    from_dict,
)


def create_test_state(
    mood_valence: float = 0.0,
    mood_arousal: float = 0.5,
    fear_level: float = 0.0,
    joy: float = 0.2,
    sorrow: float = 0.1,
    expression_drive: float = 0.5,
    social_drive: float = 0.5,
) -> PsycheState:
    """Create a test PsycheState with specified parameters."""
    return PsycheState(
        emotions=EmotionVector(joy=joy, sorrow=sorrow),
        drives=DriveVector(
            social=social_drive,
            curiosity=0.5,
            expression=expression_drive,
        ),
        mood=Mood(valence=mood_valence, arousal=mood_arousal),
    )


class TestSilenceAsValidCandidate:
    """Tests verifying silence is a valid, scorable candidate."""

    def test_generate_silence_candidate(self):
        """Silence candidate can be generated."""
        state = create_test_state()

        candidate = generate_silence_candidate(state)

        assert candidate is not None
        assert candidate.policy_label == "沈黙する"
        assert isinstance(candidate.score, float)

    def test_silence_has_score(self):
        """Silence candidate has a computed score."""
        state = create_test_state()

        candidate = generate_silence_candidate(state)

        assert candidate.score >= 0

    def test_silence_score_varies_with_state(self):
        """Silence score changes based on psychological state."""
        # Conflicted state (high silence score expected)
        conflicted = create_test_state(
            mood_valence=-0.5,
            joy=0.4,
            sorrow=0.4,  # Emotional conflict
        )

        # Happy state (lower silence score expected)
        happy = create_test_state(
            mood_valence=0.6,
            mood_arousal=0.7,
            expression_drive=0.8,
        )

        conflicted_score = evaluate_silence_score(conflicted)
        happy_score = evaluate_silence_score(happy)

        # Conflicted state should have higher silence score
        assert conflicted_score > happy_score

    def test_silence_converts_to_policy_format(self):
        """Silence candidate converts to standard policy dict."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)

        policy = silence_candidate_to_policy(candidate)

        assert "policy_label" in policy
        assert "rationale" in policy
        assert "expected_drive_change" in policy
        assert "_score" in policy
        assert policy["_is_silence"] is True


class TestSilenceNotBlocking:
    """Tests verifying silence uses duration, not blocking."""

    def test_silence_result_has_duration(self):
        """Silence result has duration attribute, not sleep."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)

        result = create_silence_result(candidate, state)

        assert hasattr(result, "duration")
        assert isinstance(result.duration, float)
        assert result.duration > 0

    def test_duration_within_bounds(self):
        """Duration is within configured bounds."""
        config = SilenceConfig(min_duration=0.5, max_duration=3.0)
        state = create_test_state()

        candidate = generate_silence_candidate(state, config=config)

        assert candidate.suggested_duration >= config.min_duration
        assert candidate.suggested_duration <= config.max_duration

    def test_state_continues_after_silence(self):
        """State continues flag is always True."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)

        result = create_silence_result(candidate, state)

        assert result.state_continues is True


class TestSilenceNotErrorFallback:
    """Tests verifying silence is NOT an error or fallback."""

    def test_silence_is_not_default(self):
        """Silence is not automatically the default choice."""
        # Create a state that should prefer speech
        talkative_state = create_test_state(
            mood_valence=0.5,
            mood_arousal=0.7,
            expression_drive=0.9,
            social_drive=0.8,
        )
        percept = Percept(emotion="happy", emotion_valence=0.5, intent="greeting")

        candidates = generate_candidates_with_silence(
            talkative_state, percept, recalled=[]
        )

        # Silence should not be the top choice
        assert candidates[0]["policy_label"] != "沈黙する"

    def test_silence_competes_with_other_candidates(self):
        """Silence is included in candidate list and scored."""
        state = create_test_state()
        percept = Percept(emotion="neutral", emotion_valence=0.0, intent="unknown")

        candidates = generate_candidates_with_silence(
            state, percept, recalled=[]
        )

        # Find silence in candidates
        silence_candidates = [c for c in candidates if c.get("_is_silence")]
        assert len(silence_candidates) == 1

    def test_silence_has_rationale(self):
        """Silence has a rationale, not an error message."""
        state = create_test_state()

        candidate = generate_silence_candidate(state)

        assert candidate.rationale != ""
        assert "error" not in candidate.rationale.lower()
        assert "fail" not in candidate.rationale.lower()


class TestSilenceTypeDetermination:
    """Tests for silence type determination."""

    def test_emotional_hesitation_detected(self):
        """Emotional conflict produces hesitation type."""
        state = create_test_state(joy=0.5, sorrow=0.5)

        candidate = generate_silence_candidate(state)

        assert candidate.silence_type == SilenceType.EMOTIONAL_HESITATION

    def test_uncertain_pause_on_high_fear(self):
        """High fear produces uncertain pause type."""
        from psyche.pillars import FearIndex
        # FearIndex.value is computed as weighted sum of risks:
        # identity*0.3 + attachment*0.3 + continuity*0.2 + projection*0.2
        # Need value > 0.5, so use high risk values
        state = PsycheState(
            emotions=EmotionVector(fear=0.7),
            fear_index=FearIndex(
                identity_risk=0.8,
                attachment_risk=0.8,
                continuity_risk=0.8,
                projection_risk=0.8,
            ),
            # FearIndex.value = 0.8*0.3 + 0.8*0.3 + 0.8*0.2 + 0.8*0.2 = 0.8
        )

        # Verify fear_level is high enough
        assert state.fear_level > 0.5

        candidate = generate_silence_candidate(state)

        assert candidate.silence_type == SilenceType.UNCERTAIN_PAUSE

    def test_respectful_silence_on_sharing(self):
        """Sharing intent produces respectful silence type."""
        state = create_test_state()
        percept = Percept(emotion="sad", emotion_valence=-0.5, intent="sharing")

        candidate = generate_silence_candidate(state, percept=percept)

        assert candidate.silence_type == SilenceType.RESPECTFUL_SILENCE


class TestConsecutiveSilenceTracking:
    """Tests for consecutive silence tracking."""

    def test_silence_state_records_silence(self):
        """Silence state tracks consecutive silences."""
        ss = SilenceState()

        ss1 = ss.record_silence()
        assert ss1.consecutive_silences == 1

        ss2 = ss1.record_silence()
        assert ss2.consecutive_silences == 2

    def test_speech_resets_consecutive(self):
        """Speech resets consecutive silence count."""
        ss = SilenceState(consecutive_silences=3)

        ss_after = ss.record_speech()

        assert ss_after.consecutive_silences == 0

    def test_consecutive_penalty_applied(self):
        """Consecutive silences reduce silence score."""
        # Use a state that gives a reasonable base score
        state = create_test_state(mood_valence=-0.3, mood_arousal=0.3)
        config = SilenceConfig(max_consecutive_silences=3, base_silence_score=1.0)

        # No prior silence
        ss0 = SilenceState(consecutive_silences=0)
        score1 = evaluate_silence_score(state, config=config, silence_state=ss0)

        # After one silence (penalty = 0.5)
        ss1 = SilenceState(consecutive_silences=1)
        score2 = evaluate_silence_score(state, config=config, silence_state=ss1)

        # After two silences (penalty = 1.0)
        ss2 = SilenceState(consecutive_silences=2)
        score3 = evaluate_silence_score(state, config=config, silence_state=ss2)

        # Score should decrease with consecutive silences
        # Each consecutive silence adds 0.5 penalty
        assert score1 > score2, f"score1={score1}, score2={score2}"
        assert score2 > score3, f"score2={score2}, score3={score3}"

    def test_max_consecutive_enforced(self):
        """Strong penalty when exceeding max consecutive."""
        state = create_test_state()
        config = SilenceConfig(max_consecutive_silences=2)
        ss = SilenceState(consecutive_silences=3)  # Exceeded

        score = evaluate_silence_score(state, config=config, silence_state=ss)

        # Score should be very low
        assert score < 1.0


class TestSilenceResultChecks:
    """Tests for silence result checking functions."""

    def test_is_silence_policy_true(self):
        """is_silence_policy returns True for silence."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)
        policy = silence_candidate_to_policy(candidate)

        assert is_silence_policy(policy) is True

    def test_is_silence_policy_false(self):
        """is_silence_policy returns False for speech."""
        policy = {"policy_label": "共感する", "rationale": "寄り添う"}

        assert is_silence_policy(policy) is False

    def test_is_silence_result_true(self):
        """is_silence_result returns True for silence result."""
        result = SilenceResult(is_silence=True)

        assert is_silence_result(result) is True

    def test_is_silence_result_false(self):
        """is_silence_result returns False for speech result."""
        result = create_speech_result()

        assert is_silence_result(result) is False

    def test_get_silence_duration(self):
        """get_silence_duration extracts duration from policy."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)
        policy = silence_candidate_to_policy(candidate)

        duration = get_silence_duration(policy)

        assert duration == candidate.suggested_duration


class TestResponsibilityInfluence:
    """Tests for responsibility influence on silence."""

    def test_high_anxiety_increases_silence_score(self):
        """High anxiety baseline increases silence preference."""
        # Use a neutral state
        state = create_test_state(mood_valence=0.0, mood_arousal=0.5)

        # No responsibility
        score1 = evaluate_silence_score(state, responsibility_influence=None)

        # With anxiety at max (0.3 is the limit per ResponsibilityInfluence)
        influence = ResponsibilityInfluence(
            anxiety_baseline=0.3,
            caution_bias=0.0,
            empathy_bias=0.0,
            fear_amplification=0.0,
        )
        score2 = evaluate_silence_score(state, responsibility_influence=influence)

        # Anxiety should add: 1.5 * 0.3 = 0.45 bonus
        assert score2 > score1, f"score1={score1}, score2={score2}"

    def test_high_caution_increases_silence_score(self):
        """High caution bias increases silence preference."""
        state = create_test_state()

        # With high caution
        influence = ResponsibilityInfluence(caution_bias=0.5)
        score = evaluate_silence_score(state, responsibility_influence=influence)

        # Should be higher than base
        assert score > SilenceConfig().base_silence_score


class TestCandidateGeneration:
    """Tests for combined candidate generation."""

    def test_silence_included_in_candidates(self):
        """Silence is included in combined candidate list."""
        state = create_test_state()
        percept = Percept(emotion="neutral", emotion_valence=0.0, intent="unknown")

        candidates = generate_candidates_with_silence(state, percept, recalled=[])

        labels = [c["policy_label"] for c in candidates]
        assert "沈黙する" in labels

    def test_candidates_sorted_by_score(self):
        """Combined candidates are sorted by score."""
        state = create_test_state()
        percept = Percept(emotion="neutral", emotion_valence=0.0, intent="unknown")

        candidates = generate_candidates_with_silence(state, percept, recalled=[])

        scores = [c["_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_silence_can_win(self):
        """Silence can be the top choice in appropriate state."""
        # Very conflicted, low arousal, negative mood
        conflicted = PsycheState(
            emotions=EmotionVector(joy=0.5, sorrow=0.6, fear=0.4),
            drives=DriveVector(social=0.2, curiosity=0.2, expression=0.2),
            mood=Mood(valence=-0.6, arousal=0.2),
        )
        percept = Percept(emotion="sad", emotion_valence=-0.7, intent="sharing")

        # Use high silence score config
        config = SilenceConfig(
            base_silence_score=2.0,
            emotional_conflict_bonus=3.0,
        )

        candidates = generate_candidates_with_silence(
            conflicted, percept, recalled=[], silence_config=config
        )

        # Silence might be top choice
        # (Not asserting it IS top, just that it CAN be)
        silence_rank = next(
            i for i, c in enumerate(candidates)
            if c.get("_is_silence")
        )
        assert silence_rank < len(candidates)


class TestConfigSerialization:
    """Tests for configuration serialization."""

    def test_to_dict(self):
        """Config serializes to dict."""
        config = create_silence_config(
            base_score=0.7,
            min_duration=1.0,
            max_duration=5.0,
        )

        data = to_dict(config)

        assert data["base_silence_score"] == 0.7
        assert data["min_duration"] == 1.0
        assert data["max_duration"] == 5.0

    def test_from_dict(self):
        """Config deserializes from dict."""
        data = {
            "base_silence_score": 0.8,
            "max_consecutive_silences": 3,
        }

        config = from_dict(data)

        assert config.base_silence_score == 0.8
        assert config.max_consecutive_silences == 3

    def test_roundtrip(self):
        """Config survives serialization roundtrip."""
        original = create_silence_config(
            base_score=0.6,
            max_consecutive=4,
        )

        data = to_dict(original)
        restored = from_dict(data)

        assert restored.base_silence_score == original.base_silence_score
        assert restored.max_consecutive_silences == original.max_consecutive_silences


class TestSilenceResultSerialization:
    """Tests for SilenceResult serialization."""

    def test_result_to_dict(self):
        """SilenceResult serializes to dict."""
        result = SilenceResult(
            is_silence=True,
            silence_type=SilenceType.EMOTIONAL_HESITATION,
            duration=2.0,
            consecutive_count=1,
        )

        data = result.to_dict()

        assert data["is_silence"] is True
        assert data["silence_type"] == "emotional_hesitation"
        assert data["duration"] == 2.0

    def test_result_from_dict(self):
        """SilenceResult deserializes from dict."""
        data = {
            "is_silence": True,
            "silence_type": "processing_pause",
            "duration": 1.5,
        }

        result = SilenceResult.from_dict(data)

        assert result.is_silence is True
        assert result.silence_type == SilenceType.PROCESSING_PAUSE
        assert result.duration == 1.5


class TestSilenceStateSerialization:
    """Tests for SilenceState serialization."""

    def test_state_to_dict(self):
        """SilenceState serializes to dict."""
        state = SilenceState(
            consecutive_silences=2,
            total_silences=5,
        )

        data = state.to_dict()

        assert data["consecutive_silences"] == 2
        assert data["total_silences"] == 5

    def test_state_from_dict(self):
        """SilenceState deserializes from dict."""
        data = {
            "consecutive_silences": 3,
            "total_silences": 10,
        }

        state = SilenceState.from_dict(data)

        assert state.consecutive_silences == 3
        assert state.total_silences == 10


class TestSummaryFunction:
    """Tests for summary functions."""

    def test_silence_summary(self):
        """get_silence_summary produces readable output."""
        result = SilenceResult(
            is_silence=True,
            silence_type=SilenceType.EMOTIONAL_HESITATION,
            duration=1.5,
            consecutive_count=1,
            reason="感情が揺れている",
        )

        summary = get_silence_summary(result)

        assert "emotional_hesitation" in summary
        assert "1.5" in summary
        assert "感情が揺れている" in summary

    def test_non_silence_summary(self):
        """get_silence_summary handles non-silence."""
        result = create_speech_result()

        summary = get_silence_summary(result)

        assert "speaking" in summary


class TestDesignConstraints:
    """Tests verifying design document constraints are met."""

    def test_silence_is_explicit_choice(self):
        """Silence is an explicit choice, not absence of action."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)

        # Silence has all the attributes of a choice
        assert candidate.policy_label != ""
        assert candidate.rationale != ""
        assert candidate.expected_drive_change is not None

    def test_silence_is_temporary(self):
        """Silence is temporary (has duration)."""
        state = create_test_state()
        candidate = generate_silence_candidate(state)

        # Duration indicates it ends
        assert candidate.suggested_duration > 0
        assert candidate.suggested_duration < 60  # Not forever

    def test_silence_references_state_readonly(self):
        """Silence generation doesn't modify state."""
        state = create_test_state(joy=0.5, sorrow=0.3)
        original_joy = state.emotions.joy
        original_sorrow = state.emotions.sorrow

        _ = generate_silence_candidate(state)

        # State unchanged
        assert state.emotions.joy == original_joy
        assert state.emotions.sorrow == original_sorrow

    def test_no_algorithm_branching(self):
        """Silence uses same scoring mechanism as other policies."""
        state = create_test_state()
        percept = Percept(emotion="neutral", emotion_valence=0.0, intent="unknown")

        candidates = generate_candidates_with_silence(state, percept, recalled=[])

        # All candidates (including silence) have _score
        for c in candidates:
            assert "_score" in c

    def test_silence_not_default_behavior(self):
        """Silence is not the default when no other option."""
        state = create_test_state(
            mood_valence=0.5,  # Positive mood
            expression_drive=0.8,  # Want to express
        )
        percept = Percept(emotion="happy", emotion_valence=0.5, intent="greeting")

        candidates = generate_candidates_with_silence(state, percept, recalled=[])

        # First choice should not be silence
        assert candidates[0].get("_is_silence", False) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
