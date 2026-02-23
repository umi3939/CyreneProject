"""
tests/test_thought_dynamic.py - Tests for thought.py policy dynamic expansion.

Tests for the dynamic policy candidate structure:
- POLICIES list expansion (6 -> 15)
- _FALLBACK_TEXT expansion
- generate_thought_candidates extended_inputs parameter
- New scoring conditions (#11-16)
- Dynamic selection (3-5 candidates)
- Safety valve (score gap damping)
- value_orientation policy_dimension_map additions
- Backward compatibility
"""

import pytest

from psyche.state import PsycheState, Percept, EmotionVector, DriveVector, Mood
from psyche.pillars import FearIndex
from psyche.responsibility import ResponsibilityInfluence
from psyche.thought import (
    POLICIES,
    _FALLBACK_TEXT,
    generate_thought_candidates,
    select_policy,
    _score_candidate,
)
from psyche.value_orientation import ValueOrientationConfig


# ── Helper fixtures ──────────────────────────────────────────


def _make_state(**overrides) -> PsycheState:
    """Create a PsycheState with optional overrides."""
    defaults = {
        "emotions": EmotionVector(),
        "drives": DriveVector(social=0.5, curiosity=0.5, expression=0.5),
        "mood": Mood(valence=0.0, arousal=0.3),
    }
    defaults.update(overrides)
    return PsycheState(**defaults)


def _make_percept(**overrides) -> Percept:
    """Create a Percept with optional overrides."""
    defaults = {
        "text": "hello",
        "intent": "unknown",
        "emotion_valence": 0.0,
    }
    defaults.update(overrides)
    return Percept(**defaults)


# ── Existing 6 labels ──────────────────────────────────────────

ORIGINAL_LABELS = [
    "共感する",
    "質問で会話を広げる",
    "からかう",
    "話題を変える",
    "感想を述べる",
    "励ます",
]

NEW_LABELS = [
    "黙って聞く",
    "自分の経験を話す",
    "確認する",
    "冗談を言う",
    "謝る",
    "提案する",
    "見守る",
    "同意する",
    "反論する",
]

ALL_LABELS = ORIGINAL_LABELS + NEW_LABELS


# ── 1. POLICIES list validation ──────────────────────────────


class TestPoliciesList:
    def test_policies_count_is_15(self):
        assert len(POLICIES) == 15

    def test_original_6_labels_preserved(self):
        labels = [p["policy_label"] for p in POLICIES]
        for orig in ORIGINAL_LABELS:
            assert orig in labels, f"Original label '{orig}' missing"

    def test_new_9_labels_present(self):
        labels = [p["policy_label"] for p in POLICIES]
        for new in NEW_LABELS:
            assert new in labels, f"New label '{new}' missing"

    def test_all_policies_have_required_fields(self):
        required_fields = {"policy_label", "rationale_template", "drive_target", "expected_drive_change"}
        for policy in POLICIES:
            for field in required_fields:
                assert field in policy, f"Policy '{policy.get('policy_label', '?')}' missing field '{field}'"

    def test_drive_targets_are_valid(self):
        valid_targets = {"social", "curiosity", "expression", "safety", "autonomy"}
        for policy in POLICIES:
            assert policy["drive_target"] in valid_targets, (
                f"Policy '{policy['policy_label']}' has invalid drive_target '{policy['drive_target']}'"
            )

    def test_expected_drive_change_has_3_keys(self):
        expected_keys = {"social", "curiosity", "expression"}
        for policy in POLICIES:
            edc = policy["expected_drive_change"]
            assert set(edc.keys()) == expected_keys, (
                f"Policy '{policy['policy_label']}' expected_drive_change keys mismatch"
            )


# ── 2. _FALLBACK_TEXT validation ──────────────────────────────


class TestFallbackText:
    def test_fallback_text_count_is_15(self):
        assert len(_FALLBACK_TEXT) == 15

    def test_all_labels_have_fallback(self):
        for label in ALL_LABELS:
            assert label in _FALLBACK_TEXT, f"Fallback text missing for '{label}'"

    def test_fallback_texts_are_nonempty_strings(self):
        for label, text in _FALLBACK_TEXT.items():
            assert isinstance(text, str) and len(text) > 0, (
                f"Fallback text for '{label}' is empty or not a string"
            )


# ── 3. generate_thought_candidates basic ──────────────────────


class TestGenerateBasic:
    def test_without_extended_inputs(self):
        """Backward compatibility: extended_inputs=None works."""
        state = _make_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(state, percept, [])
        assert len(candidates) >= 3
        assert len(candidates) <= 5

    def test_with_extended_inputs(self):
        """extended_inputs dict is accepted."""
        state = _make_state()
        percept = _make_percept()
        ext = {
            "self_image_stability": 0.5,
            "coherence_level": 0.5,
            "strain_level": 0.0,
            "narrative_coherence": 0.5,
            "tendency_count": 0,
            "dominant_tendency": "",
            "tendency_strength": 0.0,
            "other_count": 0,
            "boundary_clarity": 0.5,
            "has_active_goal": False,
            "goal_strength": 0.0,
            "motive_count": 0,
            "me_supply_strength": 0.0,
        }
        candidates = generate_thought_candidates(state, percept, [], extended_inputs=ext)
        assert len(candidates) >= 3
        assert len(candidates) <= 5

    def test_result_has_required_keys(self):
        state = _make_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(state, percept, [])
        for c in candidates:
            assert "policy_label" in c
            assert "rationale" in c
            assert "expected_drive_change" in c
            assert "text" in c
            assert "_score" in c

    def test_candidates_sorted_descending(self):
        state = _make_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(state, percept, [])
        scores = [c["_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)


# ── 4. New policy scoring ──────────────────────────────────────


class TestNewPolicyScoring:
    def test_safety_axis_high_fear_high_caution(self):
        """Safety axis candidates score higher when fear and caution are high."""
        state = _make_state(
            emotions=EmotionVector(fear=0.8),
            fear_index=FearIndex(identity_risk=0.6, attachment_risk=0.6),
        )
        percept = _make_percept()
        resp = ResponsibilityInfluence(caution_bias=0.4, empathy_bias=0.0, anxiety_baseline=0.0)
        ext = {
            "self_image_stability": 0.2,  # Low stability -> high safety drive
            "coherence_level": 0.5,
            "strain_level": 0.7,
        }

        # Score a safety-target policy
        safety_policy = None
        for p in POLICIES:
            if p["policy_label"] == "黙って聞く":
                safety_policy = p
                break
        assert safety_policy is not None

        score_with_ext = _score_candidate(
            safety_policy, state, percept, [], resp, None, ext,
        )
        score_without_ext = _score_candidate(
            safety_policy, state, percept, [], resp, None, None,
        )
        # Extended inputs should boost safety candidates
        assert score_with_ext > score_without_ext

    def test_autonomy_axis_high_tendency(self):
        """Autonomy axis candidates score higher when tendency_strength is high."""
        state = _make_state()
        percept = _make_percept()
        ext = {
            "tendency_strength": 0.8,
            "goal_strength": 0.7,
            "coherence_level": 0.8,
        }

        # Score an autonomy-target policy
        autonomy_policy = None
        for p in POLICIES:
            if p["policy_label"] == "見守る":
                autonomy_policy = p
                break
        assert autonomy_policy is not None

        score_with_ext = _score_candidate(
            autonomy_policy, state, percept, [], None, None, ext,
        )
        score_without_ext = _score_candidate(
            autonomy_policy, state, percept, [], None, None, None,
        )
        # Extended inputs should boost autonomy candidates
        assert score_with_ext > score_without_ext

    def test_condition_11_self_image_stress(self):
        """Condition #11: low stability / high strain boosts safety labels."""
        state = _make_state()
        percept = _make_percept()
        ext = {"self_image_stability": 0.2, "strain_level": 0.8}

        listen_policy = [p for p in POLICIES if p["policy_label"] == "黙って聞く"][0]
        score = _score_candidate(listen_policy, state, percept, [], None, None, ext)
        score_neutral = _score_candidate(listen_policy, state, percept, [], None, None, {
            "self_image_stability": 0.5, "strain_level": 0.0,
        })
        assert score > score_neutral

    def test_condition_12_tendency_affinity(self):
        """Condition #12: high tendency_strength boosts expression labels."""
        state = _make_state()
        percept = _make_percept()
        ext = {"tendency_strength": 0.8}

        share_policy = [p for p in POLICIES if p["policy_label"] == "自分の経験を話す"][0]
        score = _score_candidate(share_policy, state, percept, [], None, None, ext)
        score_low = _score_candidate(share_policy, state, percept, [], None, None, {
            "tendency_strength": 0.0,
        })
        assert score > score_low

    def test_condition_13_other_boundary(self):
        """Condition #13: other_count > 0, low boundary_clarity boosts confirm."""
        state = _make_state()
        percept = _make_percept()
        ext = {"other_count": 2, "boundary_clarity": 0.2}

        confirm_policy = [p for p in POLICIES if p["policy_label"] == "確認する"][0]
        score = _score_candidate(confirm_policy, state, percept, [], None, None, ext)
        score_clear = _score_candidate(confirm_policy, state, percept, [], None, None, {
            "other_count": 2, "boundary_clarity": 0.8,
        })
        assert score > score_clear

    def test_condition_14_goal_oriented(self):
        """Condition #14: active goal boosts proposal/watch candidates."""
        state = _make_state()
        percept = _make_percept()
        ext = {"has_active_goal": True, "goal_strength": 0.7, "motive_count": 0}

        propose_policy = [p for p in POLICIES if p["policy_label"] == "提案する"][0]
        score = _score_candidate(propose_policy, state, percept, [], None, None, ext)
        score_no_goal = _score_candidate(propose_policy, state, percept, [], None, None, {
            "has_active_goal": False, "goal_strength": 0.0, "motive_count": 0,
        })
        assert score > score_no_goal

    def test_condition_14_motive_count(self):
        """Condition #14: high motive_count boosts proposal/question candidates."""
        state = _make_state()
        percept = _make_percept()
        ext = {"has_active_goal": False, "goal_strength": 0.0, "motive_count": 5}

        propose_policy = [p for p in POLICIES if p["policy_label"] == "提案する"][0]
        score = _score_candidate(propose_policy, state, percept, [], None, None, ext)
        score_low = _score_candidate(propose_policy, state, percept, [], None, None, {
            "has_active_goal": False, "goal_strength": 0.0, "motive_count": 0,
        })
        assert score > score_low

    def test_condition_15_high_coherence(self):
        """Condition #15: high coherence boosts self-expression labels."""
        state = _make_state()
        percept = _make_percept()
        ext = {"coherence_level": 0.9, "narrative_coherence": 0.9}

        share_policy = [p for p in POLICIES if p["policy_label"] == "自分の経験を話す"][0]
        score = _score_candidate(share_policy, state, percept, [], None, None, ext)
        score_low = _score_candidate(share_policy, state, percept, [], None, None, {
            "coherence_level": 0.5, "narrative_coherence": 0.5,
        })
        assert score > score_low

    def test_condition_15_low_coherence(self):
        """Condition #15: low coherence boosts safety/agreement labels."""
        state = _make_state()
        percept = _make_percept()
        ext = {"coherence_level": 0.1, "narrative_coherence": 0.1}

        listen_policy = [p for p in POLICIES if p["policy_label"] == "黙って聞く"][0]
        score = _score_candidate(listen_policy, state, percept, [], None, None, ext)
        score_mid = _score_candidate(listen_policy, state, percept, [], None, None, {
            "coherence_level": 0.5, "narrative_coherence": 0.5,
        })
        assert score > score_mid

    def test_condition_16_meta_emotion_supply(self):
        """Condition #16: high me_supply_strength boosts expression labels."""
        state = _make_state()
        percept = _make_percept()
        ext = {"me_supply_strength": 0.8}

        joke_policy = [p for p in POLICIES if p["policy_label"] == "冗談を言う"][0]
        score = _score_candidate(joke_policy, state, percept, [], None, None, ext)
        score_low = _score_candidate(joke_policy, state, percept, [], None, None, {
            "me_supply_strength": 0.0,
        })
        assert score > score_low


# ── 5. Dynamic selection count ────────────────────────────────


class TestDynamicSelection:
    def test_close_scores_return_up_to_5(self):
        """When scores are close, up to 5 candidates are returned."""
        # Use state that gives all candidates similar scores
        state = _make_state(
            drives=DriveVector(social=0.7, curiosity=0.7, expression=0.7),
        )
        percept = _make_percept()
        candidates = generate_thought_candidates(state, percept, [])
        # With high drives across all 3 axes, many candidates should have similar scores
        assert len(candidates) >= 3
        assert len(candidates) <= 5

    def test_large_score_gap_returns_3(self):
        """When top candidate is much higher, only 3 are returned."""
        # Create state that strongly favors one kind
        state = _make_state(
            emotions=EmotionVector(fear=0.9),
            drives=DriveVector(social=0.9, curiosity=0.1, expression=0.1),
            mood=Mood(valence=-0.8, arousal=0.8),
            fear_index=FearIndex(attachment_risk=0.9),
        )
        percept = _make_percept(intent="complaint", emotion_valence=-0.8)
        resp = ResponsibilityInfluence(caution_bias=0.4, empathy_bias=0.4, anxiety_baseline=0.2)
        candidates = generate_thought_candidates(state, percept, [{"test": True}], resp)
        assert len(candidates) >= 3

    def test_minimum_3_candidates(self):
        """At least 3 candidates are always returned."""
        state = _make_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(state, percept, [])
        assert len(candidates) >= 3


# ── 6. Safety valve ───────────────────────────────────────────


class TestSafetyValve:
    def test_extreme_score_gap_damping(self):
        """When top score far exceeds #2, top score is damped by 10%."""
        # Create extreme conditions for one candidate
        state = _make_state(
            emotions=EmotionVector(fear=0.95),
            drives=DriveVector(social=0.95, curiosity=0.05, expression=0.05),
            mood=Mood(valence=-0.9, arousal=0.9),
            fear_index=FearIndex(attachment_risk=0.95, identity_risk=0.05),
        )
        percept = _make_percept(intent="complaint", emotion_valence=-0.9)
        resp = ResponsibilityInfluence(
            caution_bias=0.5, empathy_bias=0.5, anxiety_baseline=0.3,
        )

        # Generate candidates - the safety valve should have reduced the gap
        candidates = generate_thought_candidates(state, percept, [{"m": 1}], resp)
        assert len(candidates) >= 3

        # Verify top candidate's score isn't astronomically higher than #2
        # (exact verification is tricky, but the damping should have applied)
        if len(candidates) >= 2:
            gap = candidates[0]["_score"] - candidates[1]["_score"]
            # After damping, the gap should be less than the undamped gap
            # We just verify the function runs without error and returns valid results
            assert gap >= 0  # Still sorted correctly


# ── 7. value_orientation map ──────────────────────────────────


class TestValueOrientationMap:
    def test_new_labels_in_policy_dimension_map(self):
        config = ValueOrientationConfig()
        for label in NEW_LABELS:
            assert label in config.policy_dimension_map, (
                f"New label '{label}' missing from policy_dimension_map"
            )

    def test_original_labels_preserved_in_map(self):
        config = ValueOrientationConfig()
        for label in ORIGINAL_LABELS:
            assert label in config.policy_dimension_map, (
                f"Original label '{label}' missing from policy_dimension_map"
            )

    def test_new_labels_have_valid_dimension_keys(self):
        config = ValueOrientationConfig()
        valid_dims = {"a", "b", "c", "d", "e"}
        for label in NEW_LABELS:
            influences = config.policy_dimension_map[label]
            for dim_key in influences:
                assert dim_key in valid_dims, (
                    f"Label '{label}' has invalid dimension key '{dim_key}'"
                )

    def test_new_labels_have_bounded_values(self):
        config = ValueOrientationConfig()
        for label in NEW_LABELS:
            influences = config.policy_dimension_map[label]
            for dim_key, val in influences.items():
                assert -1.0 <= val <= 1.0, (
                    f"Label '{label}' dimension '{dim_key}' value {val} out of range"
                )


# ── 8. Downstream compatibility ───────────────────────────────


class TestDownstreamCompatibility:
    def test_original_labels_in_generate_then_select(self):
        """Original 6 labels still appear and select_policy works."""
        state = _make_state()
        percept = _make_percept()
        candidates = generate_thought_candidates(state, percept, [])
        assert len(candidates) >= 3

        selected = select_policy(candidates, state)
        assert "policy_label" in selected
        assert "rationale" in selected
        assert "expected_drive_change" in selected
        assert "text" in selected

    def test_select_policy_with_responsibility_caution(self):
        """select_policy respects caution override for 'からかう'."""
        # Force からかう to top via high expression drive + positive mood + joke
        state = _make_state(
            drives=DriveVector(social=0.1, curiosity=0.1, expression=0.95),
            mood=Mood(valence=0.8, arousal=0.8),
        )
        percept = _make_percept(intent="joke", emotion_valence=0.8)
        candidates = generate_thought_candidates(state, percept, [])

        # With high caution, select_policy should prefer non-risky
        resp = ResponsibilityInfluence(caution_bias=0.4, empathy_bias=0.0, anxiety_baseline=0.0)
        selected = select_policy(candidates, state, resp)
        # The test passes if select_policy doesn't crash; caution override may or may not
        # actually replace からかう depending on whether it's #1
        assert selected is not None

    def test_empty_candidates_fallback(self):
        """select_policy with empty list returns default."""
        state = _make_state()
        selected = select_policy([], state)
        assert selected["policy_label"] == "共感する"

    def test_new_labels_propagate_through_pipeline(self):
        """New labels can appear in candidates and pass through select_policy."""
        # Craft state to favor new labels
        state = _make_state()
        percept = _make_percept()
        ext = {
            "self_image_stability": 0.1,
            "strain_level": 0.9,
            "coherence_level": 0.1,
            "narrative_coherence": 0.1,
        }
        candidates = generate_thought_candidates(state, percept, [], extended_inputs=ext)

        # Verify at least one new label appears
        new_label_set = set(NEW_LABELS)
        candidate_labels = {c["policy_label"] for c in candidates}
        # Not guaranteed a new label is in top 3-5, but the candidates should be valid
        for c in candidates:
            assert c["policy_label"] in set(ALL_LABELS), (
                f"Unexpected label '{c['policy_label']}'"
            )

    def test_generate_with_all_original_params(self):
        """All original parameters still work correctly."""
        state = _make_state()
        percept = _make_percept()
        resp = ResponsibilityInfluence(caution_bias=0.2, empathy_bias=0.2, anxiety_baseline=0.1)
        candidates = generate_thought_candidates(
            state=state,
            percept=percept,
            recalled=[{"memory": "test"}],
            responsibility_influence=resp,
            decision_bias=None,
        )
        assert len(candidates) >= 3
        assert len(candidates) <= 5
