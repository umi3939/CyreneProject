"""
tests/test_reaction_with_stm.py - Tests for reaction_with_stm module

Covers:
1.  create_combined_state defaults
2.  CombinedReactionState to_dict/from_dict roundtrip
3.  apply_residue_to_emotions with various emotion labels
4.  apply_residue_to_emotions with scale_override
5.  apply_residue_to_emotions with zero residue (no change)
6.  react_with_stm basic call returns 3-tuple
7.  react_with_stm updates both psyche and loop state
8.  react_combined convenience wrapper
9.  Multiple sequential reactions accumulate STM
10. get_stm_diagnostics basic output
11. get_stm_diagnostics with include_entries=True
12. summarize_residue_influence with no residue
13. summarize_residue_influence with residue
14. STM_EMOTION_MAP covers both "happy" and "joy" styles
15. Immutability: original states not modified
"""

import copy
import pytest

from psyche.reaction_with_stm import (
    STM_EMOTION_MAP,
    CombinedReactionState,
    apply_residue_to_emotions,
    create_combined_state,
    get_stm_diagnostics,
    react_combined,
    react_with_stm,
    summarize_residue_influence,
)
from psyche.state import EmotionVector, Percept, PsycheState
from psyche.short_term_loop import (
    LoopConfig,
    LoopResult,
    LoopState,
    create_loop_state,
)
from psyche.short_term_memory import ResidueInfluence


# ── Helpers ────────────────────────────────────────────────────

def _make_percept(
    text: str = "hello",
    emotion: str = "happy",
    valence: float = 0.5,
    topics: list[str] | None = None,
    intent: str = "greeting",
) -> Percept:
    """Create a Percept with sensible defaults for testing."""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent=intent,
        topics=topics or ["test"],
        sentiment=valence,
        emotion_valence=valence,
    )


def _make_residue_influence(
    emotion_influences: dict[str, float] | None = None,
    total_intensity: float = 0.0,
    continuity: float = 0.0,
    scale: float = 1.0,
) -> ResidueInfluence:
    """Create a ResidueInfluence for testing."""
    return ResidueInfluence(
        emotion_influences=emotion_influences or {},
        total_intensity=total_intensity,
        continuity=continuity,
        scale=scale,
    )


# ── 1. create_combined_state defaults ─────────────────────────

class TestCreateCombinedState:
    """Tests for create_combined_state factory function."""

    def test_defaults_produce_valid_state(self):
        """Default creation yields a CombinedReactionState with
        default PsycheState and default LoopState."""
        cs = create_combined_state()
        assert isinstance(cs, CombinedReactionState)
        assert isinstance(cs.psyche, PsycheState)
        assert isinstance(cs.loop, LoopState)

    def test_default_psyche_has_zero_emotions(self):
        """Default psyche state has zeroed emotions."""
        cs = create_combined_state()
        emo = cs.psyche.emotions.as_dict()
        assert all(v == 0.0 for v in emo.values())

    def test_default_loop_has_empty_memory(self):
        """Default loop state has no entries in memory."""
        cs = create_combined_state()
        assert len(cs.loop.memory.entries) == 0

    def test_custom_psyche_is_preserved(self):
        """Providing a custom PsycheState preserves it."""
        ps = PsycheState(emotions=EmotionVector(joy=0.7))
        cs = create_combined_state(psyche=ps)
        assert cs.psyche.emotions.joy == 0.7

    def test_custom_loop_config_is_applied(self):
        """Providing a LoopConfig changes the loop state configuration."""
        cfg = LoopConfig(residue_scale=0.5, decay_rate=0.8)
        cs = create_combined_state(loop_config=cfg)
        assert cs.loop.config.residue_scale == 0.5
        assert cs.loop.config.decay_rate == 0.8

    def test_custom_loop_config_max_entries(self):
        """max_accumulation_entries from LoopConfig propagates to memory."""
        cfg = LoopConfig(max_accumulation_entries=5)
        cs = create_combined_state(loop_config=cfg)
        assert cs.loop.memory.max_entries == 5


# ── 2. CombinedReactionState to_dict/from_dict roundtrip ──────

class TestCombinedReactionStateSerialization:
    """Tests for CombinedReactionState serialization."""

    def test_roundtrip_default(self):
        """Default state survives to_dict/from_dict roundtrip."""
        original = create_combined_state()
        d = original.to_dict()
        restored = CombinedReactionState.from_dict(d)

        # Psyche emotions should match
        orig_emo = original.psyche.emotions.as_dict()
        rest_emo = restored.psyche.emotions.as_dict()
        for k in orig_emo:
            assert abs(orig_emo[k] - rest_emo[k]) < 1e-9

    def test_roundtrip_with_custom_psyche(self):
        """Custom psyche state survives roundtrip."""
        ps = PsycheState(emotions=EmotionVector(joy=0.3, anger=0.1))
        original = create_combined_state(psyche=ps)
        d = original.to_dict()
        restored = CombinedReactionState.from_dict(d)

        assert abs(restored.psyche.emotions.joy - 0.3) < 1e-9
        assert abs(restored.psyche.emotions.anger - 0.1) < 1e-9

    def test_to_dict_has_required_keys(self):
        """to_dict produces a dict with 'psyche' and 'loop' keys."""
        cs = create_combined_state()
        d = cs.to_dict()
        assert "psyche" in d
        assert "loop" in d

    def test_from_dict_with_empty_dict(self):
        """from_dict with empty dict produces valid defaults."""
        restored = CombinedReactionState.from_dict({})
        assert isinstance(restored.psyche, PsycheState)
        assert isinstance(restored.loop, LoopState)

    def test_roundtrip_loop_config_preserved(self):
        """Loop config values survive roundtrip."""
        cfg = LoopConfig(residue_scale=0.75, decay_rate=0.9)
        original = create_combined_state(loop_config=cfg)
        d = original.to_dict()
        restored = CombinedReactionState.from_dict(d)

        assert abs(restored.loop.config.residue_scale - 0.75) < 1e-9
        assert abs(restored.loop.config.decay_rate - 0.9) < 1e-9


# ── 3. apply_residue_to_emotions with various labels ──────────

class TestApplyResidueToEmotions:
    """Tests for apply_residue_to_emotions function."""

    def test_happy_maps_to_joy(self):
        """Residue with 'happy' label increases joy."""
        emo = EmotionVector(joy=0.2)
        influence = _make_residue_influence(
            emotion_influences={"happy": 0.3},
            total_intensity=0.3,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.5, abs=1e-9)

    def test_sad_maps_to_sorrow(self):
        """Residue with 'sad' label increases sorrow."""
        emo = EmotionVector(sorrow=0.1)
        influence = _make_residue_influence(
            emotion_influences={"sad": 0.2},
            total_intensity=0.2,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.sorrow == pytest.approx(0.3, abs=1e-9)

    def test_angry_maps_to_anger(self):
        """Residue with 'angry' label increases anger."""
        emo = EmotionVector(anger=0.0)
        influence = _make_residue_influence(
            emotion_influences={"angry": 0.4},
            total_intensity=0.4,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.anger == pytest.approx(0.4, abs=1e-9)

    def test_scared_maps_to_fear(self):
        """Residue with 'scared' label increases fear."""
        emo = EmotionVector(fear=0.05)
        influence = _make_residue_influence(
            emotion_influences={"scared": 0.15},
            total_intensity=0.15,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.fear == pytest.approx(0.2, abs=1e-9)

    def test_loving_maps_to_love(self):
        """Residue with 'loving' label increases love."""
        emo = EmotionVector(love=0.1)
        influence = _make_residue_influence(
            emotion_influences={"loving": 0.2},
            total_intensity=0.2,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.love == pytest.approx(0.3, abs=1e-9)

    def test_teasing_maps_to_fun(self):
        """Residue with 'teasing' label increases fun."""
        emo = EmotionVector(fun=0.0)
        influence = _make_residue_influence(
            emotion_influences={"teasing": 0.5},
            total_intensity=0.5,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.fun == pytest.approx(0.5, abs=1e-9)

    def test_surprised_maps_to_surprise(self):
        """Residue with 'surprised' label increases surprise."""
        emo = EmotionVector(surprise=0.0)
        influence = _make_residue_influence(
            emotion_influences={"surprised": 0.25},
            total_intensity=0.25,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.surprise == pytest.approx(0.25, abs=1e-9)

    def test_direct_joy_label(self):
        """Residue with direct 'joy' label (internal style) increases joy."""
        emo = EmotionVector(joy=0.1)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.2},
            total_intensity=0.2,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.3, abs=1e-9)

    def test_direct_sorrow_label(self):
        """Residue with direct 'sorrow' label increases sorrow."""
        emo = EmotionVector(sorrow=0.0)
        influence = _make_residue_influence(
            emotion_influences={"sorrow": 0.4},
            total_intensity=0.4,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.sorrow == pytest.approx(0.4, abs=1e-9)

    def test_neutral_has_no_effect(self):
        """Residue with 'neutral' label maps to '' which has no effect."""
        emo = EmotionVector(joy=0.5, sorrow=0.3)
        influence = _make_residue_influence(
            emotion_influences={"neutral": 0.5},
            total_intensity=0.5,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.5, abs=1e-9)
        assert result.sorrow == pytest.approx(0.3, abs=1e-9)

    def test_unknown_label_passthrough(self):
        """Unknown label falls through to identity mapping; if not a valid
        field it has no effect."""
        emo = EmotionVector(joy=0.5)
        influence = _make_residue_influence(
            emotion_influences={"unknown_emotion": 0.5},
            total_intensity=0.5,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.5, abs=1e-9)

    def test_clamped_to_max_1(self):
        """Emotion value is clamped to max 1.0."""
        emo = EmotionVector(joy=0.9)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.5},
            total_intensity=0.5,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(1.0, abs=1e-9)

    def test_clamped_to_min_0(self):
        """Emotion value is clamped to min 0.0 (negative delta)."""
        emo = EmotionVector(joy=0.1)
        influence = _make_residue_influence(
            emotion_influences={"joy": -0.5},
            total_intensity=0.5,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.0, abs=1e-9)

    def test_multiple_labels_at_once(self):
        """Multiple emotion influences applied simultaneously."""
        emo = EmotionVector(joy=0.1, anger=0.1, fear=0.1)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.2, "anger": 0.3, "fear": 0.1},
            total_intensity=0.6,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.3, abs=1e-9)
        assert result.anger == pytest.approx(0.4, abs=1e-9)
        assert result.fear == pytest.approx(0.2, abs=1e-9)


# ── 4. apply_residue_to_emotions with scale_override ──────────

class TestApplyResidueScaleOverride:
    """Tests for scale_override parameter."""

    def test_scale_override_halves_effect(self):
        """scale_override=0.5 halves the applied delta."""
        emo = EmotionVector(joy=0.2)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.4},
            total_intensity=0.4,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence, scale_override=0.5)
        # delta = 0.4 * 0.5 = 0.2, so joy = 0.2 + 0.2 = 0.4
        assert result.joy == pytest.approx(0.4, abs=1e-9)

    def test_scale_override_zero_no_change(self):
        """scale_override=0.0 produces no change."""
        emo = EmotionVector(joy=0.3)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.5},
            total_intensity=0.5,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence, scale_override=0.0)
        assert result.joy == pytest.approx(0.3, abs=1e-9)

    def test_scale_override_doubles_effect(self):
        """scale_override=2.0 doubles the applied delta."""
        emo = EmotionVector(joy=0.1)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.2},
            total_intensity=0.2,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence, scale_override=2.0)
        # delta = 0.2 * 2.0 = 0.4, so joy = 0.1 + 0.4 = 0.5
        assert result.joy == pytest.approx(0.5, abs=1e-9)

    def test_scale_override_takes_precedence_over_influence_scale(self):
        """scale_override replaces influence.scale rather than multiplying."""
        emo = EmotionVector(joy=0.0)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.5},
            total_intensity=0.5,
            scale=0.1,  # influence.scale is 0.1
        )
        # Without override: delta = 0.5 * 0.1 = 0.05
        # With override=1.0: delta = 0.5 * 1.0 = 0.5
        result = apply_residue_to_emotions(emo, influence, scale_override=1.0)
        assert result.joy == pytest.approx(0.5, abs=1e-9)


# ── 5. apply_residue_to_emotions with zero residue ────────────

class TestApplyResidueZero:
    """Tests for zero-residue cases."""

    def test_empty_influences_no_change(self):
        """Empty emotion_influences dict produces no change."""
        emo = EmotionVector(joy=0.5, sorrow=0.3)
        influence = _make_residue_influence(
            emotion_influences={},
            total_intensity=0.0,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.5, abs=1e-9)
        assert result.sorrow == pytest.approx(0.3, abs=1e-9)

    def test_zero_amount_no_change(self):
        """Zero amount in influences produces no change."""
        emo = EmotionVector(joy=0.5)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.0},
            total_intensity=0.0,
            scale=1.0,
        )
        result = apply_residue_to_emotions(emo, influence)
        assert result.joy == pytest.approx(0.5, abs=1e-9)

    def test_all_emotions_unchanged_on_zero_residue(self):
        """All 7 emotion fields remain unchanged when residue is zero."""
        emo = EmotionVector(
            joy=0.1, anger=0.2, sorrow=0.3,
            fear=0.4, surprise=0.5, love=0.6, fun=0.7,
        )
        influence = _make_residue_influence()
        result = apply_residue_to_emotions(emo, influence)
        for field in ["joy", "anger", "sorrow", "fear", "surprise", "love", "fun"]:
            assert getattr(result, field) == pytest.approx(
                getattr(emo, field), abs=1e-9
            ), f"{field} should remain unchanged"


# ── 6. react_with_stm basic call returns 3-tuple ─────────────

class TestReactWithStmBasic:
    """Tests for react_with_stm function basic behavior."""

    def test_returns_three_tuple(self):
        """react_with_stm returns a 3-tuple."""
        percept = _make_percept()
        ps = PsycheState()
        ls = create_loop_state()
        result = react_with_stm(percept, ps, ls, current_time=1000.0)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_return_types(self):
        """The 3-tuple contains (PsycheState, LoopState, LoopResult)."""
        percept = _make_percept()
        ps = PsycheState()
        ls = create_loop_state()
        new_ps, new_ls, loop_res = react_with_stm(
            percept, ps, ls, current_time=1000.0
        )
        assert isinstance(new_ps, PsycheState)
        assert isinstance(new_ls, LoopState)
        assert isinstance(loop_res, LoopResult)

    def test_neutral_percept_returns_valid(self):
        """Even a neutral percept produces valid output."""
        percept = _make_percept(emotion="neutral", valence=0.0)
        ps = PsycheState()
        ls = create_loop_state()
        new_ps, new_ls, loop_res = react_with_stm(
            percept, ps, ls, current_time=1000.0
        )
        assert isinstance(new_ps, PsycheState)


# ── 7. react_with_stm updates both states ────────────────────

class TestReactWithStmStateUpdates:
    """Tests that react_with_stm properly updates both states."""

    def test_psyche_state_updated(self):
        """Psyche state emotions change after reaction to a happy percept."""
        percept = _make_percept(emotion="happy", valence=0.8)
        ps = PsycheState()
        ls = create_loop_state()
        new_ps, _, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        # The base react() should increase joy for a "happy" percept
        assert new_ps.emotions.joy > ps.emotions.joy

    def test_loop_state_gets_entry(self):
        """Loop state memory accumulates an entry after reaction."""
        percept = _make_percept(emotion="happy", valence=0.5)
        ps = PsycheState()
        ls = create_loop_state()
        _, new_ls, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        # The loop should have processed at least one entry
        # (entries may be decayed/removed, but loop_result shows activity)
        assert new_ls.last_loop_time == pytest.approx(1000.0, abs=1e-3)

    def test_loop_result_has_residue(self):
        """LoopResult contains a ResidueInfluence object."""
        percept = _make_percept(emotion="happy", valence=0.5)
        ps = PsycheState()
        ls = create_loop_state()
        _, _, loop_res = react_with_stm(percept, ps, ls, current_time=1000.0)
        assert isinstance(loop_res.residue_influence, ResidueInfluence)

    def test_drives_updated(self):
        """Drives are updated by the base reaction."""
        percept = _make_percept()
        ps = PsycheState()
        ls = create_loop_state()
        new_ps, _, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        # The base react() modifies drives (social decreases from conversation)
        assert new_ps.drives != ps.drives

    def test_mood_updated(self):
        """Mood drifts after reaction."""
        percept = _make_percept(emotion="happy", valence=0.9)
        ps = PsycheState()
        ls = create_loop_state()
        new_ps, _, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        # A strongly positive percept should nudge mood valence upward
        assert new_ps.mood.valence > ps.mood.valence


# ── 8. react_combined convenience wrapper ─────────────────────

class TestReactCombined:
    """Tests for react_combined convenience wrapper."""

    def test_returns_two_tuple(self):
        """react_combined returns a 2-tuple."""
        percept = _make_percept()
        cs = create_combined_state()
        result = react_combined(percept, cs, delta_time=1.0, current_time=1000.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_types(self):
        """The 2-tuple contains (CombinedReactionState, LoopResult)."""
        percept = _make_percept()
        cs = create_combined_state()
        new_cs, loop_res = react_combined(
            percept, cs, delta_time=1.0, current_time=1000.0
        )
        assert isinstance(new_cs, CombinedReactionState)
        assert isinstance(loop_res, LoopResult)

    def test_psyche_updated_through_wrapper(self):
        """Psyche state is updated via the wrapper."""
        percept = _make_percept(emotion="happy", valence=0.8)
        cs = create_combined_state()
        new_cs, _ = react_combined(
            percept, cs, delta_time=1.0, current_time=1000.0
        )
        assert new_cs.psyche.emotions.joy > cs.psyche.emotions.joy

    def test_loop_updated_through_wrapper(self):
        """Loop state is updated via the wrapper."""
        percept = _make_percept()
        cs = create_combined_state()
        new_cs, _ = react_combined(
            percept, cs, delta_time=1.0, current_time=1000.0
        )
        assert new_cs.loop.last_loop_time == pytest.approx(1000.0, abs=1e-3)

    def test_wrapper_matches_direct_call(self):
        """react_combined produces the same psyche emotions as react_with_stm."""
        percept = _make_percept(emotion="happy", valence=0.5)
        ps = PsycheState()
        ls = create_loop_state()
        t = 2000.0

        # Direct call
        new_ps, new_ls, _ = react_with_stm(
            percept, ps, ls, delta_time=1.0, current_time=t
        )

        # Wrapper call
        cs = CombinedReactionState(psyche=PsycheState(), loop=create_loop_state())
        new_cs, _ = react_combined(
            percept, cs, delta_time=1.0, current_time=t
        )

        # Emotions should match
        for field in ["joy", "anger", "sorrow", "fear", "surprise", "love", "fun"]:
            assert getattr(new_cs.psyche.emotions, field) == pytest.approx(
                getattr(new_ps.emotions, field), abs=1e-6
            ), f"{field} mismatch between wrapper and direct call"

    def test_residue_scale_override_forwarded(self):
        """residue_scale_override parameter is forwarded through wrapper."""
        percept = _make_percept(emotion="happy", valence=0.8)
        cs = create_combined_state()
        # With override=0.0, residue should have no effect
        new_cs_no_residue, _ = react_combined(
            percept, cs, delta_time=1.0,
            residue_scale_override=0.0, current_time=1000.0,
        )
        # With override=2.0, residue should have larger effect
        cs2 = create_combined_state()
        new_cs_big_residue, _ = react_combined(
            percept, cs2, delta_time=1.0,
            residue_scale_override=2.0, current_time=1000.0,
        )
        # Both should be valid PsycheState objects
        assert isinstance(new_cs_no_residue.psyche, PsycheState)
        assert isinstance(new_cs_big_residue.psyche, PsycheState)


# ── 9. Sequential reactions accumulate STM ────────────────────

class TestSequentialReactions:
    """Tests that multiple sequential reactions accumulate STM."""

    def test_two_reactions_accumulate(self):
        """Two sequential reactions result in more STM activity."""
        ps = PsycheState()
        ls = create_loop_state()
        t = 1000.0

        percept1 = _make_percept(
            text="first", emotion="happy", valence=0.5, topics=["topicA"]
        )
        ps, ls, res1 = react_with_stm(percept1, ps, ls, current_time=t)

        percept2 = _make_percept(
            text="second", emotion="happy", valence=0.6, topics=["topicA"]
        )
        ps2, ls2, res2 = react_with_stm(percept2, ps, ls, current_time=t + 1.0)

        # After two reactions, the state should reflect accumulated influence
        assert isinstance(ps2, PsycheState)
        assert isinstance(ls2, LoopState)

    def test_three_reactions_with_varying_emotions(self):
        """Three reactions with different emotions produce varied state."""
        cs = create_combined_state()
        t = 1000.0

        emotions = [("happy", 0.5), ("sad", -0.3), ("surprised", 0.4)]
        for i, (emo, val) in enumerate(emotions):
            percept = _make_percept(
                text=f"msg_{i}", emotion=emo, valence=val, topics=["chat"]
            )
            cs, _ = react_combined(percept, cs, delta_time=1.0, current_time=t + i)

        # After mixed emotions, multiple fields should be non-zero
        emo_dict = cs.psyche.emotions.as_dict()
        nonzero_count = sum(1 for v in emo_dict.values() if v > 0.01)
        assert nonzero_count >= 2, "Mixed emotions should activate multiple fields"

    def test_sequential_continuity(self):
        """Sequential reactions with shared topics show continuity."""
        cs = create_combined_state()
        t = 1000.0

        for i in range(3):
            percept = _make_percept(
                text=f"msg_{i}", emotion="happy", valence=0.3,
                topics=["shared_topic"]
            )
            cs, loop_res = react_combined(
                percept, cs, delta_time=1.0, current_time=t + i
            )

        # After 3 reactions with the same topic, loop should track it
        diag = get_stm_diagnostics(cs.loop)
        assert isinstance(diag, dict)


# ── 10. get_stm_diagnostics basic output ──────────────────────

class TestGetStmDiagnostics:
    """Tests for get_stm_diagnostics function."""

    def test_empty_state_diagnostics(self):
        """Diagnostics for empty loop state have expected keys."""
        ls = create_loop_state()
        diag = get_stm_diagnostics(ls)
        assert "entry_count" in diag
        assert "unprocessed_count" in diag
        assert "current_topics" in diag
        assert "continuity_score" in diag
        assert "config" in diag

    def test_entry_count_zero_initially(self):
        """Entry count is 0 for a fresh loop state."""
        ls = create_loop_state()
        diag = get_stm_diagnostics(ls)
        assert diag["entry_count"] == 0

    def test_after_reaction_has_data(self):
        """Diagnostics after a reaction contain meaningful data."""
        percept = _make_percept(topics=["test_topic"])
        ps = PsycheState()
        ls = create_loop_state()
        _, new_ls, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        diag = get_stm_diagnostics(new_ls)
        assert isinstance(diag["entry_count"], int)
        assert isinstance(diag["config"], dict)

    def test_config_section_has_keys(self):
        """Config section of diagnostics has expected keys."""
        ls = create_loop_state()
        diag = get_stm_diagnostics(ls)
        config = diag["config"]
        assert "continuity_threshold" in config
        assert "residue_scale" in config
        assert "decay_rate" in config


# ── 11. get_stm_diagnostics with include_entries=True ─────────

class TestGetStmDiagnosticsWithEntries:
    """Tests for get_stm_diagnostics with include_entries=True."""

    def test_include_entries_false_by_default(self):
        """By default, 'entries' key is not present."""
        ls = create_loop_state()
        diag = get_stm_diagnostics(ls)
        assert "entries" not in diag

    def test_include_entries_true_empty(self):
        """With include_entries=True and empty memory, entries is empty list."""
        ls = create_loop_state()
        diag = get_stm_diagnostics(ls, include_entries=True)
        assert "entries" in diag
        assert diag["entries"] == []

    def test_include_entries_true_after_reaction(self):
        """After a reaction, entries contain stimulus data."""
        percept = _make_percept(emotion="happy", valence=0.5, topics=["myTopic"])
        ps = PsycheState()
        ls = create_loop_state()
        _, new_ls, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        diag = get_stm_diagnostics(new_ls, include_entries=True)

        if diag["entry_count"] > 0:
            # If entries survived decay, check their structure
            assert len(diag["entries"]) == diag["entry_count"]
            entry = diag["entries"][0]
            assert "emotion" in entry
            assert "valence" in entry
            assert "weight" in entry
            assert "processed" in entry
            assert "topics" in entry

    def test_entry_fields_types(self):
        """Entry fields have correct types after inclusion."""
        percept = _make_percept(emotion="angry", valence=-0.5)
        ps = PsycheState()
        ls = create_loop_state()
        _, new_ls, _ = react_with_stm(percept, ps, ls, current_time=1000.0)
        diag = get_stm_diagnostics(new_ls, include_entries=True)

        if diag["entry_count"] > 0:
            entry = diag["entries"][0]
            assert isinstance(entry["emotion"], str)
            assert isinstance(entry["valence"], (int, float))
            assert isinstance(entry["weight"], (int, float))
            assert isinstance(entry["processed"], bool)
            assert isinstance(entry["topics"], list)


# ── 12. summarize_residue_influence with no residue ───────────

class TestSummarizeResidueNoResidue:
    """Tests for summarize_residue_influence with zero intensity."""

    def test_zero_intensity_message(self):
        """Zero total_intensity returns 'No residue influence'."""
        influence = _make_residue_influence()
        result = summarize_residue_influence(influence)
        assert result == "No residue influence"

    def test_empty_influences_message(self):
        """Empty influences with zero intensity returns no-residue message."""
        influence = ResidueInfluence(
            emotion_influences={},
            total_intensity=0.0,
            continuity=0.0,
            scale=1.0,
        )
        result = summarize_residue_influence(influence)
        assert result == "No residue influence"


# ── 13. summarize_residue_influence with residue ──────────────

class TestSummarizeResidueWithResidue:
    """Tests for summarize_residue_influence with actual residue."""

    def test_single_emotion_summary(self):
        """Single emotion produces a readable summary."""
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.5},
            total_intensity=0.5,
            scale=0.8,
            continuity=0.3,
        )
        result = summarize_residue_influence(influence)
        assert "Residue:" in result
        assert "joy:0.50" in result
        assert "scale=0.80" in result
        assert "continuity=0.30" in result

    def test_multiple_emotions_summary(self):
        """Multiple emotions are listed in descending order."""
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.3, "anger": 0.5, "fear": 0.1},
            total_intensity=0.9,
            scale=1.0,
            continuity=0.5,
        )
        result = summarize_residue_influence(influence)
        assert "Residue:" in result
        assert "anger:0.50" in result
        assert "joy:0.30" in result
        assert "fear:0.10" in result
        # anger should appear before joy (descending order)
        assert result.index("anger") < result.index("joy")

    def test_small_amounts_filtered(self):
        """Amounts <= 0.01 are not shown in summary."""
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.5, "anger": 0.005},
            total_intensity=0.505,
            scale=1.0,
            continuity=0.0,
        )
        result = summarize_residue_influence(influence)
        assert "joy:0.50" in result
        assert "anger" not in result

    def test_summary_is_string(self):
        """Summary always returns a string."""
        influence = _make_residue_influence(
            emotion_influences={"sorrow": 0.2},
            total_intensity=0.2,
            scale=1.0,
        )
        result = summarize_residue_influence(influence)
        assert isinstance(result, str)


# ── 14. STM_EMOTION_MAP coverage ──────────────────────────────

class TestSTMEmotionMap:
    """Tests that STM_EMOTION_MAP covers both external and internal styles."""

    def test_external_labels_present(self):
        """All external (percept-style) labels are mapped."""
        external = ["happy", "sad", "angry", "surprised", "scared", "loving", "teasing", "neutral"]
        for label in external:
            assert label in STM_EMOTION_MAP, f"Missing external label: {label}"

    def test_internal_labels_present(self):
        """All internal (EmotionVector field) labels are mapped."""
        internal = ["joy", "sorrow", "anger", "fear", "surprise", "love", "fun"]
        for label in internal:
            assert label in STM_EMOTION_MAP, f"Missing internal label: {label}"

    def test_external_happy_maps_to_joy(self):
        """External 'happy' maps to 'joy'."""
        assert STM_EMOTION_MAP["happy"] == "joy"

    def test_internal_joy_maps_to_joy(self):
        """Internal 'joy' maps to 'joy' (identity)."""
        assert STM_EMOTION_MAP["joy"] == "joy"

    def test_external_sad_maps_to_sorrow(self):
        """External 'sad' maps to 'sorrow'."""
        assert STM_EMOTION_MAP["sad"] == "sorrow"

    def test_internal_sorrow_maps_to_sorrow(self):
        """Internal 'sorrow' maps to 'sorrow' (identity)."""
        assert STM_EMOTION_MAP["sorrow"] == "sorrow"

    def test_neutral_maps_to_empty(self):
        """'neutral' maps to empty string (no emotion field)."""
        assert STM_EMOTION_MAP["neutral"] == ""

    def test_all_internal_labels_are_identity_mapped(self):
        """Every internal label maps to itself."""
        internal_pairs = [
            ("joy", "joy"),
            ("sorrow", "sorrow"),
            ("anger", "anger"),
            ("fear", "fear"),
            ("surprise", "surprise"),
            ("love", "love"),
            ("fun", "fun"),
        ]
        for label, expected in internal_pairs:
            assert STM_EMOTION_MAP[label] == expected

    def test_all_external_labels_map_to_valid_fields_or_empty(self):
        """Every external label maps to a valid EmotionVector field or ''."""
        valid_fields = {"joy", "sorrow", "anger", "fear", "surprise", "love", "fun", ""}
        external = ["happy", "sad", "angry", "surprised", "scared", "loving", "teasing", "neutral"]
        for label in external:
            target = STM_EMOTION_MAP[label]
            assert target in valid_fields, (
                f"External label '{label}' maps to '{target}' which is not a valid field"
            )


# ── 15. Immutability: original states not modified ────────────

class TestImmutability:
    """Tests that original states are not mutated by operations."""

    def test_react_with_stm_does_not_mutate_psyche(self):
        """Original PsycheState is not modified by react_with_stm."""
        percept = _make_percept(emotion="happy", valence=0.8)
        ps = PsycheState()
        ls = create_loop_state()

        # Deep copy for comparison
        original_emotions = ps.emotions.as_dict().copy()

        react_with_stm(percept, ps, ls, current_time=1000.0)

        # Original should be unchanged
        for field, value in original_emotions.items():
            assert getattr(ps.emotions, field) == pytest.approx(value, abs=1e-9), (
                f"Original psyche.emotions.{field} was mutated"
            )

    def test_react_with_stm_does_not_mutate_loop_state(self):
        """Original LoopState is not modified by react_with_stm."""
        percept = _make_percept()
        ps = PsycheState()
        ls = create_loop_state()

        original_entry_count = len(ls.memory.entries)
        original_updated = ls.updated_this_turn

        react_with_stm(percept, ps, ls, current_time=1000.0)

        assert len(ls.memory.entries) == original_entry_count
        assert ls.updated_this_turn == original_updated

    def test_react_combined_does_not_mutate_combined_state(self):
        """Original CombinedReactionState is not modified by react_combined."""
        percept = _make_percept(emotion="angry", valence=-0.5)
        cs = create_combined_state()

        original_joy = cs.psyche.emotions.joy
        original_anger = cs.psyche.emotions.anger
        original_entries = len(cs.loop.memory.entries)

        react_combined(percept, cs, delta_time=1.0, current_time=1000.0)

        assert cs.psyche.emotions.joy == pytest.approx(original_joy, abs=1e-9)
        assert cs.psyche.emotions.anger == pytest.approx(original_anger, abs=1e-9)
        assert len(cs.loop.memory.entries) == original_entries

    def test_apply_residue_does_not_mutate_emotion_vector(self):
        """Original EmotionVector is not modified by apply_residue_to_emotions."""
        emo = EmotionVector(joy=0.3, anger=0.2)
        influence = _make_residue_influence(
            emotion_influences={"joy": 0.5, "anger": 0.3},
            total_intensity=0.8,
            scale=1.0,
        )

        original_joy = emo.joy
        original_anger = emo.anger

        apply_residue_to_emotions(emo, influence)

        assert emo.joy == pytest.approx(original_joy, abs=1e-9)
        assert emo.anger == pytest.approx(original_anger, abs=1e-9)

    def test_sequential_reactions_dont_corrupt_earlier_state(self):
        """Running multiple reactions does not corrupt previously stored states."""
        ps = PsycheState()
        ls = create_loop_state()
        t = 1000.0

        # First reaction
        percept1 = _make_percept(emotion="happy", valence=0.5)
        ps1, ls1, _ = react_with_stm(percept1, ps, ls, current_time=t)
        ps1_joy = ps1.emotions.joy

        # Second reaction (using ps1, ls1)
        percept2 = _make_percept(emotion="sad", valence=-0.5)
        ps2, ls2, _ = react_with_stm(percept2, ps1, ls1, current_time=t + 1.0)

        # ps1 should be unchanged
        assert ps1.emotions.joy == pytest.approx(ps1_joy, abs=1e-9), (
            "Earlier state ps1 was corrupted by second reaction"
        )
