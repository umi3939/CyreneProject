"""
tests/test_expression.py - Tests for Expression Generation Module

Verifies:
1. _build_render_prompt includes all sections (persona, policy, screen_context,
   history, psyche_enrichment, memory)
2. _parse_expression_output with valid JSON
3. _parse_expression_output with markdown-fenced JSON
4. _parse_expression_output with invalid JSON -> fallback
5. _parse_expression_output meta defaults (emotion, intensity, action)
6. _fallback_expression all branches (high fear+attachment, high fear only,
   positive mood, negative mood, neutral/policy fallback)
7. render_expression with mock LLM
8. render_expression fallback on LLM failure (ImportError path)
9. Empty memory_snippet handling
10. Empty screen_context and recent_history
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from psyche.expression import (
    render_expression,
    _build_render_prompt,
    _parse_expression_output,
    _fallback_expression,
)
from psyche.state import PsycheState, EmotionVector, DriveVector, Mood
from psyche.pillars import FearIndex


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def default_state():
    """PsycheState with all defaults (emotions 0.0, mood valence=0 arousal=0.3)."""
    return PsycheState()


@pytest.fixture
def default_policy():
    return {"policy_label": "共感する", "rationale": "ユーザに寄り添う"}


@pytest.fixture
def default_persona():
    return {
        "name": "キュレネ",
        "tone": "romantic, sweet",
        "style_rules": {
            "禁止": ["敬語", "タメ口すぎる表現"],
            "推奨": ["語尾に♪", "柔らかい表現"],
        },
    }


@pytest.fixture
def sample_memory():
    return [
        {"summary": "ユーザと一緒にゲームをした"},
        {"summary": "キュレネが笑った"},
    ]


@pytest.fixture
def high_fear_high_attachment_state():
    """State where fear_level > 0.5 and attachment_risk > 0.5."""
    return PsycheState(
        fear_index=FearIndex(
            identity_risk=0.8,
            attachment_risk=0.9,
            continuity_risk=0.7,
            projection_risk=0.6,
        ),
    )


@pytest.fixture
def high_fear_low_attachment_state():
    """State where fear_level > 0.5 but attachment_risk <= 0.5."""
    return PsycheState(
        fear_index=FearIndex(
            identity_risk=1.0,
            attachment_risk=0.3,
            continuity_risk=1.0,
            projection_risk=1.0,
        ),
    )


@pytest.fixture
def positive_mood_state():
    """State with mood valence > 0.3, no fear."""
    return PsycheState(
        mood=Mood(valence=0.5, arousal=0.5),
    )


@pytest.fixture
def negative_mood_state():
    """State with mood valence < -0.3, no fear."""
    return PsycheState(
        mood=Mood(valence=-0.5, arousal=0.5),
    )


@pytest.fixture
def neutral_state():
    """State with neutral mood (valence between -0.3 and 0.3), no fear."""
    return PsycheState(
        mood=Mood(valence=0.0, arousal=0.3),
    )


# ── 1. _build_render_prompt ──────────────────────────────────

class TestBuildRenderPrompt:

    def test_includes_persona_name(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "キュレネ" in prompt

    def test_includes_persona_tone(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "romantic, sweet" in prompt

    def test_includes_policy_label(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "共感する" in prompt

    def test_includes_policy_rationale(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "ユーザに寄り添う" in prompt

    def test_includes_emotion_info(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "支配的感情" in prompt
        assert "全体感情" in prompt
        assert "気分" in prompt

    def test_includes_fear_summary(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "喪失恐怖" in prompt

    def test_includes_memory_snippets(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "ユーザと一緒にゲームをした" in prompt
        assert "キュレネが笑った" in prompt

    def test_includes_screen_context(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            screen_context="RPGゲーム画面が表示されている",
        )
        assert "RPGゲーム画面が表示されている" in prompt

    def test_empty_screen_context_shows_placeholder(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            screen_context="",
        )
        assert "(画面情報なし)" in prompt

    def test_includes_recent_history(self, default_state, default_policy, sample_memory, default_persona):
        history = ["ユーザ: こんにちは", "キュレネ: やっほー♪"]
        prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            recent_history=history,
        )
        assert "ユーザ: こんにちは" in prompt
        assert "キュレネ: やっほー♪" in prompt

    def test_empty_history_shows_placeholder(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            recent_history=None,
        )
        assert "直近の会話" in prompt
        assert "(なし)" in prompt

    def test_includes_psyche_enrichment(self, default_state, default_policy, sample_memory, default_persona):
        enrichment = "【内省メモ】自分の変化に気づいている"
        prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            psyche_enrichment=enrichment,
        )
        assert "【内省メモ】自分の変化に気づいている" in prompt

    def test_empty_psyche_enrichment(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            psyche_enrichment="",
        )
        # Should not cause errors; the empty string simply produces nothing extra
        assert "禁止パターン:" in prompt
        # When enrichment is empty, the inner context section should not appear
        assert "内面的文脈" not in prompt

    def test_includes_prohibitions_and_recommendations(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "敬語" in prompt
        assert "タメ口すぎる表現" in prompt
        assert "語尾に♪" in prompt
        assert "柔らかい表現" in prompt

    def test_memory_truncated_to_three(self, default_state, default_policy, default_persona):
        """Only first 3 memory snippets should be included."""
        memories = [
            {"summary": f"記憶{i}"} for i in range(5)
        ]
        prompt = _build_render_prompt(default_state, default_policy, memories, default_persona)
        assert "記憶0" in prompt
        assert "記憶1" in prompt
        assert "記憶2" in prompt
        assert "記憶3" not in prompt
        assert "記憶4" not in prompt

    def test_history_truncated_to_five(self, default_state, default_policy, default_persona):
        """Only last 5 history entries should be included."""
        history = [f"line{i}" for i in range(10)]
        prompt = _build_render_prompt(
            default_state, default_policy, [], default_persona,
            recent_history=history,
        )
        assert "line5" in prompt
        assert "line9" in prompt
        assert "line4" not in prompt

    def test_persona_defaults(self, default_state, default_policy, sample_memory):
        """Persona with missing keys should use defaults."""
        minimal_persona = {}
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, minimal_persona)
        assert "キュレネ" in prompt  # default name
        assert "romantic, sweet" in prompt  # default tone
        assert "なし" in prompt  # no prohibitions/recommendations

    def test_output_format_section_present(self, default_state, default_policy, sample_memory, default_persona):
        prompt = _build_render_prompt(default_state, default_policy, sample_memory, default_persona)
        assert "═══ 出力形式（JSONのみ）═══" in prompt


# ── 2–5. _parse_expression_output ─────────────────────────────

class TestParseExpressionOutput:

    def test_valid_json_with_text_and_meta(self, default_state, default_policy):
        raw = json.dumps({
            "text": "テスト発話",
            "meta": {"emotion": "joy", "intensity": 0.8, "action": "応援する"},
        })
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "テスト発話"
        assert result["meta"]["emotion"] == "joy"
        assert result["meta"]["intensity"] == 0.8
        assert result["meta"]["action"] == "応援する"

    def test_valid_json_with_text_only_gets_meta_defaults(self, default_state, default_policy):
        """When meta is missing from JSON, defaults should be filled from state/policy."""
        raw = json.dumps({"text": "なにかの発話"})
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "なにかの発話"
        assert result["meta"]["emotion"] == default_state.dominant_emotion
        assert result["meta"]["intensity"] == round(default_state.dominant_emotion_value, 2)
        assert result["meta"]["action"] == "共感する"

    def test_partial_meta_gets_defaults_for_missing_fields(self, default_state, default_policy):
        raw = json.dumps({
            "text": "部分的メタ",
            "meta": {"emotion": "sorrow"},
        })
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["meta"]["emotion"] == "sorrow"
        assert result["meta"]["intensity"] == round(default_state.dominant_emotion_value, 2)
        assert result["meta"]["action"] == "共感する"

    def test_markdown_fenced_json(self, default_state, default_policy):
        raw = '```json\n{"text": "フェンス内発話", "meta": {"emotion": "joy", "intensity": 0.5, "action": "test"}}\n```'
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "フェンス内発話"
        assert result["meta"]["emotion"] == "joy"

    def test_markdown_fenced_no_language_tag(self, default_state, default_policy):
        raw = '```\n{"text": "言語タグなし", "meta": {}}\n```'
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "言語タグなし"

    def test_invalid_json_returns_fallback(self, default_state, default_policy):
        raw = "これはJSONではありません"
        result = _parse_expression_output(raw, default_state, default_policy)
        # Should fall back - the fallback for neutral state uses policy text
        assert "text" in result
        assert "meta" in result

    def test_json_without_text_key_returns_fallback(self, default_state, default_policy):
        raw = json.dumps({"response": "textキーがない"})
        result = _parse_expression_output(raw, default_state, default_policy)
        # No "text" key -> fallback
        assert result == _fallback_expression(default_state, default_policy)

    def test_json_array_returns_fallback(self, default_state, default_policy):
        raw = json.dumps([{"text": "配列はダメ"}])
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result == _fallback_expression(default_state, default_policy)

    def test_empty_string_returns_fallback(self, default_state, default_policy):
        raw = ""
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result == _fallback_expression(default_state, default_policy)

    def test_non_dict_meta_replaced_with_defaults(self, default_state, default_policy):
        """If meta is not a dict, it should be replaced with empty dict and defaults filled."""
        raw = json.dumps({"text": "メタが文字列", "meta": "not_a_dict"})
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "メタが文字列"
        assert isinstance(result["meta"], dict)
        assert result["meta"]["emotion"] == default_state.dominant_emotion
        assert result["meta"]["intensity"] == round(default_state.dominant_emotion_value, 2)
        assert result["meta"]["action"] == "共感する"

    def test_whitespace_around_json(self, default_state, default_policy):
        raw = '  \n  {"text": "空白あり", "meta": {}}  \n  '
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "空白あり"

    def test_policy_label_default_in_meta(self, default_state):
        """When policy has no policy_label, action defaults to 'unknown'."""
        policy_no_label = {}
        raw = json.dumps({"text": "ラベルなし"})
        result = _parse_expression_output(raw, default_state, policy_no_label)
        assert result["meta"]["action"] == "unknown"


# ── 6. _fallback_expression ───────────────────────────────────

class TestFallbackExpression:

    def test_high_fear_high_attachment(self, high_fear_high_attachment_state, default_policy):
        result = _fallback_expression(high_fear_high_attachment_state, default_policy)
        assert result["text"] == "...ねえ、どこにも行かないで。あたしのそばにいて"
        assert "meta" in result

    def test_high_fear_low_attachment(self, high_fear_low_attachment_state, default_policy):
        result = _fallback_expression(high_fear_low_attachment_state, default_policy)
        assert result["text"] == "...怖いの。あたしが、あたしでなくなってしまいそうで"

    def test_positive_mood(self, positive_mood_state, default_policy):
        result = _fallback_expression(positive_mood_state, default_policy)
        assert result["text"] == "ふふっ、なんだか楽しいわね♪"

    def test_negative_mood(self, negative_mood_state, default_policy):
        result = _fallback_expression(negative_mood_state, default_policy)
        assert result["text"] == "...少し、考えさせて"

    def test_neutral_with_policy_text(self, neutral_state):
        policy_with_text = {"policy_label": "提案する", "text": "何かしようか？"}
        result = _fallback_expression(neutral_state, policy_with_text)
        assert result["text"] == "何かしようか？"

    def test_neutral_without_policy_text(self, neutral_state, default_policy):
        result = _fallback_expression(neutral_state, default_policy)
        assert result["text"] == "そうね...あなたはどう思う？"

    def test_meta_contains_dominant_emotion(self, default_state, default_policy):
        result = _fallback_expression(default_state, default_policy)
        assert result["meta"]["emotion"] == default_state.dominant_emotion

    def test_meta_contains_intensity(self, default_state, default_policy):
        result = _fallback_expression(default_state, default_policy)
        assert result["meta"]["intensity"] == round(default_state.dominant_emotion_value, 2)

    def test_meta_contains_action_from_policy_label(self, default_state, default_policy):
        result = _fallback_expression(default_state, default_policy)
        assert result["meta"]["action"] == "共感する"

    def test_meta_action_default_when_no_policy_label(self, default_state):
        result = _fallback_expression(default_state, {})
        assert result["meta"]["action"] == "共感する"

    def test_fear_priority_over_positive_mood(self):
        """Fear branch takes priority even when mood is positive."""
        state = PsycheState(
            mood=Mood(valence=0.8, arousal=0.5),
            fear_index=FearIndex(
                identity_risk=1.0,
                attachment_risk=1.0,
                continuity_risk=1.0,
                projection_risk=1.0,
            ),
        )
        result = _fallback_expression(state, {})
        # Fear > 0.5 should take priority over positive mood
        assert "怖い" in result["text"] or "行かないで" in result["text"]

    def test_fear_exactly_at_boundary(self):
        """Fear exactly at 0.5 should NOT trigger fear branch (> 0.5 required)."""
        # value = identity*0.3 + attachment*0.3 + continuity*0.2 + projection*0.2
        # To get exactly 0.5: set all to 0.5 => 0.5*0.3 + 0.5*0.3 + 0.5*0.2 + 0.5*0.2 = 0.5
        state = PsycheState(
            mood=Mood(valence=0.5, arousal=0.5),
            fear_index=FearIndex(
                identity_risk=0.5,
                attachment_risk=0.5,
                continuity_risk=0.5,
                projection_risk=0.5,
            ),
        )
        result = _fallback_expression(state, {})
        # fear == 0.5 is NOT > 0.5, so should fall through to mood check
        assert result["text"] == "ふふっ、なんだか楽しいわね♪"

    def test_no_fear_index_means_fear_zero(self, default_policy):
        """State with fear_index=None should have fear_level=0.0."""
        state = PsycheState(mood=Mood(valence=0.0, arousal=0.3))
        assert state.fear_level == 0.0
        result = _fallback_expression(state, default_policy)
        # Neutral mood, so should use policy fallback
        assert result["text"] == "そうね...あなたはどう思う？"

    def test_with_emotions_set(self, default_policy):
        """Fallback meta reflects the dominant emotion from state."""
        state = PsycheState(
            emotions=EmotionVector(joy=0.8, sorrow=0.2),
            mood=Mood(valence=0.5, arousal=0.5),
        )
        result = _fallback_expression(state, default_policy)
        assert result["meta"]["emotion"] == "joy"
        assert result["meta"]["intensity"] == 0.8


# ── 7. render_expression with mock LLM ───────────────────────

class TestRenderExpression:

    @pytest.mark.asyncio
    async def test_render_with_import_success(self, default_state, default_policy, sample_memory, default_persona):
        """When llm_wrapper is importable, uses llm_call_with_system."""
        expected_raw = json.dumps({
            "text": "LLM応答テスト",
            "meta": {"emotion": "joy", "intensity": 0.7, "action": "共感する"},
        })

        mock_llm_fn = AsyncMock(return_value="unused")
        mock_system_call = AsyncMock(return_value=expected_raw)

        with patch(
            "psyche.expression.llm_call_with_system",
            mock_system_call,
            create=True,
        ):
            # Patch the import inside render_expression
            import psyche.expression as expr_mod
            original_render = expr_mod.render_expression

            # We need to patch the import that happens inside the function
            with patch.dict("sys.modules", {
                "src.llm_wrapper": type("Module", (), {
                    "EXPRESSION_SYSTEM_PROMPT": "test_prompt",
                    "llm_call_with_system": mock_system_call,
                })(),
            }):
                result = await render_expression(
                    default_state, default_policy, sample_memory, default_persona,
                    mock_llm_fn,
                )

        assert result["text"] == "LLM応答テスト"
        assert result["meta"]["emotion"] == "joy"

    @pytest.mark.asyncio
    async def test_render_fallback_on_import_error(self, default_state, default_policy, sample_memory, default_persona):
        """When src.llm_wrapper is not importable, falls back to llm_call_fn."""
        expected_raw = json.dumps({
            "text": "フォールバックLLM",
            "meta": {"emotion": "love", "intensity": 0.6, "action": "共感する"},
        })

        mock_llm_fn = AsyncMock(return_value=expected_raw)

        # Ensure ImportError is raised by removing the module if present
        import sys
        saved = sys.modules.pop("src.llm_wrapper", None)
        try:
            # Force ImportError by patching the import
            with patch.dict("sys.modules", {"src": None, "src.llm_wrapper": None}):
                # The import inside render_expression will raise ImportError
                result = await render_expression(
                    default_state, default_policy, sample_memory, default_persona,
                    mock_llm_fn,
                )
        finally:
            if saved is not None:
                sys.modules["src.llm_wrapper"] = saved

        assert result["text"] == "フォールバックLLM"
        mock_llm_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_render_llm_returns_bad_json(self, default_state, default_policy, sample_memory, default_persona):
        """When LLM returns invalid JSON, falls back to rule-based expression."""
        mock_llm_fn = AsyncMock(return_value="これはJSONではない")

        import sys
        saved = sys.modules.pop("src.llm_wrapper", None)
        try:
            with patch.dict("sys.modules", {"src": None, "src.llm_wrapper": None}):
                result = await render_expression(
                    default_state, default_policy, sample_memory, default_persona,
                    mock_llm_fn,
                )
        finally:
            if saved is not None:
                sys.modules["src.llm_wrapper"] = saved

        expected_fallback = _fallback_expression(default_state, default_policy)
        assert result == expected_fallback

    @pytest.mark.asyncio
    async def test_render_with_all_optional_params(self, default_state, default_policy, sample_memory, default_persona):
        """Passing all optional parameters should work without errors."""
        expected_raw = json.dumps({
            "text": "全パラメータテスト",
            "meta": {"emotion": "joy", "intensity": 0.5, "action": "共感する"},
        })
        mock_llm_fn = AsyncMock(return_value=expected_raw)

        import sys
        saved = sys.modules.pop("src.llm_wrapper", None)
        try:
            with patch.dict("sys.modules", {"src": None, "src.llm_wrapper": None}):
                result = await render_expression(
                    default_state, default_policy, sample_memory, default_persona,
                    mock_llm_fn,
                    screen_context="ゲーム画面",
                    recent_history=["ユーザ: テスト"],
                    psyche_enrichment="内省データ",
                )
        finally:
            if saved is not None:
                sys.modules["src.llm_wrapper"] = saved

        assert result["text"] == "全パラメータテスト"

    @pytest.mark.asyncio
    async def test_render_passes_prompt_to_llm(self, default_state, default_policy, sample_memory, default_persona):
        """The prompt passed to llm_call_fn should be the output of _build_render_prompt."""
        mock_llm_fn = AsyncMock(return_value=json.dumps({"text": "ok", "meta": {}}))

        import sys
        saved = sys.modules.pop("src.llm_wrapper", None)
        try:
            with patch.dict("sys.modules", {"src": None, "src.llm_wrapper": None}):
                await render_expression(
                    default_state, default_policy, sample_memory, default_persona,
                    mock_llm_fn,
                    screen_context="ctx",
                    recent_history=["h1"],
                    psyche_enrichment="enrich",
                )
        finally:
            if saved is not None:
                sys.modules["src.llm_wrapper"] = saved

        expected_prompt = _build_render_prompt(
            default_state, default_policy, sample_memory, default_persona,
            screen_context="ctx",
            recent_history=["h1"],
            psyche_enrichment="enrich",
        )
        mock_llm_fn.assert_called_once_with(expected_prompt)


# ── 8–10. Edge cases ─────────────────────────────────────────

class TestEdgeCases:

    def test_empty_memory_snippet(self, default_state, default_policy, default_persona):
        """Empty memory list should produce (なし) placeholder."""
        prompt = _build_render_prompt(default_state, default_policy, [], default_persona)
        # The memory section should show "(なし)"
        lines = prompt.split("\n")
        memory_section_idx = None
        for i, line in enumerate(lines):
            if "関連記憶（参考）:" in line:
                memory_section_idx = i
                break
        assert memory_section_idx is not None
        assert "(なし)" in lines[memory_section_idx + 1]

    def test_memory_with_missing_summary_key(self, default_state, default_policy, default_persona):
        """Memory entries without 'summary' key should produce empty strings."""
        memories = [{"content": "no summary key"}]
        prompt = _build_render_prompt(default_state, default_policy, memories, default_persona)
        assert "- " in prompt  # The line is present but with empty summary

    def test_empty_screen_context_and_none_history(self, default_state, default_policy, default_persona):
        prompt = _build_render_prompt(
            default_state, default_policy, [], default_persona,
            screen_context="",
            recent_history=None,
        )
        assert "(画面情報なし)" in prompt
        assert "(なし)" in prompt

    def test_none_screen_context_is_same_as_empty(self, default_state, default_policy, default_persona):
        """screen_context="" and not passing it should produce same result."""
        prompt_empty = _build_render_prompt(
            default_state, default_policy, [], default_persona,
            screen_context="",
        )
        prompt_default = _build_render_prompt(
            default_state, default_policy, [], default_persona,
        )
        assert prompt_empty == prompt_default

    def test_empty_recent_history_list(self, default_state, default_policy, default_persona):
        """Empty list (not None) should produce empty formatted_history."""
        prompt = _build_render_prompt(
            default_state, default_policy, [], default_persona,
            recent_history=[],
        )
        # Empty list is falsy, so formatted_history will be "", which triggers "(なし)"
        lines = prompt.split("\n")
        history_section_idx = None
        for i, line in enumerate(lines):
            if "直近の会話:" in line:
                history_section_idx = i
                break
        assert history_section_idx is not None
        assert "(なし)" in lines[history_section_idx + 1]

    def test_parse_handles_none_meta_gracefully(self, default_state, default_policy):
        """JSON with meta=null should be treated as empty dict."""
        raw = json.dumps({"text": "nullメタ", "meta": None})
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "nullメタ"
        assert result["meta"]["emotion"] == default_state.dominant_emotion

    def test_fallback_return_structure(self, default_state, default_policy):
        """Fallback always returns dict with 'text' and 'meta' keys."""
        result = _fallback_expression(default_state, default_policy)
        assert isinstance(result, dict)
        assert "text" in result
        assert "meta" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["meta"], dict)
        assert "emotion" in result["meta"]
        assert "intensity" in result["meta"]
        assert "action" in result["meta"]

    def test_parse_valid_json_return_structure(self, default_state, default_policy):
        """Parsed output always returns dict with 'text' and 'meta' keys with required fields."""
        raw = json.dumps({"text": "構造チェック", "meta": {}})
        result = _parse_expression_output(raw, default_state, default_policy)
        assert isinstance(result, dict)
        assert "text" in result
        assert "meta" in result
        assert "emotion" in result["meta"]
        assert "intensity" in result["meta"]
        assert "action" in result["meta"]

    def test_state_with_high_joy_dominant_emotion(self, default_policy):
        """State with high joy should report 'joy' as dominant emotion in meta."""
        state = PsycheState(
            emotions=EmotionVector(joy=0.9, sorrow=0.1, fear=0.05),
        )
        result = _fallback_expression(state, default_policy)
        assert result["meta"]["emotion"] == "joy"
        assert result["meta"]["intensity"] == 0.9

    def test_markdown_fenced_with_extra_text_before(self, default_state, default_policy):
        """Markdown fence with extra lines (not starting with ```) should be kept."""
        raw = '```json\n{"text": "テスト", "meta": {}}\n```'
        result = _parse_expression_output(raw, default_state, default_policy)
        assert result["text"] == "テスト"

    @pytest.mark.asyncio
    async def test_render_expression_return_type(self, default_state, default_policy, sample_memory, default_persona):
        """render_expression always returns a dict."""
        mock_llm_fn = AsyncMock(return_value=json.dumps({"text": "型テスト", "meta": {}}))

        import sys
        saved = sys.modules.pop("src.llm_wrapper", None)
        try:
            with patch.dict("sys.modules", {"src": None, "src.llm_wrapper": None}):
                result = await render_expression(
                    default_state, default_policy, sample_memory, default_persona,
                    mock_llm_fn,
                )
        finally:
            if saved is not None:
                sys.modules["src.llm_wrapper"] = saved

        assert isinstance(result, dict)
        assert "text" in result
        assert "meta" in result
