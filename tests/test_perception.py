"""
tests/test_perception.py - Tests for Input Interpretation (perception module).

Verifies:
1. Each emotion keyword maps to the correct (emotion, valence) tuple
2. Each intent keyword maps to the correct intent label
3. Neutral default when no keywords match
4. Valence values are correct and clamped to [-1, 1]
5. Topics populated from keyword matches
6. parse_percept without LLM returns heuristic result
7. parse_percept with mock LLM returns enriched result
8. LLM enrichment fallback on bad JSON
9. LLM enrichment fallback on missing fields
10. LLM enrichment with markdown code fences stripped
11. State context passed to LLM prompt
12. Multiple keywords - first match wins
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest

from psyche.perception import (
    _EMOTION_KEYWORDS,
    _INTENT_KEYWORDS,
    _heuristic_parse,
    _llm_enrich,
    parse_percept,
)
from psyche.state import EmotionVector, Mood, Percept, PsycheState


# ── Helpers ───────────────────────────────────────────────────

def _make_state(mood_valence: float = 0.3, joy: float = 0.4) -> PsycheState:
    """Create a minimal PsycheState for testing."""
    return PsycheState(
        emotions=EmotionVector(joy=joy),
        mood=Mood(valence=mood_valence, arousal=0.5),
    )


def _make_llm_fn(response: str) -> AsyncMock:
    """Create an async mock LLM function returning *response*."""
    return AsyncMock(return_value=response)


@contextmanager
def _hide_llm_wrapper():
    """Temporarily hide src.llm_wrapper from sys.modules so the local
    ``from src.llm_wrapper import ...`` inside ``_llm_enrich`` raises
    ``ImportError``, forcing fallback to the provided ``llm_call_fn``.
    """
    saved = {}
    keys_to_hide = [k for k in sys.modules if k == "src.llm_wrapper" or k.startswith("src.llm_wrapper.")]
    for k in keys_to_hide:
        saved[k] = sys.modules.pop(k)
    # Insert a None sentinel so the import raises ImportError
    sys.modules["src.llm_wrapper"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        # Remove sentinel
        sys.modules.pop("src.llm_wrapper", None)
        # Restore originals
        sys.modules.update(saved)


# ── 1. Emotion keyword mapping ───────────────────────────────

class TestEmotionKeywords:
    """Each emotion keyword must produce the documented (emotion, valence)."""

    @pytest.mark.parametrize(
        "keyword, expected_emotion, expected_valence",
        [
            ("\u5b09\u3057\u3044", "happy", 0.7),      # 嬉しい
            ("\u697d\u3057\u3044", "happy", 0.6),      # 楽しい
            ("\u597d\u304d", "loving", 0.7),            # 好き
            ("\u5e78\u305b", "happy", 0.8),             # 幸せ
            ("\u3042\u308a\u304c\u3068\u3046", "happy", 0.5),  # ありがとう
            ("\u7b11", "happy", 0.4),                   # 笑
            ("\u60b2\u3057\u3044", "sad", -0.6),       # 悲しい
            ("\u8f9b\u3044", "sad", -0.5),             # 辛い
            ("\u5bc2\u3057\u3044", "sad", -0.5),       # 寂しい
            ("\u6016\u3044", "scared", -0.6),          # 怖い
            ("\u4e0d\u5b89", "scared", -0.4),          # 不安
            ("\u6012", "angry", -0.5),                 # 怒
            ("\u30e0\u30ab", "angry", -0.5),           # ムカ
            ("\u30a4\u30e9\u30a4\u30e9", "angry", -0.5),  # イライラ
            ("\u9a5a", "surprised", 0.3),              # 驚
            ("\u3073\u3063\u304f\u308a", "surprised", 0.3),  # びっくり
        ],
    )
    def test_emotion_keyword(self, keyword, expected_emotion, expected_valence):
        text = f"\u4eca\u65e5\u306f{keyword}\u306d"  # 今日は{kw}ね
        percept = _heuristic_parse(text)
        assert percept.emotion == expected_emotion
        assert percept.emotion_valence == expected_valence
        assert percept.sentiment == expected_valence

    def test_emotion_keywords_dict_complete(self):
        """Verify the dict has exactly 16 entries as documented."""
        assert len(_EMOTION_KEYWORDS) == 16


# ── 2. Intent keyword mapping ────────────────────────────────

class TestIntentKeywords:
    """Each intent keyword must produce the documented intent label."""

    @pytest.mark.parametrize(
        "keyword, expected_intent",
        [
            ("\u3053\u3093\u306b\u3061\u306f", "greeting"),     # こんにちは
            ("\u304a\u306f\u3088\u3046", "greeting"),           # おはよう
            ("\u3084\u3042", "greeting"),                       # やあ
            ("\uff1f", "question"),                             # ？
            ("?", "question"),
            ("\u6559\u3048\u3066", "question"),                # 教えて
            ("\u4f55", "question"),                             # 何
            ("\u3042\u308a\u304c\u3068\u3046", "greeting"),    # ありがとう
            ("\u30d0\u30a4\u30d0\u30a4", "farewell"),          # バイバイ
            ("\u3055\u3088\u3046\u306a\u3089", "farewell"),    # さようなら
        ],
    )
    def test_intent_keyword(self, keyword, expected_intent):
        text = f"\u30c6\u30b9\u30c8{keyword}\u30c6\u30b9\u30c8"  # テスト{kw}テスト
        percept = _heuristic_parse(text)
        assert percept.intent == expected_intent

    def test_intent_keywords_dict_complete(self):
        """Verify the dict has exactly 10 entries as documented."""
        assert len(_INTENT_KEYWORDS) == 10


# ── 3. Neutral default ───────────────────────────────────────

class TestNeutralDefault:
    """When no keywords match, defaults should be neutral / unknown."""

    def test_no_match_emotion(self):
        percept = _heuristic_parse("\u5929\u6c17\u304c\u3044\u3044\u3067\u3059\u306d")
        assert percept.emotion == "neutral"

    def test_no_match_intent(self):
        percept = _heuristic_parse("\u5929\u6c17\u304c\u3044\u3044\u3067\u3059\u306d")
        assert percept.intent == "unknown"

    def test_no_match_valence(self):
        percept = _heuristic_parse("\u5929\u6c17\u304c\u3044\u3044\u3067\u3059\u306d")
        assert percept.emotion_valence == 0.0

    def test_no_match_sentiment(self):
        percept = _heuristic_parse("\u5929\u6c17\u304c\u3044\u3044\u3067\u3059\u306d")
        assert percept.sentiment == 0.0

    def test_no_match_topics_empty(self):
        percept = _heuristic_parse("\u5929\u6c17\u304c\u3044\u3044\u3067\u3059\u306d")
        assert percept.topics == []

    def test_empty_string(self):
        percept = _heuristic_parse("")
        assert percept.emotion == "neutral"
        assert percept.intent == "unknown"
        assert percept.emotion_valence == 0.0
        assert percept.topics == []


# ── 4. Valence clamping ──────────────────────────────────────

class TestValenceClamping:
    """emotion_valence must always be in [-1.0, 1.0]."""

    def test_positive_valence_within_range(self):
        percept = _heuristic_parse("\u5e78\u305b\u3060")  # 幸せだ
        assert -1.0 <= percept.emotion_valence <= 1.0
        assert percept.emotion_valence == 0.8

    def test_negative_valence_within_range(self):
        percept = _heuristic_parse("\u60b2\u3057\u3044\u65e5")  # 悲しい日
        assert -1.0 <= percept.emotion_valence <= 1.0
        assert percept.emotion_valence == -0.6

    @pytest.mark.parametrize(
        "keyword",
        list(_EMOTION_KEYWORDS.keys()),
    )
    def test_all_emotion_valences_clamped(self, keyword):
        """Every emotion keyword must produce a clamped valence."""
        percept = _heuristic_parse(keyword)
        assert -1.0 <= percept.emotion_valence <= 1.0


# ── 5. Topics populated ──────────────────────────────────────

class TestTopics:
    """Topics list should contain the matched emotion keyword."""

    def test_topic_from_emotion_keyword(self):
        percept = _heuristic_parse("\u5b09\u3057\u3044\u3053\u3068\u304c\u3042\u3063\u305f")  # 嬉しいことがあった
        assert "\u5b09\u3057\u3044" in percept.topics  # 嬉しい

    def test_no_topic_on_neutral(self):
        percept = _heuristic_parse("\u666e\u901a\u306e\u65e5")  # 普通の日
        assert percept.topics == []

    def test_intent_keyword_not_in_topics(self):
        """Intent keywords do not add to topics (only emotion keywords do)."""
        percept = _heuristic_parse("\u3053\u3093\u306b\u3061\u306f\u4e16\u754c")  # こんにちは世界
        assert percept.topics == []


# ── 6. parse_percept without LLM ─────────────────────────────

class TestParsePerceptNoLLM:
    """parse_percept with llm_call_fn=None falls back to heuristic only."""

    @pytest.mark.asyncio
    async def test_heuristic_only_happy(self):
        percept = await parse_percept("\u5b09\u3057\u3044\u306d")  # 嬉しいね
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7

    @pytest.mark.asyncio
    async def test_heuristic_only_neutral(self):
        percept = await parse_percept("\u666e\u901a\u306e\u30c6\u30ad\u30b9\u30c8")  # 普通のテキスト
        assert percept.emotion == "neutral"
        assert percept.intent == "unknown"

    @pytest.mark.asyncio
    async def test_text_preserved(self):
        text = "\u30c6\u30b9\u30c8\u5165\u529b\u30c6\u30ad\u30b9\u30c8"  # テスト入力テキスト
        percept = await parse_percept(text)
        assert percept.text == text
        assert percept.meaning == text

    @pytest.mark.asyncio
    async def test_greeting_intent(self):
        percept = await parse_percept("\u304a\u306f\u3088\u3046\u3054\u3056\u3044\u307e\u3059")  # おはようございます
        assert percept.intent == "greeting"

    @pytest.mark.asyncio
    async def test_farewell_intent(self):
        percept = await parse_percept("\u3055\u3088\u3046\u306a\u3089\u3001\u307e\u305f\u660e\u65e5")  # さようなら、また明日
        assert percept.intent == "farewell"


# ── 7. parse_percept with mock LLM ───────────────────────────

class TestParsePerceptWithLLM:
    """parse_percept with an LLM function returns the enriched result."""

    @pytest.mark.asyncio
    async def test_llm_enriched_result(self):
        llm_response = json.dumps({
            "meaning": "\u6328\u62f6\u3057\u3066\u3044\u308b",  # 挨拶している
            "emotion": "happy",
            "intent": "greeting",
            "emotion_valence": 0.9,
            "topics": ["\u6328\u62f6"],  # 挨拶
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)

        with _hide_llm_wrapper():
            percept = await parse_percept("\u3053\u3093\u306b\u3061\u306f", llm_call_fn=mock_fn)

        assert percept.emotion == "happy"
        assert percept.intent == "greeting"
        assert percept.meaning == "\u6328\u62f6\u3057\u3066\u3044\u308b"
        assert percept.emotion_valence == 0.9
        assert "\u6328\u62f6" in percept.topics

    @pytest.mark.asyncio
    async def test_llm_overrides_heuristic(self):
        """LLM result takes precedence over heuristic baseline."""
        llm_response = json.dumps({
            "meaning": "\u5b9f\u306f\u8907\u96d1\u306a\u611f\u60c5",  # 実は複雑な感情
            "emotion": "loving",
            "intent": "sharing",
            "emotion_valence": 0.6,
            "topics": ["\u611f\u60c5", "\u5171\u6709"],  # 感情, 共有
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)

        with _hide_llm_wrapper():
            percept = await parse_percept("\u5b09\u3057\u3044\u306d", llm_call_fn=mock_fn)

        # LLM says "loving" not "happy" from heuristic
        assert percept.emotion == "loving"
        assert percept.intent == "sharing"

    @pytest.mark.asyncio
    async def test_llm_fn_called_when_provided(self):
        llm_response = json.dumps({
            "meaning": "\u30c6\u30b9\u30c8",
            "emotion": "neutral",
            "intent": "unknown",
            "emotion_valence": 0.0,
            "topics": [],
        })
        mock_fn = _make_llm_fn(llm_response)

        with _hide_llm_wrapper():
            await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=mock_fn)

        mock_fn.assert_called_once()


# ── 8. LLM enrichment fallback on bad JSON ───────────────────

class TestLLMFallbackBadJSON:
    """When LLM returns invalid JSON, fall back to baseline heuristic."""

    @pytest.mark.asyncio
    async def test_garbage_response(self):
        mock_fn = _make_llm_fn("\u3053\u308c\u306fJSON\u3067\u306f\u3042\u308a\u307e\u305b\u3093")
        with _hide_llm_wrapper():
            percept = await parse_percept("\u5b09\u3057\u3044\u306d", llm_call_fn=mock_fn)
        # Fallback to heuristic
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7

    @pytest.mark.asyncio
    async def test_truncated_json(self):
        mock_fn = _make_llm_fn('{"meaning": "test", "emotion":')
        with _hide_llm_wrapper():
            percept = await parse_percept("\u60b2\u3057\u3044\u3088", llm_call_fn=mock_fn)
        assert percept.emotion == "sad"
        assert percept.emotion_valence == -0.6

    @pytest.mark.asyncio
    async def test_empty_response(self):
        mock_fn = _make_llm_fn("")
        with _hide_llm_wrapper():
            percept = await parse_percept("\u666e\u901a\u306e\u6587", llm_call_fn=mock_fn)
        assert percept.emotion == "neutral"

    @pytest.mark.asyncio
    async def test_json_array_not_dict(self):
        mock_fn = _make_llm_fn('[{"emotion": "happy"}]')
        with _hide_llm_wrapper():
            percept = await parse_percept("\u5b09\u3057\u3044\u306d", llm_call_fn=mock_fn)
        # Array is not a dict, so falls back
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7


# ── 9. LLM enrichment fallback on missing fields ─────────────

class TestLLMFallbackMissingFields:
    """When LLM JSON is valid but missing required 'emotion' key, fall back."""

    @pytest.mark.asyncio
    async def test_missing_emotion_key(self):
        llm_response = json.dumps({
            "meaning": "\u4f55\u304b",  # 何か
            "intent": "question",
            "topics": ["\u8cea\u554f"],  # 質問
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u4f55\u304b\u6559\u3048\u3066", llm_call_fn=mock_fn)
        # Falls back to heuristic because "emotion" key is missing
        assert percept.intent == "question"  # heuristic picks this up too

    @pytest.mark.asyncio
    async def test_partial_fields_uses_baseline_defaults(self):
        """LLM returns emotion but missing other fields: baseline fills in."""
        llm_response = json.dumps({
            "emotion": "sad",
            "emotion_valence": -0.8,
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u60b2\u3057\u3044\u65e5", llm_call_fn=mock_fn)
        assert percept.emotion == "sad"
        assert percept.emotion_valence == -0.8
        # Missing fields use baseline defaults
        assert percept.text == "\u60b2\u3057\u3044\u65e5"

    @pytest.mark.asyncio
    async def test_emotion_present_intent_missing_uses_baseline(self):
        """LLM gives emotion but no intent: intent falls back to heuristic baseline."""
        llm_response = json.dumps({
            "emotion": "happy",
            "emotion_valence": 0.5,
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept(
                "\u3053\u3093\u306b\u3061\u306f\u5b09\u3057\u3044",  # こんにちは嬉しい
                llm_call_fn=mock_fn,
            )
        assert percept.emotion == "happy"
        # intent should fall back to baseline heuristic value
        assert percept.intent == "greeting"  # baseline from heuristic


# ── 10. Markdown code fences stripped ─────────────────────────

class TestMarkdownCodeFences:
    """LLM sometimes wraps JSON in ```json ... ``` fences; they must be stripped."""

    @pytest.mark.asyncio
    async def test_triple_backtick_json_fence(self):
        inner = json.dumps({
            "meaning": "\u6328\u62f6",  # 挨拶
            "emotion": "happy",
            "intent": "greeting",
            "emotion_valence": 0.5,
            "topics": ["\u6328\u62f6"],
        }, ensure_ascii=False)
        response = f"```json\n{inner}\n```"
        mock_fn = _make_llm_fn(response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u3053\u3093\u306b\u3061\u306f", llm_call_fn=mock_fn)
        assert percept.emotion == "happy"
        assert percept.meaning == "\u6328\u62f6"

    @pytest.mark.asyncio
    async def test_triple_backtick_no_language(self):
        inner = json.dumps({
            "meaning": "\u30c6\u30b9\u30c8",
            "emotion": "neutral",
            "intent": "unknown",
            "emotion_valence": 0.0,
            "topics": [],
        })
        response = f"```\n{inner}\n```"
        mock_fn = _make_llm_fn(response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=mock_fn)
        assert percept.emotion == "neutral"

    @pytest.mark.asyncio
    async def test_code_fence_with_extra_whitespace(self):
        inner = json.dumps({
            "meaning": "\u5206\u6790",  # 分析
            "emotion": "surprised",
            "intent": "sharing",
            "emotion_valence": 0.3,
            "topics": ["\u9a5a\u304d"],  # 驚き
        }, ensure_ascii=False)
        response = f"  ```json\n{inner}\n```  "
        mock_fn = _make_llm_fn(response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u9a5a\u3044\u305f", llm_call_fn=mock_fn)
        assert percept.emotion == "surprised"


# ── 11. State context passed to LLM prompt ───────────────────

class TestStateContext:
    """When state is provided, emotion context is included in the LLM prompt."""

    @pytest.mark.asyncio
    async def test_state_context_in_prompt(self):
        state = _make_state(mood_valence=0.3, joy=0.4)
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "\u30c6\u30b9\u30c8",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=capturing_fn, state=state)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Should contain emotion summary from state
        assert "\u73fe\u5728\u306e\u611f\u60c5\u72b6\u614b" in prompt  # 現在の感情状態
        assert "valence=" in prompt

    @pytest.mark.asyncio
    async def test_no_state_no_context(self):
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "\u30c6\u30b9\u30c8",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=capturing_fn, state=None)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Without state, no emotion context should be present
        assert "\u73fe\u5728\u306e\u611f\u60c5\u72b6\u614b" not in prompt

    @pytest.mark.asyncio
    async def test_state_mood_valence_in_prompt(self):
        state = _make_state(mood_valence=-0.5)
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "\u30c6\u30b9\u30c8",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=capturing_fn, state=state)

        prompt = captured_prompts[0]
        assert "valence=-0.50" in prompt


# ── 12. First match wins ─────────────────────────────────────

class TestFirstMatchWins:
    """When multiple keywords appear, the first match in dict iteration wins."""

    def test_first_emotion_keyword_wins(self):
        """Text containing two emotion keywords: first in dict order wins."""
        # 嬉しい comes before 悲しい in the dict
        percept = _heuristic_parse("\u5b09\u3057\u3044\u3051\u3069\u60b2\u3057\u3044")
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7
        # Only the first matched keyword is in topics
        assert percept.topics == ["\u5b09\u3057\u3044"]

    def test_first_intent_keyword_wins(self):
        """Text containing two intent keywords: first in dict order wins."""
        # こんにちは comes before さようなら in the dict
        percept = _heuristic_parse("\u3053\u3093\u306b\u3061\u306f\u3001\u3055\u3088\u3046\u306a\u3089")
        assert percept.intent == "greeting"

    def test_emotion_and_intent_independent(self):
        """Emotion and intent matching are independent first-match-wins loops."""
        percept = _heuristic_parse("\u60b2\u3057\u3044\u3051\u3069\u3053\u3093\u306b\u3061\u306f")
        assert percept.emotion == "sad"
        assert percept.intent == "greeting"

    def test_only_one_topic_from_first_match(self):
        """Only one emotion keyword is appended to topics (the first match)."""
        percept = _heuristic_parse("\u6016\u3044\u4e0d\u5b89\u306a\u65e5")  # 怖い不安な日
        # 怖い comes before 不安 in the dict
        assert percept.emotion == "scared"
        assert len(percept.topics) == 1
        assert percept.topics[0] == "\u6016\u3044"


# ── Additional edge cases ─────────────────────────────────────

class TestEdgeCases:
    """Extra edge-case tests for robustness."""

    def test_keyword_as_substring(self):
        """Keywords match as substrings of larger words."""
        # 笑 is in 笑顔
        percept = _heuristic_parse("\u7b11\u9854\u304c\u3044\u3044")  # 笑顔がいい
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.4

    def test_text_and_meaning_are_same(self):
        """Heuristic sets meaning = text (no transformation)."""
        text = "\u4f55\u3067\u3082\u3044\u3044"  # 何でもいい
        percept = _heuristic_parse(text)
        assert percept.text == text
        assert percept.meaning == text

    def test_percept_is_pydantic_model(self):
        """Percept should be a Pydantic BaseModel instance."""
        percept = _heuristic_parse("\u30c6\u30b9\u30c8")
        assert isinstance(percept, Percept)

    @pytest.mark.asyncio
    async def test_llm_valence_clamping(self):
        """LLM returning out-of-range valence should be clamped."""
        llm_response = json.dumps({
            "meaning": "\u6975\u7aef",  # 極端
            "emotion": "happy",
            "intent": "sharing",
            "emotion_valence": 2.5,
            "topics": ["\u6975\u7aef"],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=mock_fn)
        assert percept.emotion_valence == 1.0

    @pytest.mark.asyncio
    async def test_llm_negative_valence_clamping(self):
        """LLM returning very negative valence should be clamped to -1.0."""
        llm_response = json.dumps({
            "meaning": "\u6975\u7aef",
            "emotion": "sad",
            "intent": "sharing",
            "emotion_valence": -3.0,
            "topics": [],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("\u30c6\u30b9\u30c8", llm_call_fn=mock_fn)
        assert percept.emotion_valence == -1.0

    def test_arigatou_triggers_both_emotion_and_intent(self):
        """ありがとう is in both _EMOTION_KEYWORDS and _INTENT_KEYWORDS."""
        percept = _heuristic_parse("\u3042\u308a\u304c\u3068\u3046\u306d")  # ありがとうね
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.5
        assert percept.intent == "greeting"
        assert "\u3042\u308a\u304c\u3068\u3046" in percept.topics

    def test_question_mark_fullwidth(self):
        """Full-width question mark triggers question intent."""
        percept = _heuristic_parse("\u672c\u5f53\uff1f")  # 本当？
        assert percept.intent == "question"

    def test_question_mark_halfwidth(self):
        """Half-width question mark triggers question intent."""
        percept = _heuristic_parse("\u672c\u5f53?")  # 本当?
        assert percept.intent == "question"

    @pytest.mark.asyncio
    async def test_llm_enrich_direct_fallback_on_null(self):
        """_llm_enrich returns baseline when LLM returns 'null' (not a dict)."""
        baseline = _heuristic_parse("\u30c6\u30b9\u30c8")

        async def bad_fn(prompt: str) -> str:
            return "null"

        with _hide_llm_wrapper():
            result = await _llm_enrich("\u30c6\u30b9\u30c8", baseline, bad_fn, None)
        # null parses to None which is not dict -> fallback
        assert result.emotion == baseline.emotion

    @pytest.mark.asyncio
    async def test_llm_exception_fallback(self):
        """_llm_enrich falls back on ValueError from LLM fn."""
        baseline = _heuristic_parse("\u30c6\u30b9\u30c8")

        async def error_fn(prompt: str) -> str:
            raise ValueError("LLM service unavailable")

        with _hide_llm_wrapper():
            result = await _llm_enrich("\u30c6\u30b9\u30c8", baseline, error_fn, None)
        assert result.emotion == baseline.emotion
        assert result.intent == baseline.intent
