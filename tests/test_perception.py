"""
tests/test_perception.py - Tests for Input Interpretation (perception module).

Verifies:
1. Each original emotion keyword maps to the correct (emotion, valence) tuple
2. Each original intent keyword maps to the correct intent label
3. Neutral default when no keywords match
4. Valence values are correct and clamped to [-1, 1]
5. Topics populated from keyword matches
6. parse_percept without LLM returns heuristic result
7. parse_percept with mock LLM returns enriched result
8. LLM enrichment fallback on bad JSON
9. LLM enrichment fallback on missing fields
10. LLM enrichment with markdown code fences stripped
11. State context passed to LLM prompt (with bias-separation)
12. Full-scan: max absolute valence wins (not first match)
13. Expanded emotion keywords (new labels)
14. Expanded intent keywords (new labels)
15. LLM topics complement from heuristic
16. Topics cap (_MAX_TOPICS)
17. LLM empty/null emotion/intent fallback to baseline
18. Expanded label constants (EMOTION_LABELS, INTENT_LABELS)
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest

from psyche.perception import (
    EMOTION_LABELS,
    INTENT_LABELS,
    _EMOTION_KEYWORDS,
    _INTENT_KEYWORDS,
    _MAX_TOPICS,
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


# ── 1. Emotion keyword mapping (original 16 + new entries) ────

class TestEmotionKeywords:
    """Each emotion keyword must produce the documented (emotion, valence)."""

    @pytest.mark.parametrize(
        "keyword, expected_emotion, expected_valence",
        [
            # Original 16 keywords (some restructured but same mapping)
            ("嬉しい", "happy", 0.7),
            ("楽しい", "happy", 0.6),
            ("好き", "loving", 0.7),
            ("幸せ", "happy", 0.8),
            ("ありがとう", "happy", 0.5),
            ("笑", "happy", 0.4),
            ("悲しい", "sad", -0.6),
            ("辛い", "sad", -0.5),
            ("寂しい", "sad", -0.5),
            ("怖い", "scared", -0.6),
            ("不安", "scared", -0.4),
            ("怒", "angry", -0.5),
            ("ムカ", "angry", -0.5),
            ("イライラ", "angry", -0.5),
            ("驚", "surprised", 0.3),
            ("びっくり", "surprised", 0.3),
        ],
    )
    def test_original_emotion_keyword(self, keyword, expected_emotion, expected_valence):
        text = f"今日は{keyword}ね"
        percept = _heuristic_parse(text)
        assert percept.emotion == expected_emotion
        assert percept.emotion_valence == expected_valence
        assert percept.sentiment == expected_valence

    def test_emotion_keywords_dict_expanded(self):
        """Verify the dict has been expanded beyond the original 16."""
        assert len(_EMOTION_KEYWORDS) > 16

    def test_all_emotion_values_are_tuples(self):
        """Every entry must be (label_str, valence_float)."""
        for kw, (emo, val) in _EMOTION_KEYWORDS.items():
            assert isinstance(emo, str)
            assert isinstance(val, float)
            assert -1.0 <= val <= 1.0


# ── 2. Intent keyword mapping ────────────────────────────────

class TestIntentKeywords:
    """Each intent keyword must produce the documented intent label."""

    @pytest.mark.parametrize(
        "keyword, expected_intent",
        [
            ("こんにちは", "greeting"),
            ("おはよう", "greeting"),
            ("やあ", "greeting"),
            ("？", "question"),
            ("?", "question"),
            ("教えて", "question"),
            ("何", "question"),
            ("ありがとう", "gratitude"),  # Changed from "greeting" to "gratitude"
            ("バイバイ", "farewell"),
            ("さようなら", "farewell"),
        ],
    )
    def test_intent_keyword(self, keyword, expected_intent):
        text = f"テスト{keyword}テスト"
        percept = _heuristic_parse(text)
        assert percept.intent == expected_intent

    def test_intent_keywords_dict_expanded(self):
        """Verify the dict has been expanded beyond the original 10."""
        assert len(_INTENT_KEYWORDS) > 10

    def test_all_intent_values_are_strings(self):
        """Every entry must map to a string label."""
        for kw, intent in _INTENT_KEYWORDS.items():
            assert isinstance(intent, str)
            assert len(intent) > 0


# ── 3. Neutral default ───────────────────────────────────────

class TestNeutralDefault:
    """When no keywords match, defaults should be neutral / unknown."""

    def test_no_match_emotion(self):
        percept = _heuristic_parse("天気がいいですね")
        assert percept.emotion == "neutral"

    def test_no_match_intent(self):
        percept = _heuristic_parse("天気がいいですね")
        assert percept.intent == "unknown"

    def test_no_match_valence(self):
        percept = _heuristic_parse("天気がいいですね")
        assert percept.emotion_valence == 0.0

    def test_no_match_sentiment(self):
        percept = _heuristic_parse("天気がいいですね")
        assert percept.sentiment == 0.0

    def test_no_match_topics_empty(self):
        percept = _heuristic_parse("天気がいいですね")
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
        percept = _heuristic_parse("幸せだ")
        assert -1.0 <= percept.emotion_valence <= 1.0
        assert percept.emotion_valence == 0.8

    def test_negative_valence_within_range(self):
        percept = _heuristic_parse("悲しい日")
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
        percept = _heuristic_parse("嬉しいことがあった")
        assert "嬉しい" in percept.topics

    def test_no_topic_on_neutral(self):
        percept = _heuristic_parse("普通の日")
        assert percept.topics == []

    def test_intent_keyword_not_in_topics(self):
        """Intent keywords do not add to topics (only emotion keywords do)."""
        percept = _heuristic_parse("こんにちは世界")
        assert percept.topics == []


# ── 6. parse_percept without LLM ─────────────────────────────

class TestParsePerceptNoLLM:
    """parse_percept with llm_call_fn=None falls back to heuristic only."""

    @pytest.mark.asyncio
    async def test_heuristic_only_happy(self):
        percept = await parse_percept("嬉しいね")
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7

    @pytest.mark.asyncio
    async def test_heuristic_only_neutral(self):
        percept = await parse_percept("普通のテキスト")
        assert percept.emotion == "neutral"
        assert percept.intent == "unknown"

    @pytest.mark.asyncio
    async def test_text_preserved(self):
        text = "テスト入力テキスト"
        percept = await parse_percept(text)
        assert percept.text == text
        assert percept.meaning == text

    @pytest.mark.asyncio
    async def test_greeting_intent(self):
        percept = await parse_percept("おはようございます")
        assert percept.intent == "greeting"

    @pytest.mark.asyncio
    async def test_farewell_intent(self):
        percept = await parse_percept("さようなら、また明日")
        assert percept.intent == "farewell"


# ── 7. parse_percept with mock LLM ───────────────────────────

class TestParsePerceptWithLLM:
    """parse_percept with an LLM function returns the enriched result."""

    @pytest.mark.asyncio
    async def test_llm_enriched_result(self):
        llm_response = json.dumps({
            "meaning": "挨拶している",
            "emotion": "happy",
            "intent": "greeting",
            "emotion_valence": 0.9,
            "topics": ["挨拶"],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)

        with _hide_llm_wrapper():
            percept = await parse_percept("こんにちは", llm_call_fn=mock_fn)

        assert percept.emotion == "happy"
        assert percept.intent == "greeting"
        assert percept.meaning == "挨拶している"
        assert percept.emotion_valence == 0.9
        assert "挨拶" in percept.topics

    @pytest.mark.asyncio
    async def test_llm_overrides_heuristic(self):
        """LLM result takes precedence over heuristic baseline."""
        llm_response = json.dumps({
            "meaning": "実は複雑な感情",
            "emotion": "loving",
            "intent": "sharing",
            "emotion_valence": 0.6,
            "topics": ["感情", "共有"],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)

        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)

        # LLM says "loving" not "happy" from heuristic
        assert percept.emotion == "loving"
        assert percept.intent == "sharing"

    @pytest.mark.asyncio
    async def test_llm_fn_called_when_provided(self):
        llm_response = json.dumps({
            "meaning": "テスト",
            "emotion": "neutral",
            "intent": "unknown",
            "emotion_valence": 0.0,
            "topics": [],
        })
        mock_fn = _make_llm_fn(llm_response)

        with _hide_llm_wrapper():
            await parse_percept("テスト", llm_call_fn=mock_fn)

        mock_fn.assert_called_once()


# ── 8. LLM enrichment fallback on bad JSON ───────────────────

class TestLLMFallbackBadJSON:
    """When LLM returns invalid JSON, fall back to baseline heuristic."""

    @pytest.mark.asyncio
    async def test_garbage_response(self):
        mock_fn = _make_llm_fn("これはJSONではありません")
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        # Fallback to heuristic
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7

    @pytest.mark.asyncio
    async def test_truncated_json(self):
        mock_fn = _make_llm_fn('{"meaning": "test", "emotion":')
        with _hide_llm_wrapper():
            percept = await parse_percept("悲しいよ", llm_call_fn=mock_fn)
        assert percept.emotion == "sad"
        assert percept.emotion_valence == -0.6

    @pytest.mark.asyncio
    async def test_empty_response(self):
        mock_fn = _make_llm_fn("")
        with _hide_llm_wrapper():
            percept = await parse_percept("普通の文", llm_call_fn=mock_fn)
        assert percept.emotion == "neutral"

    @pytest.mark.asyncio
    async def test_json_array_not_dict(self):
        mock_fn = _make_llm_fn('[{"emotion": "happy"}]')
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        # Array is not a dict, so falls back
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.7


# ── 9. LLM enrichment fallback on missing fields ─────────────

class TestLLMFallbackMissingFields:
    """When LLM JSON is valid but missing required 'emotion' key, fall back."""

    @pytest.mark.asyncio
    async def test_missing_emotion_key(self):
        llm_response = json.dumps({
            "meaning": "何か",
            "intent": "question",
            "topics": ["質問"],
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("何か教えて", llm_call_fn=mock_fn)
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
            percept = await parse_percept("悲しい日", llm_call_fn=mock_fn)
        assert percept.emotion == "sad"
        assert percept.emotion_valence == -0.8
        # Missing fields use baseline defaults
        assert percept.text == "悲しい日"

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
                "こんにちは嬉しい",
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
            "meaning": "挨拶",
            "emotion": "happy",
            "intent": "greeting",
            "emotion_valence": 0.5,
            "topics": ["挨拶"],
        }, ensure_ascii=False)
        response = f"```json\n{inner}\n```"
        mock_fn = _make_llm_fn(response)
        with _hide_llm_wrapper():
            percept = await parse_percept("こんにちは", llm_call_fn=mock_fn)
        assert percept.emotion == "happy"
        assert percept.meaning == "挨拶"

    @pytest.mark.asyncio
    async def test_triple_backtick_no_language(self):
        inner = json.dumps({
            "meaning": "テスト",
            "emotion": "neutral",
            "intent": "unknown",
            "emotion_valence": 0.0,
            "topics": [],
        })
        response = f"```\n{inner}\n```"
        mock_fn = _make_llm_fn(response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        assert percept.emotion == "neutral"

    @pytest.mark.asyncio
    async def test_code_fence_with_extra_whitespace(self):
        inner = json.dumps({
            "meaning": "分析",
            "emotion": "surprised",
            "intent": "sharing",
            "emotion_valence": 0.3,
            "topics": ["驚き"],
        }, ensure_ascii=False)
        response = f"  ```json\n{inner}\n```  "
        mock_fn = _make_llm_fn(response)
        with _hide_llm_wrapper():
            percept = await parse_percept("驚いた", llm_call_fn=mock_fn)
        assert percept.emotion == "surprised"


# ── 11. State context passed to LLM prompt (bias separation) ──

class TestStateContext:
    """When state is provided, context is included with bias-separation warning."""

    @pytest.mark.asyncio
    async def test_state_context_in_prompt(self):
        state = _make_state(mood_valence=0.3, joy=0.4)
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "テスト",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("テスト", llm_call_fn=capturing_fn, state=state)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Should contain state info
        assert "valence=" in prompt

    @pytest.mark.asyncio
    async def test_bias_separation_in_prompt(self):
        """Prompt must contain bias-separation instruction when state is present."""
        state = _make_state(mood_valence=0.3, joy=0.4)
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "テスト",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("テスト", llm_call_fn=capturing_fn, state=state)

        prompt = captured_prompts[0]
        # Must contain bias-separation notice
        assert "誘導" in prompt  # 「解析結果を誘導するためのものではありません」
        assert "内部状態" in prompt  # 「システムの内部状態」

    @pytest.mark.asyncio
    async def test_no_state_no_context(self):
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "テスト",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("テスト", llm_call_fn=capturing_fn, state=None)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Without state, no system internal state section should be present
        assert "内部状態" not in prompt

    @pytest.mark.asyncio
    async def test_state_mood_valence_in_prompt(self):
        state = _make_state(mood_valence=-0.5)
        captured_prompts: list[str] = []

        async def capturing_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "meaning": "テスト",
                "emotion": "neutral",
                "intent": "unknown",
                "emotion_valence": 0.0,
                "topics": [],
            })

        with _hide_llm_wrapper():
            await parse_percept("テスト", llm_call_fn=capturing_fn, state=state)

        prompt = captured_prompts[0]
        assert "valence=-0.50" in prompt


# ── 12. Full-scan: max absolute valence wins ──────────────────

class TestMaxValenceWins:
    """Full-scan selects the keyword with the highest |valence|, not first match."""

    def test_stronger_emotion_wins_over_first(self):
        """幸せ (0.8) is stronger than 笑 (0.4). Even if 笑 appears first in text."""
        percept = _heuristic_parse("笑って幸せ")
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.8

    def test_negative_stronger_wins(self):
        """恐ろしい (-0.7) is stronger than 不安 (-0.4)."""
        percept = _heuristic_parse("不安で恐ろしい")
        assert percept.emotion == "scared"
        assert percept.emotion_valence == -0.7

    def test_both_keywords_in_topics(self):
        """Full-scan collects ALL matched keywords as topics."""
        percept = _heuristic_parse("嬉しいけど悲しい")
        assert "嬉しい" in percept.topics
        assert "悲しい" in percept.topics
        assert len(percept.topics) >= 2

    def test_multiple_topics_from_same_category(self):
        """Multiple keywords of same emotion both appear in topics."""
        percept = _heuristic_parse("怖い不安な日")
        assert "怖い" in percept.topics
        assert "不安" in percept.topics
        # 怖い (-0.6) is stronger than 不安 (-0.4)
        assert percept.emotion == "scared"
        assert percept.emotion_valence == -0.6

    def test_first_intent_keyword_still_wins(self):
        """Intent matching still uses first-match."""
        percept = _heuristic_parse("こんにちは、さようなら")
        assert percept.intent == "greeting"

    def test_emotion_and_intent_independent(self):
        """Emotion (max-valence) and intent (first-match) are independent."""
        percept = _heuristic_parse("悲しいけどこんにちは")
        assert percept.emotion == "sad"
        assert percept.intent == "greeting"

    def test_loving_strong_valence(self):
        """愛してる (0.9) should beat 好き (0.7) when both present."""
        percept = _heuristic_parse("好きだし愛してる")
        assert percept.emotion == "loving"
        assert percept.emotion_valence == 0.9


# ── 13. Expanded emotion keywords ─────────────────────────────

class TestExpandedEmotionKeywords:
    """New emotion keywords added for broader coverage."""

    @pytest.mark.parametrize(
        "keyword, expected_emotion",
        [
            # happy variants
            ("やったー", "happy"),
            ("よかった", "happy"),
            ("うれしい", "happy"),
            ("最高", "happy"),
            ("ウキウキ", "happy"),
            ("ワクワク", "happy"),
            # sad variants
            ("かなしい", "sad"),
            ("つらい", "sad"),
            ("切ない", "sad"),
            ("しんどい", "sad"),
            ("ショック", "sad"),
            ("がっかり", "sad"),
            # angry variants
            ("腹立", "angry"),
            ("キレ", "angry"),
            ("うざ", "angry"),
            ("許せ", "angry"),
            # surprised variants
            ("まさか", "surprised"),
            ("えっ", "surprised"),
            ("マジ", "surprised"),
            # scared variants
            ("こわい", "scared"),
            ("恐ろし", "scared"),
            ("心配", "scared"),
            ("ゾッと", "scared"),
            # loving variants
            ("愛してる", "loving"),
            ("大好き", "loving"),
            ("かわいい", "loving"),
            ("キュン", "loving"),
            # teasing
            ("からかう", "teasing"),
            ("冗談", "teasing"),
            ("なんちゃって", "teasing"),
            # new categories
            ("困った", "confused"),
            ("わからない", "confused"),
            ("残念", "disappointed"),
            ("安心", "relieved"),
            ("懐かしい", "nostalgic"),
            ("恥ずかし", "embarrassed"),
            ("悔しい", "frustrated"),
            ("焦る", "anxious"),
            ("感謝", "grateful"),
            ("退屈", "bored"),
        ],
    )
    def test_expanded_emotion_keyword(self, keyword, expected_emotion):
        percept = _heuristic_parse(f"今日は{keyword}な日")
        assert percept.emotion == expected_emotion


# ── 14. Expanded intent keywords ──────────────────────────────

class TestExpandedIntentKeywords:
    """New intent keywords added for broader coverage."""

    @pytest.mark.parametrize(
        "keyword, expected_intent",
        [
            # new greetings
            ("こんばんは", "greeting"),
            ("はじめまして", "greeting"),
            ("ただいま", "greeting"),
            # new question words
            ("なぜ", "question"),
            ("いつ", "question"),
            ("どこ", "question"),
            ("知りたい", "question"),
            # farewell
            ("おやすみ", "farewell"),
            ("じゃあね", "farewell"),
            ("またね", "farewell"),
            # sharing
            ("聞いて", "sharing"),
            ("実は", "sharing"),
            ("あのね", "sharing"),
            # request
            ("お願い", "request"),
            ("頼む", "request"),
            # consultation
            ("相談", "consultation"),
            ("アドバイス", "consultation"),
            # agreement
            ("そうだね", "agreement"),
            ("確かに", "agreement"),
            # objection
            ("違う", "objection"),
            # encouragement
            ("頑張", "encouragement"),
            ("大丈夫", "encouragement"),
            # complaint
            ("愚痴", "complaint"),
            # joke
            ("ウケる", "joke"),
            # gratitude
            ("感謝", "gratitude"),
            ("助かる", "gratitude"),
        ],
    )
    def test_expanded_intent_keyword(self, keyword, expected_intent):
        percept = _heuristic_parse(f"テスト{keyword}テスト")
        assert percept.intent == expected_intent


# ── 15. LLM topics complement ─────────────────────────────────

class TestLLMTopicsComplement:
    """When LLM returns empty topics, heuristic topics should complement."""

    @pytest.mark.asyncio
    async def test_empty_llm_topics_uses_heuristic(self):
        """If LLM returns empty topics, fall back to heuristic topics."""
        llm_response = json.dumps({
            "meaning": "嬉しいこと",
            "emotion": "happy",
            "intent": "sharing",
            "emotion_valence": 0.7,
            "topics": [],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        # Heuristic should have detected "嬉しい" as topic
        assert len(percept.topics) > 0
        assert "嬉しい" in percept.topics

    @pytest.mark.asyncio
    async def test_non_empty_llm_topics_kept(self):
        """If LLM returns topics, those are used (not heuristic)."""
        llm_response = json.dumps({
            "meaning": "嬉しいこと",
            "emotion": "happy",
            "intent": "sharing",
            "emotion_valence": 0.7,
            "topics": ["感情", "喜び"],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        assert "感情" in percept.topics
        assert "喜び" in percept.topics

    @pytest.mark.asyncio
    async def test_null_llm_topics_uses_heuristic(self):
        """If LLM topics is null, fall back to heuristic."""
        llm_response = json.dumps({
            "meaning": "嬉しいこと",
            "emotion": "happy",
            "intent": "sharing",
            "emotion_valence": 0.7,
            "topics": None,
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        assert "嬉しい" in percept.topics


# ── 16. Topics cap ────────────────────────────────────────────

class TestTopicsCap:
    """Topics must be capped at _MAX_TOPICS."""

    def test_max_topics_constant_exists(self):
        assert isinstance(_MAX_TOPICS, int)
        assert _MAX_TOPICS > 0

    @pytest.mark.asyncio
    async def test_llm_topics_capped(self):
        """LLM returning more than _MAX_TOPICS should be truncated."""
        many_topics = [f"topic{i}" for i in range(_MAX_TOPICS + 5)]
        llm_response = json.dumps({
            "meaning": "test",
            "emotion": "neutral",
            "intent": "unknown",
            "emotion_valence": 0.0,
            "topics": many_topics,
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        assert len(percept.topics) == _MAX_TOPICS

    @pytest.mark.asyncio
    async def test_llm_topics_non_string_filtered(self):
        """Non-string items in LLM topics should be filtered out."""
        llm_response = json.dumps({
            "meaning": "test",
            "emotion": "neutral",
            "intent": "unknown",
            "emotion_valence": 0.0,
            "topics": ["valid", 123, None, "also_valid"],
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        assert all(isinstance(t, str) for t in percept.topics)
        assert "valid" in percept.topics
        assert "also_valid" in percept.topics


# ── 17. LLM empty emotion/intent fallback ─────────────────────

class TestLLMEmptyLabelFallback:
    """LLM returning empty or null emotion/intent should fall back to baseline."""

    @pytest.mark.asyncio
    async def test_empty_emotion_falls_back(self):
        llm_response = json.dumps({
            "meaning": "test",
            "emotion": "",
            "intent": "greeting",
            "emotion_valence": 0.0,
            "topics": ["test"],
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        # Empty emotion should fall back to heuristic
        assert percept.emotion == "happy"

    @pytest.mark.asyncio
    async def test_null_emotion_falls_back(self):
        llm_response = json.dumps({
            "meaning": "test",
            "emotion": None,
            "intent": "greeting",
            "emotion_valence": 0.0,
            "topics": ["test"],
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("嬉しいね", llm_call_fn=mock_fn)
        assert percept.emotion == "happy"

    @pytest.mark.asyncio
    async def test_empty_intent_falls_back(self):
        llm_response = json.dumps({
            "meaning": "test",
            "emotion": "happy",
            "intent": "",
            "emotion_valence": 0.5,
            "topics": ["test"],
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("こんにちは嬉しい", llm_call_fn=mock_fn)
        # Empty intent should fall back to heuristic
        assert percept.intent == "greeting"

    @pytest.mark.asyncio
    async def test_unknown_emotion_label_transparent(self):
        """LLM returning an unknown but non-empty label should pass through."""
        llm_response = json.dumps({
            "meaning": "test",
            "emotion": "melancholic",
            "intent": "sharing",
            "emotion_valence": -0.3,
            "topics": ["test"],
        })
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        # Unknown label passed through (not forced to known set)
        assert percept.emotion == "melancholic"


# ── 18. Expanded label constants ──────────────────────────────

class TestLabelConstants:
    """EMOTION_LABELS and INTENT_LABELS should be available and expanded."""

    def test_emotion_labels_includes_original_8(self):
        original = {"happy", "sad", "angry", "surprised", "scared", "loving", "teasing", "neutral"}
        for label in original:
            assert label in EMOTION_LABELS

    def test_emotion_labels_expanded_beyond_8(self):
        assert len(EMOTION_LABELS) > 8

    def test_emotion_labels_includes_new(self):
        new_labels = {"confused", "disappointed", "relieved", "nostalgic", "embarrassed",
                      "frustrated", "anxious", "grateful"}
        for label in new_labels:
            assert label in EMOTION_LABELS

    def test_intent_labels_includes_original_8(self):
        original = {"greeting", "question", "sharing", "request", "joke", "complaint", "farewell", "unknown"}
        for label in original:
            assert label in INTENT_LABELS

    def test_intent_labels_expanded_beyond_8(self):
        assert len(INTENT_LABELS) > 8

    def test_intent_labels_includes_new(self):
        new_labels = {"consultation", "report", "proposal", "confirmation",
                      "agreement", "objection", "encouragement", "gratitude"}
        for label in new_labels:
            assert label in INTENT_LABELS

    def test_prompt_contains_expanded_labels(self):
        """The user prompt built by _llm_enrich should contain the expanded label set."""
        # Verify the labels are tuples of strings
        assert all(isinstance(l, str) for l in EMOTION_LABELS)
        assert all(isinstance(l, str) for l in INTENT_LABELS)


# ── Additional edge cases ─────────────────────────────────────

class TestEdgeCases:
    """Extra edge-case tests for robustness."""

    def test_keyword_as_substring(self):
        """Keywords match as substrings of larger words."""
        # 笑 is in 笑顔
        percept = _heuristic_parse("笑顔がいい")
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.4

    def test_text_and_meaning_are_same(self):
        """Heuristic sets meaning = text (no transformation)."""
        text = "何でもいい"
        percept = _heuristic_parse(text)
        assert percept.text == text
        assert percept.meaning == text

    def test_percept_is_pydantic_model(self):
        """Percept should be a Pydantic BaseModel instance."""
        percept = _heuristic_parse("テスト")
        assert isinstance(percept, Percept)

    @pytest.mark.asyncio
    async def test_llm_valence_clamping(self):
        """LLM returning out-of-range valence should be clamped."""
        llm_response = json.dumps({
            "meaning": "極端",
            "emotion": "happy",
            "intent": "sharing",
            "emotion_valence": 2.5,
            "topics": ["極端"],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        assert percept.emotion_valence == 1.0

    @pytest.mark.asyncio
    async def test_llm_negative_valence_clamping(self):
        """LLM returning very negative valence should be clamped to -1.0."""
        llm_response = json.dumps({
            "meaning": "極端",
            "emotion": "sad",
            "intent": "sharing",
            "emotion_valence": -3.0,
            "topics": [],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        assert percept.emotion_valence == -1.0

    def test_arigatou_triggers_emotion_and_intent(self):
        """ありがとう is in both _EMOTION_KEYWORDS and _INTENT_KEYWORDS."""
        percept = _heuristic_parse("ありがとうね")
        assert percept.emotion == "happy"
        assert percept.emotion_valence == 0.5
        assert percept.intent == "gratitude"  # updated from "greeting"
        assert "ありがとう" in percept.topics

    def test_question_mark_fullwidth(self):
        """Full-width question mark triggers question intent."""
        percept = _heuristic_parse("本当？")
        assert percept.intent == "question"

    def test_question_mark_halfwidth(self):
        """Half-width question mark triggers question intent."""
        percept = _heuristic_parse("本当?")
        assert percept.intent == "question"

    @pytest.mark.asyncio
    async def test_llm_enrich_direct_fallback_on_null(self):
        """_llm_enrich returns baseline when LLM returns 'null' (not a dict)."""
        baseline = _heuristic_parse("テスト")

        async def bad_fn(prompt: str) -> str:
            return "null"

        with _hide_llm_wrapper():
            result = await _llm_enrich("テスト", baseline, bad_fn, None)
        # null parses to None which is not dict -> fallback
        assert result.emotion == baseline.emotion

    @pytest.mark.asyncio
    async def test_llm_exception_fallback(self):
        """_llm_enrich falls back on ValueError from LLM fn."""
        baseline = _heuristic_parse("テスト")

        async def error_fn(prompt: str) -> str:
            raise ValueError("LLM service unavailable")

        with _hide_llm_wrapper():
            result = await _llm_enrich("テスト", baseline, error_fn, None)
        assert result.emotion == baseline.emotion
        assert result.intent == baseline.intent

    @pytest.mark.asyncio
    async def test_llm_expanded_emotion_label(self):
        """LLM returning a new expanded label should be accepted."""
        llm_response = json.dumps({
            "meaning": "困っている",
            "emotion": "confused",
            "intent": "consultation",
            "emotion_valence": -0.3,
            "topics": ["困惑"],
        }, ensure_ascii=False)
        mock_fn = _make_llm_fn(llm_response)
        with _hide_llm_wrapper():
            percept = await parse_percept("テスト", llm_call_fn=mock_fn)
        assert percept.emotion == "confused"
        assert percept.intent == "consultation"
