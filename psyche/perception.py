"""
psyche/perception.py - Input Interpretation (auxiliary LLM).

Parses user text into a ``Percept`` using local heuristics first,
with optional LLM enrichment via ``llm_wrapper``.  Gemini is used
**only** for structured extraction — never for judgment or state changes.

Usage::

    percept = await parse_percept("こんにちは！", llm_fn, state)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Optional

from .state import Percept, PsycheState
from . import coefficient_registry

logger = logging.getLogger(__name__)

# ── Max topics returned in a single Percept ────────────────────
_MAX_TOPICS = 10

# ── Heuristic keyword maps ─────────────────────────────────────

_EMOTION_KEYWORDS: dict[str, tuple[str, float]] = {
    # --- happy ---
    "嬉しい": ("happy", 0.7), "楽しい": ("happy", 0.6), "幸せ": ("happy", 0.8),
    "ありがとう": ("happy", 0.5), "笑": ("happy", 0.4),
    "やったー": ("happy", 0.7), "よかった": ("happy", 0.5),
    "うれしい": ("happy", 0.7), "たのしい": ("happy", 0.6),
    "わーい": ("happy", 0.6), "最高": ("happy", 0.8),
    "ハッピー": ("happy", 0.6), "ウキウキ": ("happy", 0.6),
    "ワクワク": ("happy", 0.6), "るんるん": ("happy", 0.5),
    # --- sad ---
    "悲しい": ("sad", -0.6), "辛い": ("sad", -0.5), "寂しい": ("sad", -0.5),
    "かなしい": ("sad", -0.6), "つらい": ("sad", -0.5), "さみしい": ("sad", -0.5),
    "泣": ("sad", -0.5), "切ない": ("sad", -0.5),
    "しんどい": ("sad", -0.4), "落ち込": ("sad", -0.5),
    "ショック": ("sad", -0.6), "がっかり": ("sad", -0.5),
    # --- angry ---
    "怒": ("angry", -0.5), "ムカ": ("angry", -0.5), "イライラ": ("angry", -0.5),
    "腹立": ("angry", -0.6), "ふざけ": ("angry", -0.4),
    "キレ": ("angry", -0.6), "うざ": ("angry", -0.4),
    "許せ": ("angry", -0.5), "ひどい": ("angry", -0.4),
    # --- surprised ---
    "驚": ("surprised", 0.3), "びっくり": ("surprised", 0.3),
    "まさか": ("surprised", 0.3), "えっ": ("surprised", 0.2),
    "うそ": ("surprised", 0.2), "マジ": ("surprised", 0.2),
    "信じられ": ("surprised", 0.3),
    # --- scared ---
    "怖い": ("scared", -0.6), "不安": ("scared", -0.4),
    "こわい": ("scared", -0.6), "恐ろし": ("scared", -0.7),
    "おそろし": ("scared", -0.7), "ビクビク": ("scared", -0.5),
    "ドキドキ": ("scared", -0.3), "心配": ("scared", -0.4),
    "ゾッと": ("scared", -0.6), "やばい": ("scared", -0.3),
    # --- loving ---
    "好き": ("loving", 0.7),
    "愛してる": ("loving", 0.9), "大好き": ("loving", 0.8),
    "だいすき": ("loving", 0.8), "すき": ("loving", 0.6),
    "たまらない": ("loving", 0.6), "かわいい": ("loving", 0.5),
    "いとおしい": ("loving", 0.7), "キュン": ("loving", 0.5),
    # --- teasing ---
    "からかう": ("teasing", 0.3), "いじわる": ("teasing", 0.2),
    "冗談": ("teasing", 0.3), "ウソウソ": ("teasing", 0.2),
    "なんちゃって": ("teasing", 0.2), "ニヤニヤ": ("teasing", 0.3),
    # --- confused ---
    "困った": ("confused", -0.3), "わからない": ("confused", -0.2),
    "どうしよう": ("confused", -0.3), "迷う": ("confused", -0.2),
    "混乱": ("confused", -0.3), "モヤモヤ": ("confused", -0.3),
    # --- disappointed ---
    "残念": ("disappointed", -0.4), "期待はずれ": ("disappointed", -0.5),
    "つまらない": ("disappointed", -0.3), "退屈": ("bored", -0.2),
    # --- relieved ---
    "安心": ("relieved", 0.4), "ほっと": ("relieved", 0.4),
    "助かった": ("relieved", 0.5),
    # --- nostalgic ---
    "懐かしい": ("nostalgic", 0.3), "なつかしい": ("nostalgic", 0.3),
    # --- embarrassed ---
    "恥ずかし": ("embarrassed", -0.3), "はずかし": ("embarrassed", -0.3),
    "照れ": ("embarrassed", -0.2),
    # --- frustrated ---
    "悔しい": ("frustrated", -0.5), "くやしい": ("frustrated", -0.5),
    "もどかしい": ("frustrated", -0.4),
    # --- anxious ---
    "焦る": ("anxious", -0.4), "あせる": ("anxious", -0.4),
    "そわそわ": ("anxious", -0.3),
    # --- grateful ---
    "感謝": ("grateful", 0.6), "ありがたい": ("grateful", 0.5),
    "おかげ": ("grateful", 0.4),
}

_INTENT_KEYWORDS: dict[str, str] = {
    # --- greeting ---
    "こんにちは": "greeting", "おはよう": "greeting", "やあ": "greeting",
    "こんばんは": "greeting", "おっす": "greeting", "ただいま": "greeting",
    "おかえり": "greeting", "はじめまして": "greeting",
    # --- question ---
    "？": "question", "?": "question",
    "教えて": "question", "何": "question",
    "どう": "question", "なぜ": "question", "いつ": "question",
    "どこ": "question", "誰": "question", "知りたい": "question",
    # --- farewell ---
    "ありがとう": "gratitude", "バイバイ": "farewell", "さようなら": "farewell",
    "おやすみ": "farewell", "じゃあね": "farewell", "またね": "farewell",
    # --- sharing ---
    "聞いて": "sharing", "実は": "sharing", "報告": "sharing",
    "見て": "sharing", "あのね": "sharing",
    # --- request ---
    "お願い": "request", "して": "request", "してほしい": "request",
    "頼む": "request", "頼み": "request",
    # --- consultation ---
    "相談": "consultation", "悩んで": "consultation", "迷って": "consultation",
    "アドバイス": "consultation",
    # --- agreement ---
    "そうだね": "agreement", "わかる": "agreement", "確かに": "agreement",
    "だよね": "agreement", "その通り": "agreement",
    # --- objection ---
    "違う": "objection", "でも": "objection", "しかし": "objection",
    "そうかな": "objection",
    # --- encouragement ---
    "頑張": "encouragement", "がんば": "encouragement", "大丈夫": "encouragement",
    "応援": "encouragement",
    # --- complaint ---
    "文句": "complaint", "不満": "complaint", "愚痴": "complaint",
    # --- joke ---
    "冗談": "joke", "ネタ": "joke", "ウケる": "joke",
    # --- proposal ---
    "提案": "proposal", "どうかな": "proposal", "しない": "proposal",
    # --- confirmation ---
    "確認": "confirmation", "合ってる": "confirmation", "本当": "confirmation",
    # --- gratitude ---
    "感謝": "gratitude", "助かる": "gratitude",
}


# ── Public API ─────────────────────────────────────────────────

async def parse_percept(
    user_text: str,
    llm_call_fn: Optional[Callable[..., Awaitable[str]]] = None,
    state: Optional[PsycheState] = None,
) -> Percept:
    """Parse user text into a Percept.

    1. Apply local heuristics for fast baseline.
    2. If *llm_call_fn* is available, enrich via LLM (auxiliary only).
    3. Merge results.

    Args:
        user_text: Raw user input.
        llm_call_fn: Optional async LLM function for enrichment.
        state: Optional current PsycheState (biases perception).

    Returns:
        Percept with structured interpretation.
    """
    # 1. Local heuristic baseline (state passed for mood-valence bias)
    percept = _heuristic_parse(user_text, state)

    # 2. LLM enrichment (optional, auxiliary)
    if llm_call_fn:
        percept = await _llm_enrich(user_text, percept, llm_call_fn, state)

    return percept


# ── Local heuristic ────────────────────────────────────────────

# Values loaded from coefficient registry
_perception_coeffs = coefficient_registry.get("perception")
_BIAS_BANDWIDTH = _perception_coeffs["bias_bandwidth"]  # Absolute upper bound for perceptual bias
_BIAS_COEFFICIENT = _perception_coeffs["bias_coefficient"]  # Fixed coefficient: valence * coefficient = raw bias


def _compute_valence_bias(state: Optional[PsycheState]) -> float:
    """Compute mood-valence-based bias for emotion_valence scoring.

    Pure function. No state mutation. No accumulation.

    Args:
        state: Current PsycheState. If None, returns 0.0 (safety valve 4).

    Returns:
        Bias amount clamped to [-_BIAS_BANDWIDTH, +_BIAS_BANDWIDTH].
    """
    if state is None:
        return 0.0
    raw_bias = state.mood.valence * _BIAS_COEFFICIENT
    return max(-_BIAS_BANDWIDTH, min(_BIAS_BANDWIDTH, raw_bias))


def _heuristic_parse(text: str, state: Optional[PsycheState] = None) -> Percept:
    """Fast local keyword-based parse.  No LLM.

    Scans all emotion keywords and selects the match with the
    highest absolute valence.  All matched keywords are collected
    as candidate topics.

    When *state* is provided, a small mood-valence-based bias is added
    to the emotion_valence score after keyword matching.  The bias is
    clamped to ``_BIAS_BANDWIDTH`` (safety valve 1) and computed as a
    pure function with no accumulation (safety valve 3).
    """
    emotion = "neutral"
    valence = 0.0
    intent = "unknown"
    topics: list[str] = []

    # Full-scan: collect all matches, select strongest (max |valence|)
    best_abs = 0.0
    for kw, (emo, val) in _EMOTION_KEYWORDS.items():
        if kw in text:
            topics.append(kw)
            if abs(val) > best_abs:
                best_abs = abs(val)
                emotion = emo
                valence = val

    for kw, nt in _INTENT_KEYWORDS.items():
        if kw in text:
            intent = nt
            break

    # Cap topics
    topics = topics[:_MAX_TOPICS]

    # Apply mood-valence bias to emotion_valence (after keyword matching)
    bias = _compute_valence_bias(state)
    biased_valence = valence + bias

    sentiment = biased_valence  # simple mapping

    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent=intent,
        topics=topics,
        sentiment=sentiment,
        emotion_valence=max(-1.0, min(1.0, biased_valence)),
    )


# ── Expanded label sets (shared with LLM prompt) ──────────────

EMOTION_LABELS: tuple[str, ...] = (
    "happy", "sad", "angry", "surprised", "scared", "loving", "teasing", "neutral",
    "confused", "disappointed", "relieved", "nostalgic", "embarrassed",
    "frustrated", "anxious", "grateful", "bored", "proud", "jealous", "lonely",
)

INTENT_LABELS: tuple[str, ...] = (
    "greeting", "question", "sharing", "request", "joke", "complaint",
    "farewell", "unknown", "consultation", "report", "proposal", "confirmation",
    "agreement", "objection", "encouragement", "gratitude", "monologue",
)


# ── LLM enrichment ────────────────────────────────────────────

async def _llm_enrich(
    text: str,
    baseline: Percept,
    llm_call_fn: Callable[..., Awaitable[str]],
    state: Optional[PsycheState],
) -> Percept:
    """Use LLM for richer extraction.  Falls back to baseline on failure."""

    # Build state context with explicit bias-separation instruction
    state_context = ""
    if state:
        state_context = (
            "\n\n【システム内部状態（参考情報）】\n"
            "以下はシステムの内部状態であり、ユーザー発話の解析結果を誘導するためのものではありません。"
            "解析対象はあくまでユーザーの発話テキストです。\n"
            f"感情状態: {state.emotion_summary()}\n"
            f"気分: valence={state.mood.valence:.2f}"
        )

    emotion_choices = "|".join(EMOTION_LABELS)
    intent_choices = "|".join(INTENT_LABELS)

    prompt = (
        f"以下のユーザー発話を解析してJSONで返してください。\n\n"
        f"発話: 「{text}」\n"
        f"{state_context}\n\n"
        f"出力形式 (JSONのみ):\n"
        f'{{"meaning": "発言の意味を1文で要約", '
        f'"emotion": "{emotion_choices}", '
        f'"intent": "{intent_choices}", '
        f'"emotion_valence": float(-1.0〜1.0, 連続的な値を使用し極端な値への偏りを避けること), '
        f'"topics": ["表層トピックと暗黙的に関連する話題を含める（過度な推測は禁止）"]}}'
    )

    try:
        try:
            from src.llm_wrapper import PERCEPTION_SYSTEM_PROMPT, llm_call_with_system
            raw = await llm_call_with_system(PERCEPTION_SYSTEM_PROMPT, prompt)
        except ImportError:
            raw = await llm_call_fn(prompt)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict) or "emotion" not in data:
            return baseline

        # Emotion fallback: empty string or None -> use baseline
        llm_emotion = data.get("emotion")
        if not llm_emotion or not isinstance(llm_emotion, str) or llm_emotion.strip() == "":
            llm_emotion = baseline.emotion

        # Intent fallback: empty string or None -> use baseline
        llm_intent = data.get("intent")
        if not llm_intent or not isinstance(llm_intent, str) or llm_intent.strip() == "":
            llm_intent = baseline.intent

        # Topics: if LLM returns empty, complement with heuristic topics
        llm_topics = data.get("topics")
        if not llm_topics or not isinstance(llm_topics, list) or len(llm_topics) == 0:
            llm_topics = baseline.topics
        # Cap topics to limit
        llm_topics = [t for t in llm_topics if isinstance(t, str)][:_MAX_TOPICS]

        return Percept(
            text=text,
            meaning=data.get("meaning", text),
            emotion=llm_emotion,
            intent=llm_intent,
            topics=llm_topics,
            sentiment=float(data.get("emotion_valence", baseline.sentiment)),
            emotion_valence=max(-1.0, min(1.0, float(data.get("emotion_valence", baseline.emotion_valence)))),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("Perception LLM enrichment failed: %s", e)
        return baseline
