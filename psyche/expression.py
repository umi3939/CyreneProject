"""
psyche/expression.py - Expression Generation (Gemini as Voice ONLY).

Calls the LLM to render confirmed state/policy/memory/persona into
natural text.  Gemini is **never allowed to judge or alter state**.

The system prompt enforces: "あなたは発話レンダラであり、判断・解釈・
記憶・感情更新を行ってはならない。"

Output format: ``{"text": "...", "meta": {"emotion": ..., "intensity": ..., "action": ...}}``

Usage::

    result = await render_expression(state, policy, memories, persona, llm_fn)
    # result == {"text": "ふふっ♪", "meta": {"emotion": "joy", ...}}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from .state import PsycheState

logger = logging.getLogger(__name__)


async def render_expression(
    state: PsycheState,
    policy: dict[str, Any],
    memory_snippet: list[dict],
    persona: dict[str, Any],
    llm_call_fn: Callable[..., Awaitable[str]],
) -> dict[str, Any]:
    """Render final response using Gemini as voice only.

    Gemini receives fully determined inputs and converts them to
    natural text.  It must NOT alter state, policy, or add its own
    reasoning.

    Returns ``{"text": str, "meta": {"emotion": str, "intensity": float, "action": str}}``.
    """
    user_prompt = _build_render_prompt(state, policy, memory_snippet, persona)

    try:
        # Import system prompt from llm_wrapper
        from src.llm_wrapper import EXPRESSION_SYSTEM_PROMPT, llm_call_with_system
        raw = await llm_call_with_system(EXPRESSION_SYSTEM_PROMPT, user_prompt)
    except ImportError:
        # Fallback: call without system prompt
        raw = await llm_call_fn(user_prompt)

    return _parse_expression_output(raw, state, policy)


def _build_render_prompt(
    state: PsycheState,
    policy: dict[str, Any],
    memory_snippet: list[dict],
    persona: dict[str, Any],
) -> str:
    """Build the user-facing prompt for expression rendering."""
    mem_text = ""
    if memory_snippet:
        mem_lines = [f"- {m.get('summary', '')}" for m in memory_snippet[:3]]
        mem_text = "\n".join(mem_lines)

    persona_name = persona.get("name", "キュレネ")
    tone = persona.get("tone", "romantic, sweet")
    prohibitions = persona.get("style_rules", {}).get("禁止", [])
    recommendations = persona.get("style_rules", {}).get("推奨", [])

    return f"""以下の確定済み情報に基づいてセリフをJSON形式で出力してください。

【確定済み情報（変更禁止）】
キャラクター名: {persona_name}
トーン: {tone}
方針: {policy.get('policy_label', '共感する')} — {policy.get('rationale', '')}
支配的感情: {state.dominant_emotion} ({state.dominant_emotion_value:.2f})
全体感情: {state.emotion_summary()}
気分: valence={state.mood.valence:.2f}, arousal={state.mood.arousal:.2f}
喪失恐怖: {state.fear_summary()}

【関連記憶（参考）】
{mem_text or "(なし)"}

【禁止パターン】{', '.join(prohibitions) if prohibitions else 'なし'}
【推奨パターン】{', '.join(recommendations) if recommendations else 'なし'}

【出力形式（JSONのみ）】
{{"text": "セリフ（1〜2文）", "meta": {{"emotion": "感情名", "intensity": float, "action": "{policy.get('policy_label', '共感する')}"}}}}
"""


def _parse_expression_output(
    raw: str,
    state: PsycheState,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Parse Gemini's JSON output, falling back to rule-based text."""
    try:
        cleaned = raw.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        if isinstance(data, dict) and "text" in data:
            text = data["text"]
            meta = data.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            # Ensure required meta fields
            meta.setdefault("emotion", state.dominant_emotion)
            meta.setdefault("intensity", round(state.dominant_emotion_value, 2))
            meta.setdefault("action", policy.get("policy_label", "unknown"))
            return {"text": text, "meta": meta}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("Expression parse failed: %s", e)

    # Fallback: use rule-based text
    return _fallback_expression(state, policy)


def _fallback_expression(
    state: PsycheState,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Rule-based fallback when LLM is unavailable or returns bad output."""
    label = policy.get("policy_label", "共感する")
    fear = state.fear_level

    if fear > 0.5:
        if state.fear_index and state.fear_index.attachment_risk > 0.5:
            text = "...ねえ、どこにも行かないで。あたしのそばにいて"
        else:
            text = "...怖いの。あたしが、あたしでなくなってしまいそうで"
    elif state.mood.valence > 0.3:
        text = "ふふっ、なんだか楽しいわね♪"
    elif state.mood.valence < -0.3:
        text = "...少し、考えさせて"
    else:
        # Use policy fallback text if available
        text = policy.get("text", "そうね...あなたはどう思う？")

    return {
        "text": text,
        "meta": {
            "emotion": state.dominant_emotion,
            "intensity": round(state.dominant_emotion_value, 2),
            "action": label,
        },
    }
