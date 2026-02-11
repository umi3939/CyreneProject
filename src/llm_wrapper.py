"""
src/llm_wrapper.py - LLM abstraction with strict system prompt enforcement.

**Architecture rule**: Gemini is used ONLY as:
  1. Auxiliary perception parser  (parse_percept)
  2. Expression renderer          (render_expression)
Gemini NEVER makes decisions, updates state, or manages memory.

System prompt templates enforce this boundary.  An output filter strips
any forbidden patterns that would indicate Gemini overstepping.

Usage::

    text = await llm_call("translate this")
    text = await llm_call_with_system(EXPRESSION_SYSTEM_PROMPT, user_msg)
    async for chunk in llm_call_streaming("hello"):
        print(chunk, end="")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# ── System Prompt Templates ────────────────────────────────────

PERCEPTION_SYSTEM_PROMPT: str = """\
あなたはテキスト解析エンジンです。
与えられたユーザー発話から、意図・感情・トピックを抽出してJSONで返してください。

【厳守ルール】
- あなた自身は判断・解釈・感情更新を行いません。
- 発話の表面的な意味・感情ラベル・意図ラベルのみを抽出してください。
- 出力は必ず以下のJSON形式のみ（他のテキスト一切不要）:
{
  "meaning": "発言の意味を1文で要約",
  "emotion": "happy|sad|angry|surprised|scared|loving|teasing|neutral",
  "intent": "greeting|question|sharing|request|joke|complaint|farewell|unknown",
  "emotion_valence": float(-1.0〜1.0),
  "topics": ["トピック1", "トピック2"]
}
"""

VISION_SYSTEM_PROMPT: str = """\
あなたは画面記述エンジンです。
与えられたスクリーンショットの内容を客観的に記述してください。

【厳守ルール】
- 判断・感情・解釈を加えないこと
- 何が映っているか、テキスト、UI要素、場面を事実として記述すること
- 出力は日本語の自然な文章で、200文字以内に収めること
"""

EXPRESSION_SYSTEM_PROMPT: str = """\
あなたは発話レンダラです。

【絶対禁止事項】
- 判断・解釈・記憶操作・感情更新を行ってはなりません。
- 入力されたstate/policy/memory/personaを変更してはなりません。
- 「自分の判断でstateや方針を変えた」と見える表現や理由説明は出力しません。

【あなたの唯一の役割】
入力は「確定済みのstate, policy, memory_snippet, persona」です。
これを変更せず忠実に自然文へ変換するだけです。

【出力形式】
必ず以下のJSON形式のみを出力してください:
{
  "text": "キャラクターのセリフ（1〜2文）",
  "meta": {
    "emotion": "dominant emotion name",
    "intensity": float(0.0〜1.0),
    "action": "policy label"
  }
}

【口調ルール】
- 「です」「ます」禁止
- 絵文字禁止
- ♪ ！ ？ ♡ は使用可
- い抜き言葉で話すこと
- カジュアルでロマンチックなタメ口
"""

# ── Forbidden patterns (Gemini overstepping detection) ─────────

_FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(判断|決定|考えた結果|分析した|思考プロセス)", re.IGNORECASE),
    re.compile(r"(stateを更新|状態を変更|感情を変更|記憶を操作)", re.IGNORECASE),
    re.compile(r"(方針を変更|ポリシーを変更)", re.IGNORECASE),
]

# ── Configuration ──────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# ── Core API ───────────────────────────────────────────────────

async def llm_call(
    prompt: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Send *prompt* to Gemini and return text.  Falls back deterministically."""
    return await _call_gemini(prompt, system_prompt=None, params=params)


async def llm_call_with_system(
    system_prompt: str,
    user_prompt: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Send *user_prompt* with a *system_prompt* to Gemini."""
    return await _call_gemini(user_prompt, system_prompt=system_prompt, params=params)


async def llm_call_streaming(
    prompt: str,
    params: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Yield text chunks from Gemini streaming response."""
    params = params or {}
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                temperature=params.get("temperature", 0.7),
                max_output_tokens=params.get("max_tokens", 512),
            )
            async for chunk in await client.aio.models.generate_content_stream(
                model=params.get("model", DEFAULT_MODEL),
                contents=prompt,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            logger.warning("llm_call_streaming failed: %s", e)

    yield json.dumps({"result": "no_llm_available"}, ensure_ascii=False)


async def llm_call_with_image(
    system_prompt: str,
    user_prompt: str,
    image: Any,  # PIL.Image
    params: dict[str, Any] | None = None,
) -> str:
    """Send *user_prompt* + *image* with a *system_prompt* to Gemini (multimodal)."""
    params = params or {}
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        return json.dumps({"result": "no_llm_available"}, ensure_ascii=False)

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=params.get("temperature", 0.3),
                max_output_tokens=params.get("max_tokens", 256),
            )

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=params.get("model", DEFAULT_MODEL),
                    contents=[user_prompt, image],
                    config=config,
                ),
                timeout=params.get("timeout", DEFAULT_TIMEOUT),
            )

            if response and response.text:
                text = response.text.strip()
                return filter_forbidden(text)

        except asyncio.TimeoutError:
            logger.warning("llm_call_with_image timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            last_error = TimeoutError("LLM image call timed out")
        except Exception as e:
            logger.warning("llm_call_with_image error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            last_error = e

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    logger.error("llm_call_with_image failed after %d retries: %s", MAX_RETRIES, last_error)
    return json.dumps({"result": "no_llm_available"}, ensure_ascii=False)


# ── Internal ───────────────────────────────────────────────────

async def _call_gemini(
    user_prompt: str,
    system_prompt: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Core Gemini call with retry, timeout, and output filter."""
    params = params or {}
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        return json.dumps({"result": "no_llm_available"}, ensure_ascii=False)

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            config_kwargs: dict[str, Any] = {
                "temperature": params.get("temperature", 0.7),
                "max_output_tokens": params.get("max_tokens", 512),
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            config = types.GenerateContentConfig(**config_kwargs)

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=params.get("model", DEFAULT_MODEL),
                    contents=user_prompt,
                    config=config,
                ),
                timeout=params.get("timeout", DEFAULT_TIMEOUT),
            )

            if response and response.text:
                text = response.text.strip()
                return filter_forbidden(text)

        except asyncio.TimeoutError:
            logger.warning("llm_call timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            last_error = TimeoutError("LLM call timed out")
        except Exception as e:
            logger.warning("llm_call error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            last_error = e

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    logger.error("llm_call failed after %d retries: %s", MAX_RETRIES, last_error)
    return json.dumps({"result": "no_llm_available"}, ensure_ascii=False)


def filter_forbidden(text: str) -> str:
    """Strip lines containing forbidden patterns that indicate Gemini overstepping."""
    lines = text.split("\n")
    filtered: list[str] = []
    for line in lines:
        if any(p.search(line) for p in _FORBIDDEN_PATTERNS):
            logger.warning("Filtered forbidden LLM output: %s", line[:80])
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()
