"""
brain.py - Thinking engine for Cyrene

Uses Google Gemini 3 Flash Preview for screen analysis and
response generation with the Cyrene persona from identity.md.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

from src.memory_manager import MemoryManager
from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept
from psyche.memory_link import recall_with_mood
from psyche.perception import parse_percept
from psyche.expression import render_expression
from psyche.silence_hesitation import is_silence_policy
from src.llm_wrapper import llm_call, llm_call_with_image, VISION_SYSTEM_PROMPT

from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class CyreneBrain:
    """
    AI thinking engine using Gemini 3 Flash Preview.
    Generates responses based on screen content using the persona from identity.md.
    """

    def __init__(self):
        """
        Initialize the brain with Gemini API and Cyrene persona.

        Raises:
            ValueError: If GEMINI_API_KEY is not set.
            FileNotFoundError: If identity.md is not found.
        """
        # Load environment variables
        load_dotenv()

        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Initialize Gemini client
        self._client = genai.Client(api_key=api_key)
        self._model_name = "gemini-3-flash-preview"
        logger.info("Gemini client initialized")

        # Load persona from identity.md
        self._persona = self._load_identity()

        # Generation config (system instruction, temperature, etc.)
        # Gemini 2.5+ defaults safety filters to OFF, so no safety_settings needed.
        self._config = types.GenerateContentConfig(
            system_instruction=self._persona,
            temperature=1.2,  # High creativity for entertaining reactions
            max_output_tokens=4096,  # Thinking + output tokens combined
        )

        # Summary generation config (separate from chat config)
        self._summary_config = types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=2048,
        )

        # Long-term memory (with embedding support)
        self._memory = MemoryManager()
        self._turn_count = 0

        # Self-managed conversation log (replaces SDK-internal _curated_history)
        self._conversation_log: list[str] = []
        self._last_response: str = ""

        # Create chat session for short-term memory (context retention)
        self._chat = self._create_chat()
        logger.info(f"Gemini model ready ({self._model_name})")

        # Psyche (psychological state tracker) — all modules via orchestrator
        self._last_psyche_update = time.monotonic()
        self._orchestrator = PsycheOrchestrator(
            memory_count=len(self._memory._memories),
        )

        # Restore psyche state from previous session
        self._orchestrator.load()

        # 2-call structure support
        self._persona_dict = self._build_persona_dict()
        self._last_emotion: str = "neutral"
        self._perception_config: dict = {
            "temperature": 0.3,
            "max_tokens": 256,
        }

    async def _embed_text(self, text: str) -> list[float]:
        """
        Embed text using Gemini's text-embedding-004 model.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        result = await self._client.aio.models.embed_content(
            model="text-embedding-004", contents=text
        )
        return result.embeddings[0].values

    def _create_chat(self):
        """Create a new chat session with Gemini."""
        return self._client.aio.chats.create(
            model=self._model_name,
            config=self._config,
        )

    def reset_memory(self):
        """Reset conversation history and start a fresh chat session."""
        self._chat = self._create_chat()
        self._conversation_log.clear()
        self._last_response = ""
        logger.info("Memory reset - new chat session started")

    async def summarize_and_save(self):
        """
        Summarize recent conversation history and save as a long-term memory.
        Uses Gemini (single-shot, outside the chat session) to generate a summary.
        Uses self-managed conversation log instead of SDK internals.
        """
        try:
            if len(self._conversation_log) < 2:
                logger.debug("Not enough conversation to summarize")
                return

            # Use last 10 entries from our own log
            conversation_text = "\n".join(self._conversation_log[-10:])

            prompt = (
                "以下の会話を要約して、次回の会話で思い出すのに役立つ情報をJSON形式で返して。\n"
                "JSONのみ出力し、他のテキストは含めないこと。\n"
                '{"summary": "...", "keywords": ["...", "..."], "importance": 1-5}\n\n'
                f"会話ログ:\n{conversation_text}"
            )

            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._summary_config,
            )

            if not response or not response.text:
                logger.warning("Empty summary response from Gemini")
                return

            # Parse JSON from response
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)

            summary = data.get("summary", "")
            keywords = data.get("keywords", [])
            importance = int(data.get("importance", 3))

            if summary:
                self._memory.maybe_save(
                    summary, "", {},
                    importance=importance,
                )
                logger.debug(f"Memory saved: {summary}")

                # Notify orchestrator of memory save
                self._orchestrator.on_memory_saved(
                    summary=summary,
                    keywords=keywords,
                    memory_count=len(self._memory._memories),
                )
            else:
                logger.warning("Empty summary from Gemini parse")

            # Clear log after successful save
            self._conversation_log.clear()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse summary JSON: {e}")
        except Exception as e:
            logger.error(f"summarize_and_save failed: {e}")

    def _load_identity(self) -> str:
        """
        Load the Cyrene persona from identity.md.

        Returns:
            Content of identity.md as string.

        Raises:
            FileNotFoundError: If identity.md is not found.
        """
        identity_path = Path(__file__).parent / "identity.md"

        if not identity_path.exists():
            raise FileNotFoundError(f"identity.md not found at {identity_path}")

        persona_text = identity_path.read_text(encoding="utf-8")
        logger.info(f"Loaded persona from {identity_path}")
        return persona_text

    @property
    def last_emotion(self) -> str:
        return self._last_emotion

    def save_state(self) -> None:
        """Save orchestrator psyche state for next session."""
        self._orchestrator.save()

    def _build_persona_dict(self) -> dict:
        return {
            "name": "キュレネ",
            "tone": "romantic, sweet, playful",
            "style_rules": {
                "禁止": ["敬語", "絵文字", "行動描写", "説明的な回答"],
                "推奨": ["♪♡使用可", "い抜き言葉", "カジュアルなタメ口", "ロマンチック"],
            },
        }

    async def think(
        self,
        image_path: str,
        vision_summary: str = ""
    ) -> Optional[str]:
        """
        2-call structure (non-streaming version).
        Returns None if psyche decides silence.

        Args:
            image_path: Path to JPEG image file.
            vision_summary: Formatted sensor data from HybridEye (YOLO + OCR).

        Returns:
            Generated text in Cyrene's voice, or None for silence.
        """
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_file)

            # Phase 1: Perception
            perception_prompt = "この画面の内容を客観的に記述してください。"
            if vision_summary:
                perception_prompt += f"\n\n参考センサー情報:\n{vision_summary}"

            screen_description = await llm_call_with_image(
                VISION_SYSTEM_PROMPT,
                perception_prompt,
                image,
                self._perception_config,
            )

            # Phase 2: Parse (with LLM enrichment)
            percept = await parse_percept(
                screen_description,
                llm_call_fn=llm_call,
                state=self._orchestrator.psyche,
            )

            # Phase 3: Psyche update
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "viewer")

            # Phase 4: Recall
            recall_query = screen_description
            if self._last_response:
                recall_query += " " + self._last_response
            recall_percept = Percept(text=recall_query)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )
            self._orchestrator.set_recalled_memories(memories)

            # Phase 5: Policy
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "viewer"
            )

            # Phase 6: Silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence")
                return None

            # Phase 7: Expression (with psyche enrichment)
            self._last_emotion = percept.emotion or "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("viewer")
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=screen_description,
                recent_history=self._conversation_log[-5:],
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return None

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._last_response = full_text
            return full_text

        except Exception as e:
            logger.error(f"Think failed: {e}")
            return "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"

    async def think_streaming(
        self,
        image_path: str,
        vision_summary: str = ""
    ) -> AsyncGenerator[str, None]:
        """
        2-call structure: perception → psyche → expression.

        Phase 1: Gemini vision call (image → screen description text)
        Phase 2: parse_percept (text → structured Percept)
        Phase 3: orchestrator.post_response_update (psyche full pipeline)
        Phase 4: recall_with_mood (memory search)
        Phase 5: select_policy_dict (policy selection)
        Phase 6: is_silence_policy check → silence → return
        Phase 7: render_expression (expression call)
        Phase 8: sentence split + yield
        Phase 9: log + periodic memory save

        Yields:
            Complete sentences only.
        """
        self._turn_count += 1

        try:
            # Load image
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_file)

            # === Phase 1: Gemini perception call ===
            perception_prompt = "この画面の内容を客観的に記述してください。"
            if vision_summary:
                perception_prompt += f"\n\n参考センサー情報:\n{vision_summary}"

            screen_description = await llm_call_with_image(
                VISION_SYSTEM_PROMPT,
                perception_prompt,
                image,
                self._perception_config,
            )
            logger.debug(f"Perception: {screen_description}")

            # === Phase 2: parse_percept (with LLM enrichment) ===
            percept = await parse_percept(
                screen_description,
                llm_call_fn=llm_call,
                state=self._orchestrator.psyche,
            )
            logger.debug(
                f"Percept: emotion={percept.emotion}, intent={percept.intent}, "
                f"topics={percept.topics}"
            )

            # === Phase 3: psyche update ===
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "viewer")
            logger.debug("Psyche tick %d complete", self._orchestrator.tick_count)

            # === Phase 4: recall memories ===
            recall_query = screen_description
            if self._last_response:
                recall_query += " " + self._last_response
            recall_percept = Percept(text=recall_query)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )
            self._orchestrator.set_recalled_memories(memories)

            # === Phase 5: select policy ===
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "viewer"
            )
            logger.debug(
                f"Policy: {policy.get('policy_label', '?')} "
                f"(score={policy.get('_score', 0):.2f})"
            )

            # === Phase 6: silence check ===
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence: %s", policy.get("rationale", ""))
                return

            # === Phase 7: render expression (with psyche enrichment) ===
            self._last_emotion = percept.emotion or "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("viewer")
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=screen_description,
                recent_history=self._conversation_log[-5:],
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                logger.warning("Empty expression result")
                return

            logger.debug(f"Expression: {full_text}")

            # Update last_emotion from meta if available
            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            # Record to conversation log
            if vision_summary:
                self._conversation_log.append(
                    f"[画面情報] {vision_summary[:200]}"
                )
            self._conversation_log.append(f"[キュレネ] {full_text}")
            self._last_response = full_text

            # === Phase 8: Split into sentences + yield ===
            sentences = []
            current = ""

            for i, char in enumerate(full_text):
                current += char
                if char in "。！？!?♪♥♡★☆\n":
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""
                elif char == 'w':
                    next_char = full_text[i + 1] if i + 1 < len(full_text) else None
                    if next_char != 'w':
                        pre_w = current.rstrip('w')
                        if pre_w and not pre_w[-1].isascii():
                            sentence = current.strip()
                            if sentence:
                                sentences.append(sentence)
                            current = ""

            if current.strip():
                sentences.append(current.strip())

            logger.debug(f"Split into {len(sentences)} sentence(s)")

            for i, sentence in enumerate(sentences):
                logger.debug(f"[{i+1}/{len(sentences)}] Yielding: {sentence}")
                yield sentence

            # === Phase 9: Periodic memory save ===
            if self._turn_count % 5 == 0:
                logger.debug("Periodic memory save triggered")
                await self.summarize_and_save()

        except Exception as e:
            logger.error(f"Think streaming failed: {e}")
            import traceback
            traceback.print_exc()
            yield "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"


    async def think_text(
        self,
        user_text: str,
        sender_id: str = "",
        conversation_id: str = "",
    ) -> Optional[str]:
        """テキスト対話入力経路による思考（非ストリーミング版）。

        画面知覚なしでテキスト入力のみから応答を生成する。
        Returns None if psyche decides silence.
        """
        try:
            # Phase 1: parse_percept (with LLM enrichment)
            percept = await parse_percept(
                user_text,
                llm_call_fn=llm_call,
                state=self._orchestrator.psyche,
            )

            # Phase 2: text dialogue input processing
            handoff = self._orchestrator.process_text_input(
                text=user_text,
                sender_id=sender_id,
                conversation_id=conversation_id,
            )

            # Phase 3: psyche update
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "text")

            # Phase 4: recall
            recall_query = user_text
            if self._last_response:
                recall_query += " " + self._last_response
            recall_percept = Percept(text=recall_query)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )
            self._orchestrator.set_recalled_memories(memories)

            # Phase 5: policy
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "text"
            )

            # Phase 6: silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence (text input)")
                return None

            # Phase 7: expression
            self._last_emotion = percept.emotion or "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("text")
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=user_text,
                recent_history=self._conversation_log[-5:],
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return None

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._conversation_log.append(f"[ユーザー] {user_text}")
            self._conversation_log.append(f"[キュレネ] {full_text}")
            self._last_response = full_text
            return full_text

        except Exception as e:
            logger.error(f"Think text failed: {e}")
            return "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"

    async def think_streaming_text(
        self,
        user_text: str,
        sender_id: str = "",
        conversation_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """テキスト対話入力経路による思考（ストリーミング版）。

        画面知覚なしでテキスト入力のみから応答を生成する。
        Yields complete sentences.
        """
        self._turn_count += 1

        try:
            # Phase 1: parse_percept
            percept = await parse_percept(
                user_text,
                llm_call_fn=llm_call,
                state=self._orchestrator.psyche,
            )

            # Phase 2: text dialogue input processing
            handoff = self._orchestrator.process_text_input(
                text=user_text,
                sender_id=sender_id,
                conversation_id=conversation_id,
            )

            # Phase 3: psyche update
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "text")

            # Phase 4: recall
            recall_query = user_text
            if self._last_response:
                recall_query += " " + self._last_response
            recall_percept = Percept(text=recall_query)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )
            self._orchestrator.set_recalled_memories(memories)

            # Phase 5: policy
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "text"
            )

            # Phase 6: silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence (text input)")
                return

            # Phase 7: expression
            self._last_emotion = percept.emotion or "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("text")
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=user_text,
                recent_history=self._conversation_log[-5:],
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            # Log
            self._conversation_log.append(f"[ユーザー] {user_text}")
            self._conversation_log.append(f"[キュレネ] {full_text}")
            self._last_response = full_text

            # Phase 8: sentence split + yield
            sentences = []
            current = ""
            for i, char in enumerate(full_text):
                current += char
                if char in "。！？!?♪♥♡★☆\n":
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""
                elif char == 'w':
                    next_char = full_text[i + 1] if i + 1 < len(full_text) else None
                    if next_char != 'w':
                        pre_w = current.rstrip('w')
                        if pre_w and not pre_w[-1].isascii():
                            sentence = current.strip()
                            if sentence:
                                sentences.append(sentence)
                            current = ""
            if current.strip():
                sentences.append(current.strip())

            for sentence in sentences:
                yield sentence

            # Phase 9: periodic memory save
            if self._turn_count % 5 == 0:
                await self.summarize_and_save()

        except Exception as e:
            logger.error(f"Think streaming text failed: {e}")
            yield "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"


    async def think_spontaneous(self) -> Optional[str]:
        """自発起動経路による思考（非ストリーミング版）。

        外部入力なし時に内部状態から起動候補を形成し、
        起動すべきと判定された場合のみ応答を生成する。
        Returns None if no spontaneous activation or silence chosen.
        """
        try:
            result = self._orchestrator.check_spontaneous_activation()
            if result is None or not result.should_activate:
                return None

            best = max(
                result.candidates,
                key=lambda c: c.activation_strength,
            ) if result.candidates else None
            if not best:
                return None

            # Build Percept from internal activation
            percept = Percept(
                text=f"[内部起動] {best.description}",
                emotion="neutral",
                intent="spontaneous",
            )

            # Psyche update
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "internal")

            # Recall
            recall_percept = Percept(text=percept.text)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )
            self._orchestrator.set_recalled_memories(memories)

            # Policy
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "internal"
            )

            # Silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence (spontaneous)")
                return None

            # Expression
            self._last_emotion = "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("internal")
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context="（自発的思考 — 外部入力なし）",
                recent_history=self._conversation_log[-5:],
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return None

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._conversation_log.append(f"[キュレネ/自発] {full_text}")
            self._last_response = full_text
            return full_text

        except Exception as e:
            logger.error(f"Think spontaneous failed: {e}")
            return None

    async def think_streaming_spontaneous(
        self,
    ) -> AsyncGenerator[str, None]:
        """自発起動経路による思考（ストリーミング版）。

        外部入力なし時に内部状態から起動候補を形成し、
        起動すべきと判定された場合のみ応答を生成する。
        Yields complete sentences.
        """
        try:
            result = self._orchestrator.check_spontaneous_activation()
            if result is None or not result.should_activate:
                return

            best = max(
                result.candidates,
                key=lambda c: c.activation_strength,
            ) if result.candidates else None
            if not best:
                return

            self._turn_count += 1

            # Build Percept from internal activation
            percept = Percept(
                text=f"[内部起動] {best.description}",
                emotion="neutral",
                intent="spontaneous",
            )

            # Psyche update
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "internal")

            # Recall
            recall_percept = Percept(text=percept.text)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )
            self._orchestrator.set_recalled_memories(memories)

            # Policy
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "internal"
            )

            # Silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence (spontaneous)")
                return

            # Expression
            self._last_emotion = "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("internal")
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context="（自発的思考 — 外部入力なし）",
                recent_history=self._conversation_log[-5:],
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._conversation_log.append(f"[キュレネ/自発] {full_text}")
            self._last_response = full_text

            # Sentence split + yield
            sentences = []
            current = ""
            for i, char in enumerate(full_text):
                current += char
                if char in "。！？!?♪♥♡★☆\n":
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""
                elif char == 'w':
                    next_char = full_text[i + 1] if i + 1 < len(full_text) else None
                    if next_char != 'w':
                        pre_w = current.rstrip('w')
                        if pre_w and not pre_w[-1].isascii():
                            sentence = current.strip()
                            if sentence:
                                sentences.append(sentence)
                            current = ""
            if current.strip():
                sentences.append(current.strip())

            for sentence in sentences:
                yield sentence

        except Exception as e:
            logger.error(f"Think streaming spontaneous failed: {e}")
            yield "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"


async def _test_brain():
    """Test function for brain module."""
    print("Starting brain test...")

    # Check if test image exists
    test_image_path = Path("test_capture.jpg")
    if not test_image_path.exists():
        print("Error: test_capture.jpg not found. Run 'python vision.py' first.")
        return

    print(f"Found test image: {test_image_path}")

    # Initialize brain and think
    try:
        brain = CyreneBrain()
        print("Analyzing screen with Cyrene's brain...")

        response = await brain.think(str(test_image_path))

        print("\n" + "=" * 40)
        print("Cyrene:")
        print(response)
        print("=" * 40)

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(_test_brain())
