"""
brain.py - Thinking engine for Cyrene

Uses Google Gemini 3 Flash Preview for screen analysis and
response generation with the Cyrene persona from identity.md.
"""

import asyncio
import json
import logging
import os
import re
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TAG_VALENCE: dict[str, float] = {
    "happy": 0.7, "sad": -0.6, "angry": -0.5, "surprised": 0.3,
    "scared": -0.5, "loving": 0.8, "teasing": 0.4, "neutral": 0.0,
}


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
                logger.info("Not enough conversation to summarize")
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
                logger.info(f"Memory saved: {summary}")

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

    def _update_psyche(self, response_text: str, vision_summary: str = ""):
        """
        Update psyche state from Gemini's response emotion tag.
        Delegates to PsycheOrchestrator for full pipeline execution.
        """
        # Extract emotion tag like [happy] from response
        m = re.match(r"\[(\w+)\]", response_text)
        tag = m.group(1).lower() if m else "neutral"
        valence = _TAG_VALENCE.get(tag, 0.0)

        # Build percept from the response
        percept = Percept(
            text=vision_summary or response_text[:100],
            meaning=response_text[:200],
            emotion=tag,
            intent="expression",
            emotion_valence=valence,
        )

        # Time delta
        now = time.monotonic()
        delta = now - self._last_psyche_update
        self._last_psyche_update = now

        # Delegate to orchestrator (runs all psyche phases)
        self._orchestrator.post_response_update(percept, delta, "viewer")

    @property
    def last_emotion(self) -> str:
        return self._last_emotion

    def _build_persona_dict(self) -> dict:
        return {
            "name": "キュレネ",
            "tone": "romantic, sweet, playful",
            "style_rules": {
                "禁止": ["敬語", "絵文字", "行動描写", "説明的な回答"],
                "推奨": ["♪♡使用可", "い抜き言葉", "カジュアルなタメ口", "ロマンチック"],
            },
        }

    async def _build_prompt(self, vision_summary: str = "") -> str:
        """
        Build the prompt for screen reaction.
        Async because recall() uses embedding search.

        Args:
            vision_summary: Formatted string from HybridEye analysis.

        Returns:
            Prompt string for Gemini.
        """
        parts = ["""この画面を見て、あなたの言葉でリアクションして。

【画像について】
画像を高解像度で提供しています。画面の隅々、小さなUIテキスト、背景の変化まで詳細に観察し、反応に反映してください。"""]

        # Build recall query from vision data + last AI response
        recall_query = vision_summary
        if self._last_response:
            recall_query += " " + self._last_response

        # Recall related long-term memories (mood-congruent bias)
        recall_percept = Percept(text=recall_query)
        memories = await recall_with_mood(
            recall_percept, self._orchestrator.psyche, self._memory, top_k=3
        )
        if memories:
            lines = [f"- [{m['date'][:10]}] {m['summary']}" for m in memories]
            memory_block = "\n".join(lines)
            parts.append(f"\n【過去の思い出】\n{memory_block}")

        # Psyche state section (full enrichment from orchestrator)
        parts.append(f"\n{self._orchestrator.get_prompt_enrichment()}")

        # Policy suggestions section
        policy_percept = Percept(text=recall_query)
        policy_text = self._orchestrator.get_policy_suggestions(
            policy_percept, memories or [],
        )
        if policy_text:
            parts.append(f"\n{policy_text}")

        if vision_summary:
            parts.append(f"""
{vision_summary}

このセンサー情報と画像を組み合わせて状況を判断してください。
ただし、センサー情報を機械的に読み上げるのではなく、自然な会話として反映すること。""")

        parts.append("""
【感情タグ - 必須】
発話の先頭に、今の感情を表すタグを必ず1つ付けて出力すること。

使用可能なタグ:
- [happy] 嬉しい、興奮、楽しい、ワクワク
- [sad] 悲しい、がっかり、寂しい
- [angry] 怒り、イライラ、ムカつく
- [surprised] 驚き、びっくり
- [scared] 怖い、焦り、ピンチ、緊張
- [loving] 甘え、愛情表現、デレデレ
- [teasing] からかい、いたずらっぽい、ニヤニヤ
- [neutral] 通常、落ち着いている

【ルール】
1. 画面の内容に対してリアクションや感想を言う。
2. もし前と同じような画面で特に言うことがなければ、「PASS」とだけ出力する。
3. 短く、感情豊かに。必ず感情タグから始めること。""")

        return "\n".join(parts)

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

            # Phase 2: Parse
            percept = await parse_percept(screen_description)

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

            # Phase 5: Policy
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "viewer"
            )

            # Phase 6: Silence check
            if is_silence_policy(policy):
                logger.info("Psyche chose silence")
                return None

            # Phase 7: Expression
            self._last_emotion = percept.emotion or "neutral"
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=screen_description,
                recent_history=self._conversation_log[-5:],
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
            logger.info(f"Perception: {screen_description}")

            # === Phase 2: parse_percept ===
            percept = await parse_percept(screen_description)
            logger.info(
                f"Percept: emotion={percept.emotion}, intent={percept.intent}, "
                f"topics={percept.topics}"
            )

            # === Phase 3: psyche update ===
            now = time.monotonic()
            delta = now - self._last_psyche_update
            self._last_psyche_update = now
            self._orchestrator.post_response_update(percept, delta, "viewer")
            logger.info("Psyche tick %d complete", self._orchestrator.tick_count)

            # === Phase 4: recall memories ===
            recall_query = screen_description
            if self._last_response:
                recall_query += " " + self._last_response
            recall_percept = Percept(text=recall_query)
            memories = await recall_with_mood(
                recall_percept, self._orchestrator.psyche, self._memory, top_k=3
            )

            # === Phase 5: select policy ===
            policy = self._orchestrator.select_policy_dict(
                percept, memories or [], "viewer"
            )
            logger.info(
                f"Policy: {policy.get('policy_label', '?')} "
                f"(score={policy.get('_score', 0):.2f})"
            )

            # === Phase 6: silence check ===
            if is_silence_policy(policy):
                logger.info("Psyche chose silence: %s", policy.get("rationale", ""))
                return

            # === Phase 7: render expression ===
            self._last_emotion = percept.emotion or "neutral"
            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=screen_description,
                recent_history=self._conversation_log[-5:],
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                logger.warning("Empty expression result")
                return

            logger.info(f"Expression: {full_text}")

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

            logger.info(f"Split into {len(sentences)} sentence(s)")

            for i, sentence in enumerate(sentences):
                logger.info(f"[{i+1}/{len(sentences)}] Yielding: {sentence}")
                yield sentence

            # === Phase 9: Periodic memory save ===
            if self._turn_count % 5 == 0:
                logger.info("Periodic memory save triggered")
                await self.summarize_and_save()

        except Exception as e:
            logger.error(f"Think streaming failed: {e}")
            import traceback
            traceback.print_exc()
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
