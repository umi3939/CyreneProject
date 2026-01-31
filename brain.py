"""
brain.py - Thinking engine for Cyrene

Uses Google Gemini 2.5 Flash for screen analysis and
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

from memory_manager import MemoryManager
from psyche.state import PsycheState, Percept
from psyche.pillars import AttachmentState, ContinuityState, IdentityState, ProjectionState
from psyche.reaction import react
from psyche.memory_link import recall_with_mood
from psyche.fear import compute_fear_index
from psyche import attachment_manager, identity_manager, continuity_manager, projection_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TAG_VALENCE: dict[str, float] = {
    "happy": 0.7, "sad": -0.6, "angry": -0.5, "surprised": 0.3,
    "scared": -0.5, "loving": 0.8, "teasing": 0.4, "neutral": 0.0,
}


class CyreneBrain:
    """
    AI thinking engine using Gemini 2.5 Flash.
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
        self._model_name = "gemini-2.5-flash"
        logger.info("Gemini client initialized")

        # Load persona from identity.md
        self._persona = self._load_identity()

        # Generation config (system instruction, temperature, etc.)
        # Gemini 2.5+ defaults safety filters to OFF, so no safety_settings needed.
        self._config = types.GenerateContentConfig(
            system_instruction=self._persona,
            temperature=1.2,  # High creativity for entertaining reactions
            max_output_tokens=1024,  # Thinking + output tokens combined
        )

        # Summary generation config (separate from chat config)
        self._summary_config = types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=512,
        )

        # Long-term memory (with embedding support)
        self._memory = MemoryManager(embed_fn=self._embed_text)
        self._turn_count = 0

        # Self-managed conversation log (replaces SDK-internal _curated_history)
        self._conversation_log: list[str] = []
        self._last_response: str = ""

        # Create chat session for short-term memory (context retention)
        self._chat = self._create_chat()
        logger.info(f"Gemini model ready ({self._model_name})")

        # Psyche (psychological state tracker)
        self._last_psyche_update = time.monotonic()
        self._init_psyche()

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
                await self._memory.add_memory(summary, keywords, importance)
                logger.info(f"Memory saved: {summary}")

                # Update continuity pillar with new memory count
                if self._psyche.continuity is not None:
                    self._psyche = self._psyche.model_copy(update={
                        "continuity": self._psyche.continuity.model_copy(
                            update={
                                "memory_count": len(self._memory._memories),
                            }
                        ),
                    })
                    self._recompute_fear()
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

    def _init_psyche(self):
        """Initialize the psychological state with four pillars and fear index."""
        identity = IdentityState(
            core_traits=["romantic", "sweet", "playful", "caring", "confident"],
            trait_confidence={
                "romantic": 0.9,
                "sweet": 0.9,
                "playful": 0.8,
                "caring": 0.9,
                "confident": 0.8,
            },
        )
        attachment = AttachmentState()
        continuity = ContinuityState(
            memory_count=len(self._memory._memories),
        )
        projection = ProjectionState(
            goals=[{
                "id": "entertain",
                "description": "視聴者を楽しませる",
                "progress": 0.1,
                "status": "active",
            }],
        )

        fear = compute_fear_index(
            identity_risk=identity_manager.calc_identity_risk(identity),
            attachment_risk=attachment_manager.calc_attachment_risk(attachment),
            continuity_risk=continuity_manager.calc_continuity_risk(
                memory_count=continuity.memory_count,
            ),
            projection_risk=projection_manager.calc_projection_risk(projection),
        )

        self._psyche = PsycheState(
            identity=identity,
            attachment=attachment,
            continuity=continuity,
            projection=projection,
            fear_index=fear,
        )
        logger.info(
            f"Psyche initialized: fear={fear.value:.2f}, "
            f"dominant_fear={fear.dominant_fear}"
        )

    def _format_psyche_for_prompt(self) -> str:
        """Generate the 【心理状態】 section string for the Gemini prompt."""
        p = self._psyche
        lines = [
            "【心理状態（内面）】",
            f"感情: {p.emotion_summary()}",
            f"ムード: valence={p.mood.valence:.2f}, arousal={p.mood.arousal:.2f}",
            f"ドライブ: social={p.drives.social:.2f}, "
            f"curiosity={p.drives.curiosity:.2f}, "
            f"expression={p.drives.expression:.2f}",
            p.fear_summary(),
        ]
        if p.dominant_emotion_value > 0.3:
            lines.append(
                f"支配的感情: {p.dominant_emotion} "
                f"({p.dominant_emotion_value:.2f})"
            )
        lines.append(
            "この内面状態を自然に反映した反応をしてください。"
            "機械的に読み上げないこと。"
        )
        return "\n".join(lines)

    def _update_psyche(self, response_text: str, vision_summary: str = ""):
        """
        Update psyche state from Gemini's response emotion tag.

        Extracts [happy] etc. from the response, converts to a Percept,
        runs react(), updates attachment bond, and recomputes fear.
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

        # React: emotion update, decay, drive shift, mood drift
        self._psyche = react(percept, self._psyche, delta_time=delta)

        # Update attachment: every interaction is a positive bond event
        if self._psyche.attachment is not None:
            self._psyche = self._psyche.model_copy(update={
                "attachment": attachment_manager.update_bond(
                    self._psyche.attachment, "viewer", "positive", abs(valence)
                ),
            })

        # Recompute fear from updated pillars
        self._recompute_fear()

        logger.info(
            f"Psyche updated: emotion={tag}, valence={valence:.1f}, "
            f"mood={self._psyche.mood.valence:.2f}, "
            f"fear={self._psyche.fear_level:.2f}"
        )

    def _recompute_fear(self):
        """Recompute fear index from current pillar states."""
        p = self._psyche
        fear = compute_fear_index(
            identity_risk=(
                identity_manager.calc_identity_risk(p.identity)
                if p.identity else 0.0
            ),
            attachment_risk=(
                attachment_manager.calc_attachment_risk(p.attachment)
                if p.attachment else 0.7
            ),
            continuity_risk=continuity_manager.calc_continuity_risk(
                memory_count=p.continuity.memory_count if p.continuity else 0,
            ),
            projection_risk=(
                projection_manager.calc_projection_risk(p.projection)
                if p.projection else 0.7
            ),
        )
        self._psyche = self._psyche.model_copy(update={"fear_index": fear})

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
            recall_percept, self._psyche, self._memory, top_k=3
        )
        if memories:
            lines = [f"- [{m['date'][:10]}] {m['summary']}" for m in memories]
            memory_block = "\n".join(lines)
            parts.append(f"\n【過去の思い出】\n{memory_block}")

        # Psyche state section
        parts.append(f"\n{self._format_psyche_for_prompt()}")

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
        Analyze screen image and generate a response.
        Returns None if AI decides to stay silent (PASS).

        Args:
            image_path: Path to JPEG image file.
            vision_summary: Formatted sensor data from HybridEye (YOLO + OCR).

        Returns:
            Generated text in Cyrene's voice, or None for silence.
        """
        prompt = await self._build_prompt(vision_summary)

        try:
            # Load image directly with PIL (no file upload needed)
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_file)

            # Send message via chat session (retains conversation history)
            response = await self._chat.send_message([prompt, image])

            if response and response.text:
                result = response.text.strip()

                # Check for PASS (AI wants to stay silent)
                if result.upper() == "PASS":
                    logger.info("AI chose silence (PASS)")
                    return None

                # Update psyche from response emotion tag
                self._update_psyche(result, vision_summary)

                return result
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Think failed: {e}")
            return "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"

    async def think_streaming(
        self,
        image_path: str,
        vision_summary: str = ""
    ) -> AsyncGenerator[str, None]:
        """
        Analyze screen image and generate a response, then yield sentences.

        Args:
            image_path: Path to JPEG image file.
            vision_summary: Formatted sensor data from HybridEye (YOLO + OCR).

        Yields:
            Complete sentences only.
        """
        self._turn_count += 1
        prompt = await self._build_prompt(vision_summary)

        try:
            # Load image directly with PIL (no file upload needed)
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_file)

            # === STEP 1: Get FULL response via chat session ===
            logger.info("Requesting response from Gemini...")
            response = await self._chat.send_message([prompt, image])

            # Extract full text
            if not response or not response.text:
                logger.warning("Empty response from Gemini")
                return

            full_text = response.text.strip()
            logger.info(f"Full response: {full_text}")

            # Record to self-managed conversation log
            if vision_summary:
                self._conversation_log.append(
                    f"[画面情報] {vision_summary[:200]}"
                )
            self._conversation_log.append(f"[キュレネ] {full_text}")
            self._last_response = full_text

            # Check for PASS
            if full_text.upper() == "PASS":
                logger.info("AI chose silence (PASS)")
                return

            # Update psyche from response emotion tag
            self._update_psyche(full_text, vision_summary)

            # === STEP 2: Split into sentences ===
            # Terminators: 。！？!?\n
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
                    # Treat trailing w/ww/www as delimiter only after Japanese text
                    next_char = full_text[i + 1] if i + 1 < len(full_text) else None
                    if next_char != 'w':
                        pre_w = current.rstrip('w')
                        if pre_w and not pre_w[-1].isascii():
                            sentence = current.strip()
                            if sentence:
                                sentences.append(sentence)
                            current = ""

            # Add remaining text (if any)
            if current.strip():
                sentences.append(current.strip())

            logger.info(f"Split into {len(sentences)} sentence(s)")

            # === STEP 3: Yield each complete sentence ===
            for i, sentence in enumerate(sentences):
                logger.info(f"[{i+1}/{len(sentences)}] Yielding: {sentence}")
                yield sentence

            # === STEP 4: Periodic memory save (every 5 turns) ===
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
