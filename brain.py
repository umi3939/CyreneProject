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
from collections import deque
from dataclasses import dataclass, field
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
from src.llm_wrapper import (
    llm_call,
    llm_call_with_image,
    VISION_SYSTEM_PROMPT,
    get_error_stats,
    get_fallback_state,
    is_api_available,
    should_safe_shutdown,
    reset_error_stats,
    reset_fallback_state,
)

from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


# ── API Fallback Detection ────────────────────────────────────

# Marker substring present in all LLM fallback responses
_FALLBACK_INDICATOR = "no_llm_available"


def _is_perception_fallback(text: str) -> bool:
    """Check if a perception result is a fallback (API unavailable) response.

    When perception call fails, llm_wrapper returns a JSON string containing
    "no_llm_available". This function detects that case so the think methods
    can skip the current frame instead of feeding garbage into psyche.
    """
    return _FALLBACK_INDICATOR in text


class SafeShutdownRequested(Exception):
    """Raised when fallback mode duration exceeds the maximum.

    The main loop should catch this, persist internal state, and exit.
    This exception never propagates into psyche processing.
    """
    pass


# ── Structured Context Entry ────────────────────────────────────

@dataclass(frozen=True)
class ContextEntry:
    """Immutable structured dialogue history entry.

    Attributes:
        speaker_label: Display label (e.g. "キュレネ", "ユーザー", "画面情報", "キュレネ/自発")
        text: Utterance body text.
        pathway: Input pathway identifier: "vision" / "text" / "internal".
        partner_id: Dialogue partner identifier (sender_id, "viewer", "internal").
        timestamp: Monotonic clock value at entry creation.
    """
    speaker_label: str
    text: str
    pathway: str
    partner_id: str
    timestamp: float


# Time gap thresholds (seconds) and their display annotations.
# Design: 3-level staged annotations. Thresholds are configurable fixed values.
_TIME_GAP_THRESHOLDS: list[tuple[float, str]] = [
    (300.0, "（かなり時間が経った）"),   # >= 5 minutes
    (60.0, "（しばらく間があった）"),     # >= 1 minute
    (15.0, "（少し間があった）"),          # >= 15 seconds
]


class DialogueContextManager:
    """Manages structured dialogue history with FIFO eviction and text rendering.

    This class replaces the simple ``list[str]`` conversation log with structured
    entries carrying pathway, partner_id, and timestamp metadata.  It provides a
    text representation for the expression call (Gemini) that never includes
    policy labels, scores, or internal psychological information.

    Args:
        max_entries: Maximum entries in FIFO buffer.
        window_size: Number of entries to include in text representation.
        time_gap_thresholds: Optional override for gap annotation thresholds.
    """

    def __init__(
        self,
        max_entries: int = 100,
        window_size: int = 20,
        time_gap_thresholds: list[tuple[float, str]] | None = None,
    ):
        self._max_entries = max(1, max_entries)
        self._window_size = max(1, window_size)
        self._entries: deque[ContextEntry] = deque(maxlen=self._max_entries)
        self._prev_session_last: ContextEntry | None = None
        self._time_gap_thresholds = time_gap_thresholds or _TIME_GAP_THRESHOLDS

    # ── Public: add entry ───────────────────────────────────────

    def add_entry(
        self,
        speaker_label: str,
        text: str,
        pathway: str,
        partner_id: str,
        timestamp: float | None = None,
    ) -> None:
        """Append a structured entry to the context buffer."""
        ts = timestamp if timestamp is not None else time.monotonic()
        entry = ContextEntry(
            speaker_label=speaker_label,
            text=text,
            pathway=pathway,
            partner_id=partner_id,
            timestamp=ts,
        )
        self._entries.append(entry)

    # ── Public: session reset ───────────────────────────────────

    def reset_session(self) -> None:
        """Clear entries for session boundary (e.g. summarize_and_save).

        Saves the last entry as previous-session context before clearing.
        """
        if self._entries:
            self._prev_session_last = self._entries[-1]
        self._entries.clear()

    # ── Public: text representation for expression call ─────────

    def render_text(self) -> str:
        """Build text representation for the expression call.

        Returns a string of ``[speaker_label] text`` lines for the most recent
        ``window_size`` entries, with time-gap annotations inserted between
        entries where the interval exceeds thresholds.

        Policy labels, partner IDs, and pathway identifiers are excluded.
        """
        window = list(self._entries)[-self._window_size:]
        if not window:
            return ""

        lines: list[str] = []

        # Prepend previous-session last entry if available and window doesn't
        # already contain entries from before the reset
        if self._prev_session_last is not None:
            prev = self._prev_session_last
            lines.append(f"[前回最後の発話] [{prev.speaker_label}] {prev.text}")
            # Gap annotation between prev session and first current entry
            gap_ann = self._gap_annotation(prev.timestamp, window[0].timestamp)
            if gap_ann:
                lines.append(gap_ann)

        for i, entry in enumerate(window):
            # Insert gap annotation between consecutive entries
            if i > 0:
                gap_ann = self._gap_annotation(window[i - 1].timestamp, entry.timestamp)
                if gap_ann:
                    lines.append(gap_ann)
            lines.append(f"[{entry.speaker_label}] {entry.text}")

        return "\n".join(lines)

    # ── Public: structured data access (read-only) ──────────────

    def get_entries(self) -> list[ContextEntry]:
        """Return a copy of all entries (read-only access)."""
        return list(self._entries)

    def get_window_entries(self) -> list[ContextEntry]:
        """Return entries within the current window."""
        return list(self._entries)[-self._window_size:]

    @property
    def prev_session_last(self) -> ContextEntry | None:
        return self._prev_session_last

    def __len__(self) -> int:
        return len(self._entries)

    # ── Internal ────────────────────────────────────────────────

    def _gap_annotation(self, t_prev: float, t_curr: float) -> str:
        """Return time gap annotation string, or empty string if below all thresholds."""
        gap = t_curr - t_prev
        for threshold, annotation in self._time_gap_thresholds:
            if gap >= threshold:
                return annotation
        return ""


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

        # Structured dialogue context manager (replaces simple list[str] log)
        self._context = DialogueContextManager(
            max_entries=100,
            window_size=20,
        )
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
        """Reset conversation history and start a fresh chat session.

        Saves the last entry as previous-session context before clearing.
        """
        self._chat = self._create_chat()
        self._context.reset_session()
        self._last_response = ""
        logger.info("Memory reset - new chat session started")

    async def summarize_and_save(self):
        """
        Summarize recent conversation history and save as a long-term memory.
        Uses Gemini (single-shot, outside the chat session) to generate a summary.
        Uses self-managed conversation log instead of SDK internals.
        """
        try:
            entries = self._context.get_entries()
            if len(entries) < 2:
                logger.debug("Not enough conversation to summarize")
                return

            # Use last 10 entries for summarization
            recent = entries[-10:]
            conversation_text = "\n".join(
                f"[{e.speaker_label}] {e.text}" for e in recent
            )

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

            # Reset conversation context after successful save
            self.reset_memory()

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

    @property
    def fear_level(self) -> float:
        """Current aggregate fear level (0.0-1.0) from the 4-pillar system."""
        return self._orchestrator.fear_level

    def _log_policy_suggestions(
        self, percept, memories: list, user_id: str
    ) -> None:
        """Log policy suggestions for internal decision transparency."""
        try:
            suggestions = self._orchestrator.get_policy_suggestions(
                percept, memories or [], user_id
            )
            if suggestions:
                logger.debug("Policy suggestions:\n%s", suggestions)
        except Exception:
            pass

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

    # ── API Error Resilience: Observation & Fallback Mode ──────

    @property
    def api_error_summary(self) -> dict:
        """Return READ-ONLY API error statistics for observation.

        This data is never referenced by psyche processing.
        """
        return get_error_stats().get_summary()

    @property
    def api_fallback_status(self) -> dict:
        """Return READ-ONLY fallback mode status for observation."""
        return get_fallback_state().get_status()

    @property
    def is_api_available(self) -> bool:
        """Whether the API is currently reachable (not in fallback mode)."""
        return is_api_available()

    def _check_safe_shutdown(self) -> None:
        """Check if fallback mode duration requires safe shutdown.

        If the API has been unreachable for longer than the configured
        maximum fallback duration, persist state and raise SafeShutdownRequested.
        This ensures the system does not run indefinitely without external input.
        """
        if should_safe_shutdown():
            logger.critical(
                "Fallback mode duration exceeded maximum. "
                "Persisting state and requesting safe shutdown."
            )
            self.save_state()
            raise SafeShutdownRequested(
                "API unreachable for too long, safe shutdown initiated"
            )

    async def think(
        self,
        image_path: str = "",
        vision_summary: str = "",
        image: Optional[Image.Image] = None,
    ) -> Optional[str]:
        """
        2-call structure (non-streaming version).
        Returns None if psyche decides silence.

        Args:
            image_path: Path to JPEG image file (used if image is None).
            vision_summary: Formatted sensor data from HybridEye (YOLO + OCR).
            image: PIL Image directly (preferred over image_path).

        Returns:
            Generated text in Cyrene's voice, or None for silence.

        Raises:
            SafeShutdownRequested: If fallback mode duration exceeds maximum.
        """
        try:
            # Check if safe shutdown is needed due to prolonged API unavailability
            self._check_safe_shutdown()

            if image is None:
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

            # Vision pathway: if perception call failed (API unreachable),
            # skip this frame entirely. Psyche processing is not executed
            # with invalid perception data.
            if _is_perception_fallback(screen_description):
                logger.warning("Perception call returned fallback, skipping frame")
                return None

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
            self._log_policy_suggestions(percept, memories, "viewer")

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
                recent_history=self._context.get_window_entries(),
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return None

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._last_response = full_text

            # Notify self-action perception
            if full_text:
                self._orchestrator.notify_self_output(
                    response_text=full_text,
                    policy_label=policy.get("policy_label", ""),
                )

            return full_text

        except SafeShutdownRequested:
            raise  # Propagate to main loop for safe shutdown
        except Exception as e:
            logger.error(f"Think failed: {e}")
            return "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"

    async def think_streaming(
        self,
        image_path: str = "",
        vision_summary: str = "",
        image: Optional[Image.Image] = None,
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
            # Check if safe shutdown is needed due to prolonged API unavailability
            self._check_safe_shutdown()

            # Load image (direct PIL Image preferred over file path)
            if image is None:
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

            # Vision pathway: if perception call failed, skip this frame
            if _is_perception_fallback(screen_description):
                logger.warning("Perception call returned fallback, skipping frame")
                return

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
            self._log_policy_suggestions(percept, memories, "viewer")
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
                recent_history=self._context.get_window_entries(),
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

            # Record to structured context
            if vision_summary:
                self._context.add_entry(
                    speaker_label="画面情報",
                    text=vision_summary[:200],
                    pathway="vision",
                    partner_id="viewer",
                )
            self._context.add_entry(
                speaker_label="キュレネ",
                text=full_text,
                pathway="vision",
                partner_id="viewer",
            )
            self._last_response = full_text

            # Notify self-action perception
            if full_text:
                self._orchestrator.notify_self_output(
                    response_text=full_text,
                    policy_label=policy.get("policy_label", ""),
                )

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

        except SafeShutdownRequested:
            raise  # Propagate to main loop for safe shutdown
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

        Raises:
            SafeShutdownRequested: If fallback mode duration exceeds maximum.
        """
        try:
            # Check if safe shutdown is needed
            self._check_safe_shutdown()

            # Phase 1: parse_percept (with LLM enrichment)
            # Text pathway: parse_percept has its own LLM-less fallback
            # (heuristic analysis). If LLM enrichment fails, it continues
            # with heuristic results. No frame skip needed.
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
            self._log_policy_suggestions(percept, memories, "text")

            # Phase 6: silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence (text input)")
                return None

            # Phase 7: expression
            self._last_emotion = percept.emotion or "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("text")

            # Record user input before expression call
            self._context.add_entry(
                speaker_label="ユーザー",
                text=user_text,
                pathway="text",
                partner_id=sender_id or "text",
            )

            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=user_text,
                recent_history=self._context.get_window_entries(),
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return None

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._context.add_entry(
                speaker_label="キュレネ",
                text=full_text,
                pathway="text",
                partner_id=sender_id or "text",
            )
            self._last_response = full_text

            # Notify self-action perception
            if full_text:
                self._orchestrator.notify_self_output(
                    response_text=full_text,
                    policy_label=policy.get("policy_label", ""),
                )

            return full_text

        except SafeShutdownRequested:
            raise  # Propagate to main loop for safe shutdown
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

        Raises:
            SafeShutdownRequested: If fallback mode duration exceeds maximum.
        """
        self._turn_count += 1

        try:
            # Check if safe shutdown is needed
            self._check_safe_shutdown()

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
            self._log_policy_suggestions(percept, memories, "text")

            # Phase 6: silence check
            if is_silence_policy(policy):
                logger.debug("Psyche chose silence (text input)")
                return

            # Phase 7: expression
            self._last_emotion = percept.emotion or "neutral"
            enrichment = self._orchestrator.get_prompt_enrichment("text")

            # Record user input before expression
            self._context.add_entry(
                speaker_label="ユーザー",
                text=user_text,
                pathway="text",
                partner_id=sender_id or "text",
            )

            expr_result = await render_expression(
                state=self._orchestrator.psyche,
                policy=policy,
                memory_snippet=memories or [],
                persona=self._persona_dict,
                llm_call_fn=llm_call,
                screen_context=user_text,
                recent_history=self._context.get_window_entries(),
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            # Record response
            self._context.add_entry(
                speaker_label="キュレネ",
                text=full_text,
                pathway="text",
                partner_id=sender_id or "text",
            )
            self._last_response = full_text

            # Notify self-action perception
            if full_text:
                self._orchestrator.notify_self_output(
                    response_text=full_text,
                    policy_label=policy.get("policy_label", ""),
                )

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

        except SafeShutdownRequested:
            raise  # Propagate to main loop for safe shutdown
        except Exception as e:
            logger.error(f"Think streaming text failed: {e}")
            yield "ごめんなさい、ちょっと回線が重いみたい…少し待ってね？"


    async def think_spontaneous(self) -> Optional[str]:
        """自発起動経路による思考（非ストリーミング版）。

        外部入力なし時に内部状態から起動候補を形成し、
        起動すべきと判定された場合のみ応答を生成する。
        Returns None if no spontaneous activation or silence chosen.

        Note: Spontaneous pathway does not use external API for perception
        (psyche only). However, the expression call uses API.
        Safe shutdown check is still performed.

        Raises:
            SafeShutdownRequested: If fallback mode duration exceeds maximum.
        """
        try:
            # Check safe shutdown (spontaneous can still run during fallback,
            # but not indefinitely)
            self._check_safe_shutdown()

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
            self._log_policy_suggestions(percept, memories, "internal")

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
                recent_history=self._context.get_window_entries(),
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return None

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._context.add_entry(
                speaker_label="キュレネ/自発",
                text=full_text,
                pathway="internal",
                partner_id="internal",
            )
            self._last_response = full_text

            # Notify self-action perception
            if full_text:
                self._orchestrator.notify_self_output(
                    response_text=full_text,
                    policy_label=policy.get("policy_label", ""),
                )

            return full_text

        except SafeShutdownRequested:
            raise  # Propagate to main loop for safe shutdown
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

        Raises:
            SafeShutdownRequested: If fallback mode duration exceeds maximum.
        """
        try:
            # Check safe shutdown
            self._check_safe_shutdown()

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
            self._log_policy_suggestions(percept, memories, "internal")

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
                recent_history=self._context.get_window_entries(),
                psyche_enrichment=enrichment,
            )
            full_text = expr_result.get("text", "")
            meta = expr_result.get("meta", {})

            if not full_text:
                return

            if meta.get("emotion"):
                self._last_emotion = meta["emotion"]

            self._context.add_entry(
                speaker_label="キュレネ/自発",
                text=full_text,
                pathway="internal",
                partner_id="internal",
            )
            self._last_response = full_text

            # Notify self-action perception
            if full_text:
                self._orchestrator.notify_self_output(
                    response_text=full_text,
                    policy_label=policy.get("policy_label", ""),
                )

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

        except SafeShutdownRequested:
            raise  # Propagate to main loop for safe shutdown
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
