"""
tests/test_e2e_smoke.py - End-to-end smoke test with real Gemini API.

Verifies that the system's 2-call structure (perception + expression) works
correctly with actual Gemini API responses. All tests are skipped when
GEMINI_API_KEY is not set in the environment.

Design document: design_e2e_smoke_test.md

Hierarchy:
  1. Connection foundation (API key, LLM wrapper init)
  2. Perception pathway (parse_percept with real LLM)
  3. Expression pathway (render_expression with real LLM)
  4. Integrated pipeline (multi-turn orchestrator cycle)
  5. Text input pathway (think_text pipeline)
  6. State contamination guard

Safety:
  - No persistent state is written (temp data dir used)
  - Orchestrator instances are disposable (never saved)
  - All assertions are format-only (type, keys, non-empty)
  - API content quality/meaning is never evaluated
  - Normal 'python -m pytest tests/' does NOT run these tests (marker skip)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

# ── Skip marker: all tests require GEMINI_API_KEY ────────────────

_HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="GEMINI_API_KEY not set; e2e smoke tests skipped",
)

# Per-test timeout for API calls (seconds)
_API_TIMEOUT = 60.0

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────


def _make_log_dir() -> Path:
    """Create e2e log directory if needed, return path."""
    log_dir = Path(__file__).parent.parent / "logs" / "e2e_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_log(log_dir: Path, test_name: str, data: dict[str, Any]) -> None:
    """Write a JSON log for a single test result."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{test_name}.json"
    log_path = log_dir / filename
    log_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _make_temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with minimal required files."""
    data = tmp_path / "data"
    data.mkdir()

    # Minimal example_memories.json
    memories = [
        {
            "id": 1,
            "summary": "e2e test memory entry",
            "keywords": ["test"],
            "importance": 3,
            "date": "2026-01-01T00:00:00",
            "protected": False,
            "last_recalled": None,
        },
    ]
    (data / "example_memories.json").write_text(
        json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Minimal attachments
    (data / "example_attachments.json").write_text(
        json.dumps({}, ensure_ascii=False), encoding="utf-8"
    )

    # Minimal identity
    (data / "identity.json").write_text(
        json.dumps(
            {
                "core_traits": ["romantic", "caring"],
                "trait_confidence": {"romantic": 0.9, "caring": 0.8},
                "pending_changes": [],
                "risk": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Minimal projections
    (data / "projections.json").write_text(
        json.dumps(
            {
                "goals": [
                    {
                        "id": "e2e_test",
                        "description": "e2e test goal",
                        "progress": 0.1,
                        "status": "active",
                    }
                ],
                "risk": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Minimal state
    (data / "state.json").write_text(
        json.dumps(
            {
                "test_user": {
                    "emotions": {
                        "joy": 0.0, "sad": 0.0, "fear": 0.0,
                        "anger": 0.0, "calm": 0.5,
                    },
                    "drives": {"social": 0.5, "curiosity": 0.5},
                    "mood": 0.0,
                    "last_updated": datetime.now().isoformat(timespec="seconds"),
                    "loss_aversion": 0.3,
                    "fear_index": 0.0,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Minimal persona
    (data / "persona.json").write_text(
        json.dumps(
            {
                "name": "キュレネ",
                "first_person": "あたし",
                "second_person": "あなた",
                "tone": "romantic, sweet",
                "style_rules": {"禁止": ["です", "ます"], "推奨": ["♪", "！"]},
                "example_lines": ["ふふっ♪"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # responsibility.json (empty)
    (data / "responsibility.json").write_text(
        json.dumps({}, ensure_ascii=False), encoding="utf-8"
    )

    return data


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using the same logic as brain.py."""
    sentences = []
    current = ""
    for i, char in enumerate(text):
        current += char
        if char in "。！？!?♪♥♡★☆\n":
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ""
        elif char == 'w':
            next_char = text[i + 1] if i + 1 < len(text) else None
            if next_char != 'w':
                pre_w = current.rstrip('w')
                if pre_w and not pre_w[-1].isascii():
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def log_dir():
    """Provide a log directory for e2e test results."""
    return _make_log_dir()


@pytest.fixture()
def temp_data(tmp_path):
    """Provide a temporary data directory for disposable orchestrator."""
    return _make_temp_data_dir(tmp_path)


@pytest.fixture()
def orchestrator(temp_data):
    """Create a disposable PsycheOrchestrator (never saved to disk)."""
    from psyche.orchestrator import PsycheOrchestrator

    orch = PsycheOrchestrator(memory_count=1, data_dir=temp_data)
    return orch


# ── Shared persona dict ─────────────────────────────────────────

_PERSONA = {
    "name": "キュレネ",
    "tone": "romantic, sweet, playful",
    "style_rules": {
        "禁止": ["敬語", "絵文字"],
        "推奨": ["♪♡使用可", "い抜き言葉", "カジュアルなタメ口"],
    },
}


# ── Hierarchy 1: Connection Foundation ───────────────────────────


class TestConnectionFoundation:
    """Verify API key presence and LLM wrapper initialization."""

    def test_api_key_present(self):
        """GEMINI_API_KEY is set in environment."""
        key = os.getenv("GEMINI_API_KEY")
        assert key is not None
        assert len(key) > 0

    def test_llm_wrapper_init(self):
        """LLM wrapper can be imported and basic call structure exists."""
        from src.llm_wrapper import (
            llm_call,
            llm_call_with_system,
            llm_call_with_image,
            VISION_SYSTEM_PROMPT,
            EXPRESSION_SYSTEM_PROMPT,
            PERCEPTION_SYSTEM_PROMPT,
        )
        assert callable(llm_call)
        assert callable(llm_call_with_system)
        assert callable(llm_call_with_image)
        assert isinstance(VISION_SYSTEM_PROMPT, str)
        assert len(VISION_SYSTEM_PROMPT) > 0
        assert isinstance(EXPRESSION_SYSTEM_PROMPT, str)
        assert len(EXPRESSION_SYSTEM_PROMPT) > 0
        assert isinstance(PERCEPTION_SYSTEM_PROMPT, str)
        assert len(PERCEPTION_SYSTEM_PROMPT) > 0

    @pytest.mark.asyncio
    async def test_basic_llm_call(self, log_dir):
        """A simple LLM call returns a non-empty string."""
        from src.llm_wrapper import llm_call

        t0 = time.monotonic()
        result = await asyncio.wait_for(
            llm_call(
                "テスト: 1+1の答えを1文字で返してください。",
                params={"temperature": 0.1, "max_tokens": 32},
            ),
            timeout=_API_TIMEOUT,
        )
        elapsed = time.monotonic() - t0

        assert isinstance(result, str)
        assert len(result) > 0
        # Must not be fallback
        assert "no_llm_available" not in result

        _write_log(log_dir, "basic_llm_call", {
            "result_length": len(result),
            "result_preview": result[:200],
            "elapsed_seconds": round(elapsed, 3),
        })


# ── Hierarchy 2: Perception Pathway ─────────────────────────────


class TestPerceptionPathway:
    """Verify parse_percept works with real Gemini responses."""

    @pytest.mark.asyncio
    async def test_parse_percept_text(self, orchestrator, log_dir):
        """parse_percept returns a valid Percept from text input."""
        from psyche.perception import parse_percept
        from psyche.state import Percept
        from src.llm_wrapper import llm_call

        t0 = time.monotonic()
        percept = await asyncio.wait_for(
            parse_percept(
                "こんにちは！今日はいい天気だね♪",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )
        elapsed = time.monotonic() - t0

        # Format checks only
        assert isinstance(percept, Percept)
        assert isinstance(percept.emotion, str)
        assert len(percept.emotion) > 0
        assert isinstance(percept.intent, str)
        assert len(percept.intent) > 0
        assert isinstance(percept.topics, list)
        assert isinstance(percept.emotion_valence, float)
        assert -1.0 <= percept.emotion_valence <= 1.0

        _write_log(log_dir, "parse_percept_text", {
            "emotion": percept.emotion,
            "intent": percept.intent,
            "topics": percept.topics,
            "emotion_valence": percept.emotion_valence,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_parse_percept_screen_description(self, orchestrator, log_dir):
        """parse_percept handles screen description text (vision-like input)."""
        from psyche.perception import parse_percept
        from psyche.state import Percept
        from src.llm_wrapper import llm_call

        screen_text = (
            "画面にはゲームのタイトル画面が表示されている。"
            "中央に大きなロゴがあり、「スタート」ボタンが光っている。"
            "背景は夜空で星が流れている。"
        )

        t0 = time.monotonic()
        percept = await asyncio.wait_for(
            parse_percept(
                screen_text,
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )
        elapsed = time.monotonic() - t0

        assert isinstance(percept, Percept)
        assert isinstance(percept.emotion, str)
        assert len(percept.emotion) > 0
        assert isinstance(percept.topics, list)

        _write_log(log_dir, "parse_percept_screen", {
            "emotion": percept.emotion,
            "intent": percept.intent,
            "topics": percept.topics,
            "emotion_valence": percept.emotion_valence,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_percept_structure_consistency(self, orchestrator):
        """Percept structure has all required fields with correct types."""
        from psyche.perception import parse_percept
        from src.llm_wrapper import llm_call

        percept = await asyncio.wait_for(
            parse_percept(
                "悲しいことがあったんだ...",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # Dict representation check
        percept_dict = percept.model_dump()
        required_keys = {"text", "meaning", "emotion", "intent", "topics",
                         "sentiment", "emotion_valence"}
        assert required_keys.issubset(set(percept_dict.keys()))
        assert isinstance(percept_dict["text"], str)
        assert isinstance(percept_dict["meaning"], str)
        assert isinstance(percept_dict["emotion"], str)
        assert isinstance(percept_dict["intent"], str)
        assert isinstance(percept_dict["topics"], list)
        assert isinstance(percept_dict["sentiment"], (int, float))
        assert isinstance(percept_dict["emotion_valence"], (int, float))


# ── Hierarchy 3: Expression Pathway ──────────────────────────────


class TestExpressionPathway:
    """Verify render_expression works with real Gemini responses."""

    @pytest.mark.asyncio
    async def test_render_expression_basic(self, orchestrator, log_dir):
        """render_expression returns valid text and meta from real API."""
        from psyche.expression import render_expression
        from src.llm_wrapper import llm_call

        policy = {
            "policy_label": "共感する",
            "rationale": "相手の気持ちに寄り添う",
            "text": "ふふっ♪",
        }

        t0 = time.monotonic()
        result = await asyncio.wait_for(
            render_expression(
                state=orchestrator.psyche,
                policy=policy,
                memory_snippet=[],
                persona=_PERSONA,
                llm_call_fn=llm_call,
                screen_context="ユーザーが楽しそうにゲームをしている画面",
            ),
            timeout=_API_TIMEOUT,
        )
        elapsed = time.monotonic() - t0

        # Format checks
        assert isinstance(result, dict)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert "meta" in result
        assert isinstance(result["meta"], dict)

        _write_log(log_dir, "render_expression_basic", {
            "text_length": len(result["text"]),
            "text_preview": result["text"][:200],
            "meta": result.get("meta", {}),
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_expression_text_splittable(self, orchestrator, log_dir):
        """Expression output text can be split into sentences."""
        from psyche.expression import render_expression
        from src.llm_wrapper import llm_call

        policy = {
            "policy_label": "励ます",
            "rationale": "相手を元気づける",
            "text": "大丈夫だよ♪",
        }

        result = await asyncio.wait_for(
            render_expression(
                state=orchestrator.psyche,
                policy=policy,
                memory_snippet=[],
                persona=_PERSONA,
                llm_call_fn=llm_call,
                screen_context="ユーザーが落ち込んでいる様子",
            ),
            timeout=_API_TIMEOUT,
        )

        text = result.get("text", "")
        assert isinstance(text, str)
        assert len(text) > 0

        sentences = _split_sentences(text)

        # At least 1 sentence should be produced
        assert len(sentences) >= 1
        for s in sentences:
            assert isinstance(s, str)
            assert len(s) > 0

        _write_log(log_dir, "expression_splittable", {
            "original_text": text[:200],
            "sentence_count": len(sentences),
            "sentences": [s[:100] for s in sentences],
        })


# ── Hierarchy 4: Integrated Pipeline ────────────────────────────


class TestIntegratedPipeline:
    """Verify full 2-call structure with orchestrator phase updates."""

    @pytest.mark.asyncio
    async def test_single_turn_pipeline(self, orchestrator, log_dir):
        """Single turn: perception -> psyche update -> policy -> expression."""
        from psyche.perception import parse_percept
        from psyche.expression import render_expression
        from psyche.silence_hesitation import is_silence_policy
        from psyche.state import Percept
        from src.llm_wrapper import llm_call

        screen_text = (
            "ユーザーがチャットで挨拶している画面。"
            "テキスト入力欄に「こんにちは」と書かれている。"
        )

        t0 = time.monotonic()

        # Step 1: Perception
        percept = await asyncio.wait_for(
            parse_percept(
                screen_text,
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )
        assert isinstance(percept, Percept)

        # Step 2: Psyche update
        orchestrator.post_response_update(percept, 1.0, "viewer")

        # Step 3: Policy selection
        policy = orchestrator.select_policy_dict(percept, [], "viewer")
        assert isinstance(policy, dict)
        assert "policy_label" in policy

        # Step 4: Expression (skip if silence)
        response_text = ""
        if not is_silence_policy(policy):
            enrichment = orchestrator.get_prompt_enrichment("viewer")
            result = await asyncio.wait_for(
                render_expression(
                    state=orchestrator.psyche,
                    policy=policy,
                    memory_snippet=[],
                    persona=_PERSONA,
                    llm_call_fn=llm_call,
                    screen_context=screen_text,
                    psyche_enrichment=enrichment,
                ),
                timeout=_API_TIMEOUT,
            )
            assert isinstance(result, dict)
            assert "text" in result
            assert isinstance(result["text"], str)
            response_text = result.get("text", "")

        elapsed = time.monotonic() - t0

        _write_log(log_dir, "single_turn_pipeline", {
            "percept_emotion": percept.emotion,
            "percept_intent": percept.intent,
            "policy_label": policy.get("policy_label", ""),
            "is_silence": is_silence_policy(policy),
            "response_preview": response_text[:100] if response_text else "(silence)",
            "tick_count": orchestrator.tick_count,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_multi_turn_pipeline(self, orchestrator, log_dir):
        """Multiple turns: verify phase updates complete without exceptions
        and psyche state changes across turns."""
        from psyche.perception import parse_percept
        from psyche.expression import render_expression
        from psyche.silence_hesitation import is_silence_policy
        from src.llm_wrapper import llm_call

        inputs = [
            "おはよう！今日はいい天気だね！",
            "最近ちょっと疲れちゃったんだよね...",
            "でも明日は楽しみなことがあるんだ！",
        ]

        turn_results = []
        states_before = []
        states_after = []

        for i, text in enumerate(inputs):
            t0 = time.monotonic()

            state_before = orchestrator.psyche.emotion_summary()
            states_before.append(state_before)

            # Perception
            percept = await asyncio.wait_for(
                parse_percept(
                    text,
                    llm_call_fn=llm_call,
                    state=orchestrator.psyche,
                ),
                timeout=_API_TIMEOUT,
            )

            # Psyche update
            orchestrator.post_response_update(percept, 2.0, "viewer")

            # Policy
            policy = orchestrator.select_policy_dict(percept, [], "viewer")

            # Expression (if not silence)
            response_text = ""
            if not is_silence_policy(policy):
                enrichment = orchestrator.get_prompt_enrichment("viewer")
                result = await asyncio.wait_for(
                    render_expression(
                        state=orchestrator.psyche,
                        policy=policy,
                        memory_snippet=[],
                        persona=_PERSONA,
                        llm_call_fn=llm_call,
                        screen_context=text,
                        psyche_enrichment=enrichment,
                    ),
                    timeout=_API_TIMEOUT,
                )
                response_text = result.get("text", "")

                # Notify self-action perception
                if response_text:
                    orchestrator.notify_self_output(
                        response_text=response_text,
                        policy_label=policy.get("policy_label", ""),
                    )

            state_after = orchestrator.psyche.emotion_summary()
            states_after.append(state_after)
            elapsed = time.monotonic() - t0

            turn_results.append({
                "turn": i + 1,
                "input_text": text,
                "percept_emotion": percept.emotion,
                "policy_label": policy.get("policy_label", ""),
                "response_preview": response_text[:100] if response_text else "(silence)",
                "state_before": state_before,
                "state_after": state_after,
                "elapsed_seconds": round(elapsed, 3),
            })

        # Tick count should have advanced
        assert orchestrator.tick_count >= len(inputs)

        # Log state change observation (soft check, not assertion)
        all_same = all(b == a for b, a in zip(states_before, states_after))

        _write_log(log_dir, "multi_turn_pipeline", {
            "total_turns": len(inputs),
            "final_tick_count": orchestrator.tick_count,
            "all_states_same": all_same,
            "turns": turn_results,
        })

    @pytest.mark.asyncio
    async def test_orchestrator_no_exception_on_phase_update(self, orchestrator, log_dir):
        """Orchestrator phase update completes without raising exceptions."""
        from psyche.perception import parse_percept
        from src.llm_wrapper import llm_call

        percept = await asyncio.wait_for(
            parse_percept(
                "テスト入力です",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # Phase update should not raise
        orchestrator.post_response_update(percept, 1.0, "viewer")

        # Policy selection should not raise
        policy = orchestrator.select_policy_dict(percept, [], "viewer")
        assert isinstance(policy, dict)

        # Enrichment generation should not raise
        enrichment = orchestrator.get_prompt_enrichment("viewer")
        assert isinstance(enrichment, str)

        _write_log(log_dir, "no_exception_phase_update", {
            "tick_count": orchestrator.tick_count,
            "enrichment_length": len(enrichment),
            "policy_label": policy.get("policy_label", ""),
        })


# ── Hierarchy 5: Text Input Pathway ─────────────────────────────


class TestTextInputPathway:
    """Verify text dialogue input pathway with real API."""

    @pytest.mark.asyncio
    async def test_text_input_pipeline(self, orchestrator, log_dir):
        """Text input -> process_text_input -> psyche update -> expression."""
        from psyche.perception import parse_percept
        from psyche.expression import render_expression
        from psyche.silence_hesitation import is_silence_policy
        from src.llm_wrapper import llm_call

        user_text = "今日はどんな一日だった？"

        t0 = time.monotonic()

        # Step 1: parse_percept
        percept = await asyncio.wait_for(
            parse_percept(
                user_text,
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # Step 2: process_text_input
        handoff = orchestrator.process_text_input(
            text=user_text,
            sender_id="test_user",
            conversation_id="e2e_test",
        )

        # Step 3: psyche update
        orchestrator.post_response_update(percept, 1.0, "text")

        # Step 4: policy
        policy = orchestrator.select_policy_dict(percept, [], "text")
        assert isinstance(policy, dict)
        assert "policy_label" in policy

        # Step 5: expression (if not silence)
        response_text = ""
        if not is_silence_policy(policy):
            enrichment = orchestrator.get_prompt_enrichment("text")
            result = await asyncio.wait_for(
                render_expression(
                    state=orchestrator.psyche,
                    policy=policy,
                    memory_snippet=[],
                    persona=_PERSONA,
                    llm_call_fn=llm_call,
                    screen_context=user_text,
                    psyche_enrichment=enrichment,
                ),
                timeout=_API_TIMEOUT,
            )
            response_text = result.get("text", "")

            if response_text:
                orchestrator.notify_self_output(
                    response_text=response_text,
                    policy_label=policy.get("policy_label", ""),
                )

        elapsed = time.monotonic() - t0

        _write_log(log_dir, "text_input_pipeline", {
            "user_text": user_text,
            "percept_emotion": percept.emotion,
            "percept_intent": percept.intent,
            "policy_label": policy.get("policy_label", ""),
            "response_preview": response_text[:200] if response_text else "(silence)",
            "tick_count": orchestrator.tick_count,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_text_input_percept_structure(self, orchestrator):
        """Text input produces a Percept with correct structure via real API."""
        from psyche.perception import parse_percept
        from psyche.state import Percept
        from src.llm_wrapper import llm_call

        percept = await asyncio.wait_for(
            parse_percept(
                "嬉しいニュースがあるの！聞いて！",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        assert isinstance(percept, Percept)
        assert isinstance(percept.emotion, str)
        assert len(percept.emotion) > 0
        assert isinstance(percept.intent, str)
        assert len(percept.intent) > 0
        assert isinstance(percept.topics, list)
        assert isinstance(percept.emotion_valence, float)
        assert -1.0 <= percept.emotion_valence <= 1.0
        assert isinstance(percept.meaning, str)
        assert len(percept.meaning) > 0


# ── Hierarchy 6: State Contamination Guard ───────────────────────


class TestStateContaminationGuard:
    """Verify that e2e tests do not contaminate persistent state."""

    def test_no_save_file_created(self, temp_data):
        """No psyche_state.json or similar save files should be created."""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch = PsycheOrchestrator(memory_count=0, data_dir=temp_data)

        # Run a tick
        percept = Percept(text="test", emotion="neutral", intent="unknown")
        orch.post_response_update(percept, 1.0, "viewer")

        # Check: no save file should exist in temp_data
        save_files = list(temp_data.glob("psyche_state*.json"))
        assert len(save_files) == 0, (
            f"Unexpected save files found: {save_files}"
        )

    def test_orchestrator_independent_instances(self, temp_data):
        """Two orchestrator instances are fully independent."""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch1 = PsycheOrchestrator(memory_count=0, data_dir=temp_data)
        orch2 = PsycheOrchestrator(memory_count=0, data_dir=temp_data)

        percept = Percept(text="test input", emotion="happy", intent="greeting",
                          emotion_valence=0.7)
        orch1.post_response_update(percept, 1.0, "viewer")

        # orch1 should have advanced, orch2 should still be at 0
        assert orch1.tick_count >= 1
        assert orch2.tick_count == 0
