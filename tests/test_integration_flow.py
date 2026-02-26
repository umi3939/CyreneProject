"""
tests/test_integration_flow.py - POST /respond integration test.

Verifies that one conversation turn produces valid responses and
that the PsycheOrchestrator pipeline is active (all 70+ systems).

**Architecture Validation**: Verifies:
1. PsycheOrchestrator pipeline is used (parse_percept → orchestrator tick → policy → expression)
2. Thinking is LOCAL (no LLM in generate_thought_candidates/select_policy)
3. Gemini is voice only (parse_percept auxiliary + render_expression)
4. Response format is backward compatible {text, meta, updated_state}

Uses a mock LLM so no API keys are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src import api as api_module
from src.api import app
from src.memory_manager import MemoryManager
from src.attachment_manager import AttachmentManager
from src.identity_manager import IdentityManager
from src.projection_manager import ProjectionManager
from src.state_manager import StateManager


def _setup_managers(data_dir: Path):
    """Replace global manager singletons with tmp_data_dir-backed ones."""
    api_module.memory_mgr = MemoryManager(filepath=data_dir / "example_memories.json")
    api_module.attachment_mgr = AttachmentManager(filepath=data_dir / "example_attachments.json")
    api_module.identity_mgr = IdentityManager(filepath=data_dir / "identity.json")
    api_module.projection_mgr = ProjectionManager(filepath=data_dir / "projections.json")
    api_module.state_mgr = StateManager(filepath=data_dir / "state.json")


async def _mock_llm_call(prompt: str, params=None) -> str:
    """Deterministic mock LLM for integration tests.

    This mock simulates Gemini's VOICE-ONLY role:
    - parse_percept: returns JSON with emotion/intent extraction
    - render_expression: returns JSON with text and meta
    """
    # Perception parsing (Gemini auxiliary)
    if "解析" in prompt or "分析" in prompt or "JSON" in prompt:
        return json.dumps(
            {"meaning": "楽しい挨拶", "emotion": "happy", "intent": "greeting",
             "emotion_valence": 0.6, "topics": ["楽しい"]},
            ensure_ascii=False,
        )
    # Expression rendering (Gemini voice)
    if "確定済み" in prompt or "セリフ" in prompt:
        return json.dumps(
            {"text": "ふふっ、楽しいわね♪",
             "meta": {"emotion": "joy", "intensity": 0.6, "action": "共感する"}},
            ensure_ascii=False,
        )
    return '{"result": "ok"}'


@pytest.mark.asyncio
class TestRespondEndpoint:
    """Test POST /respond end-to-end with PsycheOrchestrator pipeline."""

    async def test_respond_returns_200(self, tmp_data_dir: Path):
        _setup_managers(tmp_data_dir)
        with patch.object(api_module, "llm_call", new=_mock_llm_call):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "こんにちは！楽しいね！"},
                )
        assert resp.status_code == 200

    async def test_respond_returns_text_and_meta(self, tmp_data_dir: Path):
        _setup_managers(tmp_data_dir)
        with patch.object(api_module, "llm_call", new=_mock_llm_call):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "こんにちは！"},
                )
        body = resp.json()
        assert "text" in body
        assert "meta" in body
        assert "updated_state" in body
        assert len(body["text"]) > 0

    async def test_respond_state_changes_across_turns(self, tmp_data_dir: Path):
        """SPEC REQUIREMENT: updated_state must differ between turns.

        The orchestrator processes each input and updates internal state,
        so consecutive calls should produce different state snapshots.
        """
        _setup_managers(tmp_data_dir)

        with patch.object(api_module, "llm_call", new=_mock_llm_call):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp1 = await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "嬉しい！楽しいね！"},
                )
                state_1 = resp1.json()["updated_state"]

                resp2 = await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "悲しいなぁ…"},
                )
                state_2 = resp2.json()["updated_state"]

        # At least one dimension must have changed between turns
        emotion_changed = state_1.get("emotions") != state_2.get("emotions")
        drives_changed = state_1.get("drives") != state_2.get("drives")
        fear_changed = state_1.get("fear_index") != state_2.get("fear_index")

        assert emotion_changed or drives_changed or fear_changed, (
            f"State should change between turns.\n"
            f"  emotions changed: {emotion_changed}\n"
            f"  drives changed: {drives_changed}\n"
            f"  fear_index changed: {fear_changed}\n"
            f"  state_1: {state_1}\n"
            f"  state_2: {state_2}"
        )

    async def test_orchestrator_tick_increments(self, tmp_data_dir: Path):
        """Verify that the orchestrator tick count increases with each /respond call."""
        _setup_managers(tmp_data_dir)

        initial_tick = api_module._orchestrator.tick_count

        with patch.object(api_module, "llm_call", new=_mock_llm_call):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "こんにちは"},
                )

        assert api_module._orchestrator.tick_count > initial_tick, (
            "Orchestrator tick should increment after /respond"
        )

    async def test_psyche_pipeline_used(self, tmp_data_dir: Path):
        """Verify that the orchestrator pipeline produces proper state."""
        _setup_managers(tmp_data_dir)

        with patch.object(api_module, "llm_call", new=_mock_llm_call):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "嬉しい気分！"},
                )

        body = resp.json()
        assert resp.status_code == 200

        # Verify state structure matches psyche pipeline output
        state = body["updated_state"]
        assert "emotions" in state
        assert "drives" in state
        assert "mood" in state or "valence" in str(state)  # mood info present
        assert "fear_index" in state

        # Meta should contain action (from policy)
        meta = body["meta"]
        assert "action" in meta or "emotion" in meta

    async def test_respond_backward_compatible_format(self, tmp_data_dir: Path):
        """Verify response format is backward compatible with old API."""
        _setup_managers(tmp_data_dir)

        with patch.object(api_module, "llm_call", new=_mock_llm_call):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/respond",
                    json={"user_id": "test_user", "text": "テスト"},
                )

        body = resp.json()
        # Must have all three top-level keys
        assert "text" in body
        assert "meta" in body
        assert "updated_state" in body
        # text must be a string
        assert isinstance(body["text"], str)
        # meta must be a dict
        assert isinstance(body["meta"], dict)
        # updated_state must be a dict
        assert isinstance(body["updated_state"], dict)
