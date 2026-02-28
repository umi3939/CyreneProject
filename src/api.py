"""
src/api.py - FastAPI service for the Cyrene chatbot.

**Architecture**: Uses PsycheOrchestrator for the full psyche pipeline.
- All 70+ orchestrator systems (Phase 1-35, enrichment, save/load) are active
- Gemini is ONLY used for parse_percept (auxiliary) and render_expression (voice)
- All decisions, state updates, and policy selection are LOCAL via orchestrator

Pipeline: parse_percept → orchestrator.process_text_input → orchestrator.post_response_update
          → recall_with_mood → orchestrator.select_policy_dict → render_expression

Usage::

    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.logging_config import configure_logging, is_debug_enabled

# Configure logging on module load
configure_logging()

# Orchestrator (full psyche pipeline)
from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept
from psyche.perception import parse_percept
from psyche.expression import render_expression
from psyche.memory_link import recall_with_mood
from psyche.silence_hesitation import is_silence_policy

# src managers (persistence)
from src.llm_wrapper import llm_call, llm_call_with_system
from src.memory_manager import MemoryManager
from src.attachment_manager import AttachmentManager
from src.identity_manager import IdentityManager
from src.projection_manager import ProjectionManager
from src.state_manager import StateManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Cyrene AI Chatbot", version="3.0.0")

# ── Singletons (initialised at import time) ────────────────────
memory_mgr = MemoryManager(llm_call=llm_call)
attachment_mgr = AttachmentManager()
identity_mgr = IdentityManager()
projection_mgr = ProjectionManager()
state_mgr = StateManager()

# PsycheOrchestrator: full pipeline with all 70+ systems
_orchestrator = PsycheOrchestrator(
    memory_count=memory_mgr.count if hasattr(memory_mgr, 'count') else 0,
)
_orchestrator.load()

# Tracking for psyche delta time
_last_psyche_update = time.monotonic()

# Load persona for expression rendering
DATA_DIR = Path(__file__).parent.parent / "data"
PERSONA_FILE = DATA_DIR / "persona.json"

def _load_persona() -> dict:
    if PERSONA_FILE.exists():
        try:
            return json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"name": "キュレネ", "tone": "romantic, sweet", "style_rules": {}}

_persona = _load_persona()


# ── Request / Response schemas ─────────────────────────────────

class RespondRequest(BaseModel):
    user_id: str = "default_user"
    text: str


class RespondResponse(BaseModel):
    text: str
    meta: dict[str, Any]
    updated_state: dict[str, Any]


# ── Main endpoint ──────────────────────────────────────────────

@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest):
    """Execute one full conversation turn using PsycheOrchestrator pipeline.

    This routes through the orchestrator (70+ systems, Phase 1-35, enrichment,
    save/load) instead of calling psyche functions directly.

    Pipeline mirrors brain.py think_text:
    1. parse_percept (Gemini auxiliary)
    2. orchestrator.process_text_input (text dialogue input processing)
    3. orchestrator.post_response_update (full psyche tick)
    4. recall_with_mood (memory retrieval)
    5. orchestrator.select_policy_dict (policy selection with all biases)
    6. silence check
    7. render_expression (Gemini voice with enrichment)
    8. orchestrator.notify_self_output (self-action perception)
    9. orchestrator.save (periodic persistence)
    """
    global _last_psyche_update

    user_id = req.user_id
    logger.debug("[TURN START] user_id=%s", user_id)

    # Phase 1: parse_percept (Gemini auxiliary -- JSON extraction only)
    percept = await parse_percept(
        req.text,
        llm_call_fn=llm_call,
        state=_orchestrator.psyche,
    )
    logger.debug("[PERCEPT] emotion=%s, intent=%s, valence=%.2f",
                 percept.emotion, percept.intent, percept.emotion_valence)

    # Phase 2: text dialogue input processing (orchestrator)
    _orchestrator.process_text_input(
        text=req.text,
        sender_id=user_id,
        conversation_id=user_id,
    )

    # Phase 3: psyche update (full orchestrator tick -- all 70+ systems)
    now = time.monotonic()
    delta = now - _last_psyche_update
    _last_psyche_update = now
    _orchestrator.post_response_update(percept, delta, user_id)
    logger.debug("[PSYCHE TICK] tick=%d complete", _orchestrator.tick_count)

    # Phase 4: recall memories
    recall_percept = Percept(text=req.text)
    memories = await recall_with_mood(
        recall_percept, _orchestrator.psyche, memory_mgr, top_k=3
    )
    _orchestrator.set_recalled_memories(memories)
    logger.debug("[MEMORY] Recalled %d memories", len(memories or []))

    # Phase 5: policy selection (with all orchestrator biases)
    policy = _orchestrator.select_policy_dict(
        percept, memories or [], user_id
    )
    logger.debug("[POLICY] selected: %s", policy.get("policy_label", "unknown"))

    # Phase 6: silence check
    if is_silence_policy(policy):
        logger.debug("Psyche chose silence")
        # Return minimal response for silence
        psyche_state = _orchestrator.psyche
        updated_state = _psyche_state_to_dict(psyche_state)
        filtered_state = _filter_state_for_production(updated_state)
        return RespondResponse(
            text="",
            meta={"emotion_label": percept.emotion},
            updated_state=filtered_state,
        )

    # Phase 7: render expression (Gemini voice with psyche enrichment)
    enrichment = _orchestrator.get_prompt_enrichment(user_id)
    expr_result = await render_expression(
        state=_orchestrator.psyche,
        policy=policy,
        memory_snippet=memories or [],
        persona=_persona,
        llm_call_fn=llm_call,
        screen_context=req.text,
        psyche_enrichment=enrichment,
    )
    response_text = expr_result.get("text", "...")
    meta = expr_result.get("meta", {})
    logger.debug("[RESPONSE] %s", response_text[:50] + "..." if len(response_text) > 50 else response_text)

    # Phase 8: notify self-action perception
    if response_text:
        _orchestrator.notify_self_output(
            response_text=response_text,
            policy_label=policy.get("policy_label", ""),
        )

    # Side effects: memory save
    memory_mgr.maybe_save(
        req.text,
        response_text,
        _orchestrator.psyche.to_dict(),
        importance=_estimate_importance(percept),
        involves_attachment=True,
    )

    # Side effects: attachment update
    # NOTE: src層の管理データ(data/example_attachments.json)であり、
    # psyche内部のattachment(psyche/pillars.py AttachmentState)とは独立。
    # psyche内部のattachmentはorchestrator Phase 3で感情バレンスに基づき
    # 動的に更新される。この呼び出しはpsycheの内部状態に影響しない。
    attachment_mgr.update_bond(user_id, "partner", positive=(percept.emotion_valence >= 0.0), importance=3)

    # Build response state from orchestrator
    psyche_state = _orchestrator.psyche
    updated_state = _psyche_state_to_dict(psyche_state)

    logger.debug("[TURN END] fear_level=%.4f, mood=%.2f",
                 psyche_state.fear_level, psyche_state.mood.valence)

    # Filter response for production
    filtered_meta = _filter_meta_for_production(meta, percept)
    filtered_state = _filter_state_for_production(updated_state)

    return RespondResponse(text=response_text, meta=filtered_meta, updated_state=filtered_state)


# ── Lifecycle events ──────────────────────────────────────────

@app.on_event("shutdown")
async def _on_shutdown():
    """Persist orchestrator state on server shutdown."""
    try:
        _orchestrator.save()
        logger.info("Orchestrator state saved on shutdown")
    except Exception as e:
        logger.error("Failed to save orchestrator state: %s", e)


# ── Management endpoints ───────────────────────────────────────

@app.get("/state/{user_id}")
async def get_state(user_id: str):
    return state_mgr.get_state(user_id)


@app.get("/memories")
async def list_memories():
    return memory_mgr.memories


@app.get("/projections")
async def list_projections():
    return projection_mgr.goals


@app.post("/projections")
async def add_projection(body: dict):
    return projection_mgr.add_goal(body.get("description", ""))


@app.get("/identity")
async def get_identity():
    return identity_mgr.state


@app.get("/psyche_tick_count")
async def get_psyche_tick_count():
    """Get current orchestrator tick count (for monitoring)."""
    return {"tick_count": _orchestrator.tick_count}


# ── Helpers ────────────────────────────────────────────────────

def _psyche_state_to_dict(state) -> dict:
    """Convert PsycheState to persistence dict (backward compatible)."""
    d = state.to_dict()
    # Add 5-dim emotions for backward compat with existing tests
    emo = state.emotions.as_dict()
    d["emotions"] = {
        "joy": emo.get("joy", 0.0),
        "sad": emo.get("sorrow", 0.0),
        "fear": emo.get("fear", 0.0),
        "anger": emo.get("anger", 0.0),
        "calm": max(0.0, 1.0 - max(emo.get("joy", 0), emo.get("sorrow", 0), emo.get("fear", 0), emo.get("anger", 0))),
    }
    return d


def _estimate_importance(percept: Percept) -> int:
    """Estimate event importance from percept."""
    importance = 3
    if abs(percept.emotion_valence) > 0.6:
        importance = 4
    if percept.intent in ("sharing", "complaint"):
        importance = max(importance, 4)
    return importance


def _filter_meta_for_production(meta: dict, percept: Percept) -> dict:
    """Filter meta information for production response.

    In production: only return minimal info (emotion_label).
    In debug mode: return full meta.
    """
    if is_debug_enabled():
        return meta

    # Production: minimal meta only
    return {
        "emotion_label": percept.emotion,
    }


def _filter_state_for_production(state: dict) -> dict:
    """Filter state information for production response.

    In production: hide psychological values, responsibility, decision rationale.
    In debug mode: return full state.
    """
    if is_debug_enabled():
        return state

    # Production: minimal state only (no internal psychological values)
    return {
        "last_updated": state.get("last_updated", ""),
    }
