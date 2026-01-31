"""
src/api.py - FastAPI service for the Cyrene chatbot.

**Architecture**: Uses psyche pipeline with strict Local Brain / Gemini Voice separation.
- Gemini is ONLY used for parse_percept (auxiliary) and render_expression (voice)
- All decisions, state updates, and policy selection are LOCAL

Pipeline: parse_percept → react → recall_by_mood → generate_thought_candidates/select_policy → render_expression

Usage::

    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.logging_config import configure_logging, is_debug_enabled

# Configure logging on module load
configure_logging()

# psyche pipeline (LOCAL logic)
from psyche import (
    PsycheState,
    Percept,
    parse_percept,
    react,
    recall_by_mood,
    generate_thought_candidates,
    select_policy,
    render_expression,
    compute_fear_index,
    ResponsibilityManager,
)

# src managers (persistence)
from src.llm_wrapper import llm_call, llm_call_with_system
from src.memory_manager import MemoryManager
from src.attachment_manager import AttachmentManager
from src.identity_manager import IdentityManager
from src.projection_manager import ProjectionManager
from src.state_manager import StateManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Cyrene AI Chatbot", version="2.0.0")

# ── Singletons (initialised at import time) ────────────────────
memory_mgr = MemoryManager(llm_call=llm_call)
attachment_mgr = AttachmentManager()
identity_mgr = IdentityManager()
projection_mgr = ProjectionManager()
state_mgr = StateManager()
responsibility_mgr = ResponsibilityManager()

# Load persona for expression rendering
import json
from pathlib import Path
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
    """Execute one full conversation turn using psyche pipeline.

    Per integration_instructions.md, the order is:
    1. 入力（Percept）を受け取る
    2. 次の入力時に、前回のDecision Recordを評価（責任の重みを更新）
    3. 更新された責任・感情・喪失をPsycheStateに反映
    4. 現在の心理状態・責任影響を考慮して候補Policyを生成
    5. Policyを1つ確定する
    6. 確定した瞬間に、責任の決定記録を記録
    7. 応答を生成・返却
    """
    user_id = req.user_id
    logger.debug("[TURN START] user_id=%s", user_id)

    # 0. Load current state and convert to PsycheState
    state_dict = state_mgr.get_state(user_id)
    psyche_state = _dict_to_psyche_state(state_dict)

    # Calculate time delta for decay
    try:
        last = datetime.fromisoformat(state_dict.get("last_updated", "2026-01-01T00:00:00"))
        delta_seconds = min(3600.0, max(0.0, (datetime.now() - last).total_seconds()))
    except ValueError:
        delta_seconds = 1.0

    # 1. parse_percept (Gemini auxiliary — JSON extraction only)
    percept = await parse_percept(req.text, llm_call, psyche_state)
    logger.debug("[PERCEPT] emotion=%s, intent=%s, valence=%.2f",
                 percept.emotion, percept.intent, percept.emotion_valence)

    # 2. Evaluate previous decision's outcome based on user's current reaction
    #    (責任評価は必ず「次の入力」で行う)
    _evaluate_previous_decision(user_id, percept)

    # 3. Get responsibility influence AFTER evaluation (updated weights)
    #    (更新された責任をPsycheStateに反映)
    responsibility_influence = responsibility_mgr.get_influence(user_id)
    logger.debug(
        "[RESPONSIBILITY] caution=%.4f, empathy=%.4f, anxiety=%.4f, fear_amp=%.4f",
        responsibility_influence.caution_bias,
        responsibility_influence.empathy_bias,
        responsibility_influence.anxiety_baseline,
        responsibility_influence.fear_amplification,
    )

    # 4. react (LOCAL — update emotions, drives, mood with responsibility influence)
    psyche_state = react(
        percept, psyche_state, delta_time=delta_seconds,
        responsibility_influence=responsibility_influence,
    )

    # 3. recall_by_mood (LOCAL — mood-congruent memory retrieval)
    recalled = await recall_by_mood(percept, psyche_state, memory_mgr, top_k=3)
    logger.debug("[MEMORY] Recalled %d memories", len(recalled))

    # 5. generate_thought_candidates (現在の心理状態・責任影響を考慮して候補Policyを生成)
    candidates = generate_thought_candidates(
        psyche_state, percept, recalled, responsibility_influence,
    )
    logger.debug(
        "[POLICY CANDIDATES] %s",
        [(c["policy_label"], round(c.get("_score", 0), 2)) for c in candidates],
    )

    # 6. select_policy (Policyを1つ確定する — 判断は1回のみ)
    policy = select_policy(candidates, psyche_state, responsibility_influence)
    logger.debug("[POLICY] selected: %s", policy.get("policy_label", "unknown"))

    # 8. render_expression (応答を生成 — Gemini voice only)
    expression = await render_expression(
        psyche_state, policy, recalled, _persona, llm_call
    )
    response_text = expression.get("text", "...")
    meta = expression.get("meta", {})
    logger.debug("[RESPONSE] %s", response_text[:50] + "..." if len(response_text) > 50 else response_text)

    # 9. Side effects (LOCAL)
    # 9a. Maybe save memory
    memory_mgr.maybe_save(
        req.text,
        response_text,
        psyche_state.to_dict(),
        importance=_estimate_importance(percept),
        involves_attachment=True,
    )

    # 9b. Update attachment bond
    attachment_mgr.update_bond(user_id, "partner", positive=True, importance=3)

    # 9c. Record the decision (確定した瞬間に責任の決定記録を記録)
    decision_context = {
        "target_partner": user_id,
        "emotional_state": psyche_state.mood.valence_label,
        "fear_level": psyche_state.fear_level,
        "involves_attachment": True,
    }
    _, decision_id = responsibility_mgr.record_decision(user_id, policy, decision_context)
    logger.debug(
        "[DECISION RECORD CREATED] id=%s, policy=%s, fear=%.2f, emotion=%s",
        decision_id,
        policy.get("policy_label", "unknown"),
        psyche_state.fear_level,
        psyche_state.mood.valence_label,
    )

    # 9d. Recalculate fear_index from pillar risks
    fear_index = compute_fear_index(
        identity_risk=identity_mgr.get_risk(),
        attachment_risk=attachment_mgr.get_risk(user_id),
        continuity_risk=_continuity_risk(memory_mgr.count),
        projection_risk=projection_mgr.get_risk(),
    )

    # 9e. Update psyche_state with new fear_index
    psyche_state = PsycheState(
        emotions=psyche_state.emotions,
        drives=psyche_state.drives,
        mood=psyche_state.mood,
        identity=psyche_state.identity,
        attachment=psyche_state.attachment,
        continuity=psyche_state.continuity,
        projection=psyche_state.projection,
        fear_index=fear_index,
        loss_aversion=psyche_state.loss_aversion,
        last_updated=datetime.now().isoformat(timespec="seconds"),
    )

    # 9f. Persist state
    updated_state = _psyche_state_to_dict(psyche_state)
    state_mgr.set_state(user_id, updated_state)

    logger.debug("[TURN END] fear_level=%.4f, mood=%.2f", psyche_state.fear_level, psyche_state.mood.valence)

    # Filter response for production (hide internal psychological values)
    filtered_meta = _filter_meta_for_production(meta, percept)
    filtered_state = _filter_state_for_production(updated_state)

    return RespondResponse(text=response_text, meta=filtered_meta, updated_state=filtered_state)


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


@app.get("/responsibility/{user_id}")
async def get_responsibility(user_id: str):
    """Get responsibility state summary for a user."""
    return responsibility_mgr.get_summary(user_id)


# ── Helpers ────────────────────────────────────────────────────

def _evaluate_previous_decision(user_id: str, percept: Percept) -> None:
    """Evaluate the previous decision's outcome based on current user reaction.

    結果を観測してから責任を評価する。
    ユーザーの反応（percept）から、前回の判断の結果を推定する。
    責任評価は必ず「次の入力」で行う。
    """
    # Get the most recent unevaluated decision
    resp_state = responsibility_mgr.get_state(user_id)
    unevaluated = [
        d for d in resp_state.recent_decisions
        if not d.get("evaluated", False)
    ]

    if not unevaluated:
        logger.debug("[DECISION EVAL] No unevaluated decisions")
        return

    # Evaluate the most recent one
    decision = unevaluated[-1]
    decision_id = decision.get("id")
    if not decision_id:
        return

    # Infer outcome from percept
    outcome = _infer_outcome_from_percept(percept)
    logger.debug(
        "[DECISION EVAL START] id=%s, policy=%s, inferred_reaction=%s",
        decision_id,
        decision.get("policy_label", "unknown"),
        outcome.get("user_reaction", "unknown"),
    )

    responsibility_mgr.evaluate_outcome(user_id, decision_id, outcome)

    # Log updated responsibility state
    updated_state = responsibility_mgr.get_state(user_id)
    logger.debug(
        "[DECISION EVAL DONE] total_weight=%.4f, harm=%.4f, confidence=%.4f",
        updated_state.total_weight,
        updated_state.accumulated_harm,
        updated_state.accumulated_confidence,
    )


def _infer_outcome_from_percept(percept: Percept) -> dict:
    """Infer decision outcome from user's reaction (percept).

    ユーザーの反応から、前回の判断の結果を推定する。
    """
    valence = percept.emotion_valence
    intent = percept.intent
    emotion = percept.emotion

    # Determine user reaction
    if valence > 0.3 or emotion in ("happy", "loving"):
        user_reaction = "positive"
    elif valence < -0.3 or emotion in ("angry", "sad"):
        user_reaction = "negative"
    elif emotion == "scared" or intent == "complaint":
        user_reaction = "confused"
    else:
        user_reaction = "neutral"

    # Estimate relationship delta
    relationship_delta = 0.0
    if user_reaction == "positive":
        relationship_delta = 0.1
    elif user_reaction == "negative":
        relationship_delta = -0.15
    elif user_reaction == "confused":
        relationship_delta = -0.05

    # Expectation gap (higher for negative/confused)
    expectation_gap = 0.0
    if user_reaction in ("negative", "confused"):
        expectation_gap = abs(valence) * 0.5

    return {
        "user_reaction": user_reaction,
        "relationship_delta": relationship_delta,
        "expectation_gap": expectation_gap,
    }

def _dict_to_psyche_state(d: dict) -> PsycheState:
    """Convert legacy 5-dim state dict to PsycheState."""
    return PsycheState.from_dict(d)


def _psyche_state_to_dict(state: PsycheState) -> dict:
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


def _continuity_risk(memory_count: int) -> float:
    """Compute continuity risk based on memory count."""
    if memory_count < 5:
        return 0.6
    if memory_count < 20:
        return 0.3
    return 0.1


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
