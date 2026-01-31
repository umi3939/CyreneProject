"""
psyche/projection_manager.py - Projection Pillar Manager

Manages future goals and purpose. All functions are pure
(immutable pattern): they return new state objects.
"""

from __future__ import annotations

import uuid

from .pillars import ProjectionState


def add_goal(state: ProjectionState, description: str) -> ProjectionState:
    """Add a new goal and return updated ProjectionState."""
    new_goal = {
        "id": uuid.uuid4().hex[:8],
        "description": description,
        "progress": 0.0,
        "status": "active",
    }
    goals = list(state.goals) + [new_goal]
    return ProjectionState(
        goals=goals,
        risk=_calc_risk(goals),
    )


def update_goal_progress(
    state: ProjectionState, goal_id: str, delta: float
) -> ProjectionState:
    """Update progress on a goal by delta and return new state.

    Progress is clamped to [0.0, 1.0]. If progress reaches 1.0,
    status is set to "completed".
    """
    goals = []
    for g in state.goals:
        if g.get("id") == goal_id:
            new_progress = max(0.0, min(1.0, g.get("progress", 0.0) + delta))
            updated = dict(g)
            updated["progress"] = new_progress
            if new_progress >= 1.0:
                updated["status"] = "completed"
            goals.append(updated)
        else:
            goals.append(dict(g))

    return ProjectionState(
        goals=goals,
        risk=_calc_risk(goals),
    )


def remove_goal(state: ProjectionState, goal_id: str) -> ProjectionState:
    """Remove a goal by ID and return new state."""
    goals = [dict(g) for g in state.goals if g.get("id") != goal_id]
    return ProjectionState(
        goals=goals,
        risk=_calc_risk(goals),
    )


def reset(state: ProjectionState) -> ProjectionState:
    """Clear all goals (for testing). Returns empty ProjectionState."""
    return ProjectionState(goals=[], risk=_calc_risk([]))


def calc_projection_risk(state: ProjectionState) -> float:
    """Compute projection risk (0.0 - 1.0)."""
    return _calc_risk(state.goals)


def _calc_risk(goals: list[dict]) -> float:
    """High risk when no goals or all goals are stalled."""
    if not goals:
        return 0.7  # No purpose → high risk

    active = [g for g in goals if g.get("status") == "active"]
    if not active:
        return 0.5  # All completed/removed → moderate risk

    # Check for stagnation (no progress on any active goal)
    progressing = [g for g in active if g.get("progress", 0.0) > 0.0]
    if not progressing:
        return 0.6  # All stalled → moderate-high risk

    return 0.1  # Active goals with progress → low risk
