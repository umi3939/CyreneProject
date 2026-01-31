"""
psyche/attachment_manager.py - Attachment Pillar Manager

Manages emotional bonds with interaction partners.
All functions are pure (immutable pattern).
"""

from __future__ import annotations

from .pillars import AttachmentState


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def update_bond(
    state: AttachmentState,
    partner_id: str,
    event_type: str = "positive",
    intensity: float = 0.5,
) -> AttachmentState:
    """Update a bond strength based on an interaction event.

    Args:
        state: Current attachment state.
        partner_id: Identifier for the interaction partner.
        event_type: "positive" or "negative".
        intensity: Event intensity (0.0 - 1.0).

    Returns:
        New AttachmentState with updated bond.
    """
    bonds = dict(state.bonds)
    current = bonds.get(partner_id, 0.0)

    if event_type == "positive":
        bonds[partner_id] = _clamp01(current + intensity * 0.1)
    else:
        bonds[partner_id] = _clamp01(current - intensity * 0.15)

    last = dict(state.last_interaction)
    # Timestamp is managed by caller; we just track the partner
    # (actual timestamp set externally)

    return AttachmentState(
        bonds=bonds,
        last_interaction=last,
        risk=_calc_risk(bonds),
    )


def apply_daily_decay(
    state: AttachmentState, days_elapsed: float = 1.0
) -> AttachmentState:
    """Decay all bonds by time passage.

    Each bond is multiplied by 0.98^days_elapsed.
    """
    decay_factor = 0.98 ** days_elapsed
    bonds = {pid: _clamp01(v * decay_factor) for pid, v in state.bonds.items()}
    return AttachmentState(
        bonds=bonds,
        last_interaction=dict(state.last_interaction),
        risk=_calc_risk(bonds),
    )


def get_top_partners(
    state: AttachmentState, n: int = 3
) -> list[tuple[str, float]]:
    """Return the top-n partners sorted by bond strength (descending)."""
    sorted_bonds = sorted(state.bonds.items(), key=lambda x: x[1], reverse=True)
    return sorted_bonds[:n]


def calc_attachment_risk(state: AttachmentState) -> float:
    """Compute attachment risk (0.0 - 1.0).

    High risk when:
    - Highest bond < 0.3 (no strong connections)
    - No bonds at all
    """
    return _calc_risk(state.bonds)


def _calc_risk(bonds: dict[str, float]) -> float:
    if not bonds:
        return 0.7  # No bonds → high risk
    max_bond = max(bonds.values())
    if max_bond < 0.3:
        return 0.6  # Weak bonds → moderate-high risk
    if max_bond < 0.5:
        return 0.3  # Moderate bonds
    return max(0.0, 0.2 - max_bond * 0.1)  # Strong bonds → low risk
