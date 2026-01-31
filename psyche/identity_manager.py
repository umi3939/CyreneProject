"""
psyche/identity_manager.py - Identity Pillar Manager

Manages core trait identity, change proposals, and identity risk.
All functions are pure (immutable pattern): they return new state
objects rather than mutating in place.
"""

from __future__ import annotations

from .pillars import IdentityState


def propose_identity_change(state: IdentityState, change: dict) -> dict:
    """Evaluate a proposed identity change against core traits.

    Args:
        state: Current identity state.
        change: Dict with at least ``trait`` and ``new_value`` keys.

    Returns:
        Dict with ``change``, ``requires_confirmation`` (bool), and ``reason``.
    """
    trait = change.get("trait", "")
    conflicts = trait in state.core_traits
    return {
        "change": change,
        "requires_confirmation": conflicts,
        "reason": (
            f"'{trait}' is a core trait - change needs confirmation"
            if conflicts
            else "no conflict with core traits"
        ),
    }


def apply_identity_change(state: IdentityState, change: dict) -> IdentityState:
    """Apply a confirmed identity change and return a new IdentityState.

    The change is removed from pending_changes and trait_confidence
    is updated for the affected trait.
    """
    trait = change.get("trait", "")
    new_value = change.get("new_value", "")

    new_traits = list(state.core_traits)
    if trait and trait not in new_traits:
        new_traits.append(trait)

    new_confidence = dict(state.trait_confidence)
    new_confidence[trait] = new_confidence.get(trait, 0.5)

    # Remove this change from pending if present
    new_pending = [p for p in state.pending_changes if p != change]

    return IdentityState(
        core_traits=new_traits,
        trait_confidence=new_confidence,
        pending_changes=new_pending,
        risk=calc_identity_risk_from_values(new_pending, new_confidence),
    )


def calc_identity_risk(state: IdentityState) -> float:
    """Compute identity risk (0.0 - 1.0).

    High risk when:
    - Many pending (unresolved) changes
    - Low average trait confidence
    """
    return calc_identity_risk_from_values(
        state.pending_changes, state.trait_confidence
    )


def calc_identity_risk_from_values(
    pending: list, confidence: dict[str, float]
) -> float:
    """Pure helper for risk calculation."""
    # Pending changes contribute up to 0.5 risk
    pending_risk = min(len(pending) * 0.15, 0.5)

    # Low confidence contributes up to 0.5 risk
    if confidence:
        avg_conf = sum(confidence.values()) / len(confidence)
        conf_risk = max(0.0, 0.5 - avg_conf) * 1.0  # low conf → higher risk
    else:
        conf_risk = 0.3  # no traits defined yet → moderate risk

    return min(pending_risk + conf_risk, 1.0)
