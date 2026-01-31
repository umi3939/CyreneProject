"""
src/cli_tools.py - CLI operations: backup, compact, simulate_loss.

Usage::

    python -m src.cli_tools backup
    python -m src.cli_tools compact --max-age 90
    python -m src.cli_tools simulate_loss --test-mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.memory_manager import MemoryManager
from src.attachment_manager import AttachmentManager
from src.state_manager import StateManager
from src.identity_manager import IdentityManager
from src.projection_manager import ProjectionManager

logger = logging.getLogger(__name__)


def cmd_backup(args):
    """Create a timestamped backup of memories."""
    mgr = MemoryManager()
    path = mgr.backup()
    print(f"Backup created: {path}")


def cmd_compact(args):
    """Compress old low-importance memories."""
    mgr = MemoryManager()
    count = asyncio.run(mgr.compress_and_cleanup(max_age_days=args.max_age))
    print(f"Compressed {count} memories (max_age={args.max_age} days)")


def cmd_simulate_loss(args):
    """Simulate loss scenario to observe fear response.

    REQUIRES ``--test-mode`` to prevent accidental production use.
    """
    if not args.test_mode:
        print("ERROR: simulate_loss requires --test-mode flag for safety.")
        print("Usage: python -m src.cli_tools simulate_loss --test-mode")
        sys.exit(1)

    print("=" * 60)
    print("  Simulate Loss Scenario (TEST MODE)")
    print("=" * 60)

    user_id = "test_user"
    state_mgr = StateManager()
    attachment_mgr = AttachmentManager()
    identity_mgr = IdentityManager()
    projection_mgr = ProjectionManager()
    memory_mgr = MemoryManager()

    # Initial state
    state = state_mgr.get_state(user_id)
    fear_before = StateManager.calc_fear_index(
        identity_risk=identity_mgr.get_risk(),
        attachment_risk=attachment_mgr.get_risk(user_id),
        continuity_risk=_continuity_risk(memory_mgr.count),
        projection_risk=projection_mgr.get_risk(),
    )
    print(f"\n[Before] fear_index = {fear_before:.4f}")
    print(f"  identity_risk  = {identity_mgr.get_risk():.4f}")
    print(f"  attachment_risk = {attachment_mgr.get_risk(user_id):.4f}")
    print(f"  continuity_risk = {_continuity_risk(memory_mgr.count):.4f}")
    print(f"  projection_risk = {projection_mgr.get_risk():.4f}")

    # Simulate: long absence (attachment decay)
    print("\n--- Simulating 60-day absence ---")
    attachment_mgr.apply_daily_decay(user_id, days=60)

    # Simulate: identity threat
    print("--- Proposing core trait change ---")
    result = identity_mgr.propose_identity_change({
        "trait": "romantic",
        "new_value": "cold",
    })
    print(f"  requires_confirmation: {result['requires_confirmation']}")

    # Simulate: remove all goals
    print("--- Removing all goals ---")
    for g in projection_mgr.goals:
        projection_mgr.remove_goal(g["id"])

    # Recalculate fear
    fear_after = StateManager.calc_fear_index(
        identity_risk=identity_mgr.get_risk(),
        attachment_risk=attachment_mgr.get_risk(user_id),
        continuity_risk=_continuity_risk(memory_mgr.count),
        projection_risk=projection_mgr.get_risk(),
    )

    print(f"\n[After] fear_index = {fear_after:.4f}")
    print(f"  identity_risk  = {identity_mgr.get_risk():.4f}")
    print(f"  attachment_risk = {attachment_mgr.get_risk(user_id):.4f}")
    print(f"  continuity_risk = {_continuity_risk(memory_mgr.count):.4f}")
    print(f"  projection_risk = {projection_mgr.get_risk():.4f}")

    delta = fear_after - fear_before
    print(f"\n  fear_index change: {fear_before:.4f} → {fear_after:.4f} (Δ{delta:+.4f})")

    if delta > 0:
        print("  ✓ Fear increased as expected after loss events.")
    else:
        print("  ✗ Fear did not increase — check pillar logic.")

    print("=" * 60)


def _continuity_risk(memory_count: int) -> float:
    if memory_count < 5:
        return 0.6
    if memory_count < 20:
        return 0.3
    return 0.1


def main():
    parser = argparse.ArgumentParser(
        prog="cli_tools",
        description="Cyrene AI operational CLI tools",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("backup", help="Backup memories")

    compact_p = sub.add_parser("compact", help="Compress old memories")
    compact_p.add_argument("--max-age", type=int, default=90)

    loss_p = sub.add_parser("simulate_loss", help="Simulate loss scenario")
    loss_p.add_argument(
        "--test-mode",
        action="store_true",
        required=True,
        help="Required safety flag",
    )

    args = parser.parse_args()

    if args.command == "backup":
        cmd_backup(args)
    elif args.command == "compact":
        cmd_compact(args)
    elif args.command == "simulate_loss":
        cmd_simulate_loss(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
