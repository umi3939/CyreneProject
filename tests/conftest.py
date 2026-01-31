"""
tests/conftest.py - Shared fixtures for pytest.

Provides temporary data directories and mock LLM functions
so tests run fully offline without API keys.

Note: Tests run with CYRENE_DEBUG=1 to get full internal state visibility.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

# Enable debug mode for tests (internal verification)
os.environ["CYRENE_DEBUG"] = "1"


@pytest.fixture()
def tmp_data_dir(tmp_path: Path):
    """Create a temporary data directory with sample files."""
    data = tmp_path / "data"
    data.mkdir()

    # example_memories.json
    memories = [
        {
            "id": 1,
            "summary": "ユーザーと楽しい会話をした",
            "keywords": ["会話", "楽しい"],
            "importance": 3,
            "date": "2025-06-01T10:00:00",
            "protected": False,
            "last_recalled": None,
        },
        {
            "id": 2,
            "summary": "ユーザーが悲しんでいた時に励ました",
            "keywords": ["悲しい", "励まし"],
            "importance": 4,
            "date": "2025-06-05T20:00:00",
            "protected": True,
            "last_recalled": None,
        },
    ]
    (data / "example_memories.json").write_text(
        json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # example_attachments.json
    attachments = {
        "test_user": {
            "bonds": {"user_A": 0.5},
            "last_interaction": {"user_A": "2025-06-01T10:00:00"},
            "risk": 0.3,
        }
    }
    (data / "example_attachments.json").write_text(
        json.dumps(attachments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # identity.json
    identity = {
        "core_traits": ["romantic", "caring"],
        "trait_confidence": {"romantic": 0.9, "caring": 0.8},
        "pending_changes": [],
        "risk": 0.0,
    }
    (data / "identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # projections.json
    projections = {
        "goals": [
            {
                "id": "g1",
                "description": "テスト目標",
                "progress": 0.1,
                "status": "active",
            }
        ],
        "risk": 0.1,
    }
    (data / "projections.json").write_text(
        json.dumps(projections, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # state.json — use a recent timestamp so delta_seconds is small in tests
    from datetime import datetime

    state = {
        "test_user": {
            "emotions": {"joy": 0.0, "sad": 0.0, "fear": 0.0, "anger": 0.0, "calm": 0.5},
            "drives": {"social": 0.5, "curiosity": 0.5},
            "mood": 0.0,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "loss_aversion": 0.3,
            "fear_index": 0.0,
        }
    }
    (data / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # persona.json
    persona = {
        "name": "キュレネ",
        "first_person": "あたし",
        "second_person": "あなた",
        "tone": "romantic, sweet",
        "style_rules": {
            "禁止": ["です", "ます"],
            "推奨": ["♪", "！"],
            "語尾変換": {"かな": "かしら"},
        },
        "example_lines": ["ふふっ♪"],
    }
    (data / "persona.json").write_text(
        json.dumps(persona, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return data


@pytest.fixture()
def mock_llm_call():
    """Async mock LLM that returns deterministic JSON responses."""

    async def _call(prompt: str, params: dict | None = None) -> str:
        if "sentiment" in prompt or "分析" in prompt:
            return json.dumps(
                {
                    "sentiment": 0.6,
                    "keywords": ["楽しい", "会話"],
                    "intent": "greeting",
                    "importance": 3,
                },
                ensure_ascii=False,
            )
        if "応答方針" in prompt:
            return json.dumps(
                [
                    {
                        "policy_label": "empathize",
                        "rationale": "共感する",
                        "expected_drive_change": {"social": -0.05, "curiosity": -0.02},
                        "text": "あら、嬉しいわ♡",
                    }
                ],
                ensure_ascii=False,
            )
        if "セリフ" in prompt or "生成" in prompt:
            return "ふふっ、楽しいわね♪"
        return '{"result": "ok"}'

    return _call
