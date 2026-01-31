"""
tests/test_memory.py - Memory recall scoring and save judgment tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.memory_manager import MemoryManager


class TestRecallScoring:
    """Verify that recall() returns scored results in the correct order."""

    @pytest.mark.asyncio
    async def test_recall_returns_results(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        results = await mgr.recall("楽しい会話", top_k=2)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_recall_keyword_match_ranks_higher(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        results = await mgr.recall("楽しい", top_k=2)
        # The memory with "楽しい" keyword should be first
        assert "楽しい" in results[0].get("keywords", [])

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        assert await mgr.recall("", top_k=3) == []

    @pytest.mark.asyncio
    async def test_recall_no_memories(self, tmp_path: Path):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]", encoding="utf-8")
        mgr = MemoryManager(filepath=empty_file)
        assert await mgr.recall("anything") == []

    @pytest.mark.asyncio
    async def test_recall_respects_top_k(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        results = await mgr.recall("会話", top_k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_recall_marks_last_recalled(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        results = await mgr.recall("楽しい", top_k=1)
        assert results[0]["last_recalled"] is not None


class TestMaybeSave:
    """Verify save decision logic."""

    def test_save_when_importance_ge_3(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        before = mgr.count
        saved = mgr.maybe_save("テストイベント", "応答", {}, importance=3)
        assert saved is True
        assert mgr.count == before + 1

    def test_skip_when_importance_lt_3(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        before = mgr.count
        saved = mgr.maybe_save("低重要度", "応答", {}, importance=1)
        assert saved is False
        assert mgr.count == before

    def test_save_when_explicit_request(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        saved = mgr.maybe_save(
            "何か", "応答", {}, importance=1, explicit_request=True
        )
        assert saved is True

    def test_save_when_attachment_involved(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        saved = mgr.maybe_save(
            "何か", "応答", {}, importance=1, involves_attachment=True
        )
        assert saved is True

    def test_saved_entry_has_correct_fields(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        mgr.maybe_save("保存テスト", "応答テスト", {}, importance=4)
        last = mgr.memories[-1]
        assert "id" in last
        assert "summary" in last
        assert "keywords" in last
        assert "importance" in last
        assert "date" in last
        assert last["importance"] == 4

    def test_protected_flag_on_high_importance(self, tmp_data_dir: Path):
        mgr = MemoryManager(filepath=tmp_data_dir / "example_memories.json")
        mgr.maybe_save("重要", "応答", {}, importance=5)
        last = mgr.memories[-1]
        assert last["protected"] is True
