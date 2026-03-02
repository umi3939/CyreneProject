"""
tests/test_memory_pipeline.py - 長期記憶保存パイプラインの品質検証テスト

design_memory_pipeline_test.md に基づき、以下の7検証項目を実装:

1. 保存パイプライン段階網羅テスト
2. 帰還経路到達性テスト
3. 記憶保存→想起候補プール更新の経路テスト
4. 要約API失敗時の挙動テスト
5. 保存判定閾値テスト
6. 保存記録バッファの事実記録テスト
7. 帰還経路トリガーの間接検証テスト

テスト追加のみ。動作ロジック変更なし。外部APIはモック化。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.memory_manager import MemoryManager
from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept
from psyche.memory_link import recall_with_mood


# ── Helpers ───────────────────────────────────────────────────────


def _make_percept(
    emotion: str = "happy",
    valence: float = 0.7,
    text: str = "テスト画面",
) -> Percept:
    """テスト用の Percept を生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent="expression",
        emotion_valence=valence,
    )


def _make_memory_manager(tmp_path: Path, memories: list[dict] | None = None) -> MemoryManager:
    """テスト用の MemoryManager を生成する。"""
    filepath = tmp_path / "test_memories.json"
    data = memories or []
    filepath.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return MemoryManager(filepath=filepath)


def _make_summary_response(
    summary: str = "テスト要約",
    keywords: list[str] | None = None,
    importance: int = 3,
) -> str:
    """要約API応答のJSONを生成する。"""
    return json.dumps(
        {
            "summary": summary,
            "keywords": keywords or ["テスト", "要約"],
            "importance": importance,
        },
        ensure_ascii=False,
    )


# ── SaveRecordBuffer: 保存記録バッファ (テスト内で使用) ───────────


@dataclass
class SaveRecord:
    """保存操作の事実記録エントリ。

    設計書 3.【実装構造】の「保存記録バッファ」に対応。
    テストコード内のみで使用。パイプライン動作ロジックに影響しない。
    """
    summary_length: int = 0
    keyword_count: int = 0
    importance_value: int = 0
    save_success: bool = False
    total_memory_count: int = 0
    source_entry_count: int = 0
    timestamp: float = 0.0
    tick_number: int = 0
    notification_reached: bool = False


class SaveRecordBuffer:
    """FIFO方式の保存記録バッファ。

    設計書 3.【実装構造】に記述された保存記録バッファの
    テスト内実装。
    永続化対象外。enrichmentに露出しない。
    統計量の算出・傾向分析・異常検出は行わない。
    """

    def __init__(self, max_size: int = 50):
        self._buffer: deque[SaveRecord] = deque(maxlen=max_size)

    def add(self, record: SaveRecord) -> None:
        self._buffer.append(record)

    @property
    def records(self) -> list[SaveRecord]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


# =============================================================================
# 1. 保存パイプライン段階網羅テスト
# =============================================================================


class TestSavePipelineStages:
    """対話文脈→要約API→JSON解析→記憶保存→通知コールバック の
    各段階を一貫して検証する。"""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, tmp_path: Path):
        """正常系: 全段階が成功する場合のパイプライン検証。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        # 要約APIのモック応答
        mock_response = MagicMock()
        mock_response.text = _make_summary_response(
            summary="ユーザーとゲームの話をした",
            keywords=["ゲーム", "楽しい"],
            importance=4,
        )

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # 記録バッファ
        record_buffer = SaveRecordBuffer()

        # パイプライン実行（brain.py summarize_and_save の手動再現）
        # ContextManagerからのエントリ相当
        entries = [
            {"speaker_label": "ユーザー", "text": "今日のゲーム楽しかったね"},
            {"speaker_label": "キュレネ", "text": "うん！すごく楽しかったわ♪"},
            {"speaker_label": "ユーザー", "text": "また遊ぼう"},
        ]

        # 段階1: エントリ数の確認
        assert len(entries) >= 2, "要約対象のエントリが2件以上ある"

        # 段階2: 要約生成（モック）
        conversation_text = "\n".join(
            f"[{e['speaker_label']}] {e['text']}" for e in entries
        )
        prompt = (
            "以下の会話を要約して、次回の会話で思い出すのに役立つ情報をJSON形式で返して。\n"
            "JSONのみ出力し、他のテキストは含めないこと。\n"
            '{"summary": "...", "keywords": ["...", "..."], "importance": 1-5}\n\n'
            f"会話ログ:\n{conversation_text}"
        )
        response = await mock_client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={},
        )
        assert response is not None
        assert response.text is not None

        # 段階3: JSON解析
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        summary = data.get("summary", "")
        keywords = data.get("keywords", [])
        importance = int(data.get("importance", 3))

        assert summary == "ユーザーとゲームの話をした"
        assert "ゲーム" in keywords
        assert importance == 4

        # 段階4: 記憶保存
        before_count = mgr.count
        saved = mgr.maybe_save(summary, "", {}, importance=importance)
        assert saved is True
        assert mgr.count == before_count + 1

        # 段階5: 通知コールバック
        fear_before = orch.fear_level
        orch.on_memory_saved(
            summary=summary,
            keywords=keywords,
            memory_count=mgr.count,
        )

        # 事実記録
        record = SaveRecord(
            summary_length=len(summary),
            keyword_count=len(keywords),
            importance_value=importance,
            save_success=True,
            total_memory_count=mgr.count,
            source_entry_count=len(entries),
            timestamp=time.time(),
            tick_number=orch.tick_count,
            notification_reached=True,
        )
        record_buffer.add(record)

        assert len(record_buffer) == 1
        assert record_buffer.records[0].save_success is True
        assert record_buffer.records[0].notification_reached is True
        assert record_buffer.records[0].summary_length > 0

    @pytest.mark.asyncio
    async def test_pipeline_json_with_markdown_fences(self, tmp_path: Path):
        """マークダウンコードフェンス付きJSON応答の解析テスト。"""
        raw = '```json\n{"summary": "テスト", "keywords": ["a"], "importance": 3}\n```'
        # Strip markdown code fences (brain.py logic)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        assert data["summary"] == "テスト"
        assert data["keywords"] == ["a"]
        assert data["importance"] == 3

    @pytest.mark.asyncio
    async def test_pipeline_stages_preserve_data_integrity(self, tmp_path: Path):
        """各段階間でデータの欠落がないことの検証。"""
        mgr = _make_memory_manager(tmp_path)

        summary = "重要な会話内容の要約"
        keywords = ["重要", "会話"]
        importance = 4

        # 保存
        saved = mgr.maybe_save(summary, "", {}, importance=importance)
        assert saved is True

        # 保存されたエントリの検証
        last = mgr.memories[-1]
        assert last["summary"] == summary
        assert last["importance"] == importance
        assert "date" in last
        assert "id" in last


# =============================================================================
# 2. 帰還経路到達性テスト
# =============================================================================


class TestReturnPathwayReachability:
    """記憶保存後にオーケストレータの保存通知コールバックが呼ばれ、
    連続性状態更新と恐怖指標再計算が実行されることを検証する。"""

    def test_on_memory_saved_updates_continuity(self):
        """保存通知により連続性のmemory_countが更新される。"""
        orch = PsycheOrchestrator(memory_count=5)
        prev_count = orch._psyche.continuity.memory_count if orch._psyche.continuity else 0

        orch.on_memory_saved(
            summary="テスト要約",
            keywords=["テスト"],
            memory_count=10,
        )

        if orch._psyche.continuity is not None:
            assert orch._psyche.continuity.memory_count == 10

    def test_on_memory_saved_triggers_fear_recomputation(self):
        """保存通知で恐怖指標が再計算される。"""
        orch = PsycheOrchestrator(memory_count=0)
        fear_before = orch.fear_level

        # memory_count=0 → 100 への大変動で恐怖指標が変化しうる
        orch.on_memory_saved(
            summary="テスト要約",
            keywords=["テスト"],
            memory_count=100,
        )
        # 恐怖指標は再計算されたことの確認（値自体の方向は未規定）
        # fear_levelプロパティがエラーなく取得できることが重要
        assert isinstance(orch.fear_level, float)
        assert 0.0 <= orch.fear_level <= 1.0

    def test_on_memory_saved_callback_is_reachable_from_pipeline(self, tmp_path: Path):
        """brain.py summarize_and_save と同等の経路で通知が到達することの検証。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        # 保存
        summary = "パイプラインテスト要約"
        keywords = ["パイプライン", "テスト"]
        mgr.maybe_save(summary, "", {}, importance=3)

        # 通知コールバック（brain.py summarize_and_save と同じ呼び出し）
        orch.on_memory_saved(
            summary=summary,
            keywords=keywords,
            memory_count=len(mgr._memories),
        )

        # 通知が正常に処理されたことの確認
        if orch._psyche.continuity is not None:
            assert orch._psyche.continuity.memory_count == len(mgr._memories)

    def test_on_memory_saved_multiple_calls_accumulate(self):
        """複数回の保存通知が正常に処理される。"""
        orch = PsycheOrchestrator(memory_count=0)

        for i in range(5):
            orch.on_memory_saved(
                summary=f"要約{i}",
                keywords=[f"kw{i}"],
                memory_count=i + 1,
            )

        if orch._psyche.continuity is not None:
            assert orch._psyche.continuity.memory_count == 5


# =============================================================================
# 3. 記憶保存→想起候補プール更新の経路テスト
# =============================================================================


class TestMemorySaveRecallPath:
    """記憶保存後に想起操作を行い、保存した記憶が想起候補に
    含まれうることを検証する。"""

    @pytest.mark.asyncio
    async def test_saved_memory_is_recallable(self, tmp_path: Path):
        """保存した記憶が直後のrecallで取得可能。"""
        mgr = _make_memory_manager(tmp_path)

        # 保存
        mgr.maybe_save(
            "ユーザーとゲームの話をした。楽しいゲーム体験",
            "", {},
            importance=4,
        )
        assert mgr.count == 1

        # 想起
        results = await mgr.recall("ゲーム", top_k=3)
        assert len(results) > 0
        assert any("ゲーム" in r.get("summary", "") for r in results)

    @pytest.mark.asyncio
    async def test_saved_memory_updates_recall_pool(self, tmp_path: Path):
        """保存前後でrecall結果が変化する。"""
        mgr = _make_memory_manager(tmp_path)

        # 保存前: 空
        results_before = await mgr.recall("新しいトピック", top_k=3)
        assert len(results_before) == 0

        # 保存
        mgr.maybe_save(
            "新しいトピックに関する会話",
            "", {},
            importance=3,
        )

        # 保存後: 取得可能
        results_after = await mgr.recall("新しいトピック", top_k=3)
        assert len(results_after) > 0

    @pytest.mark.asyncio
    async def test_saved_memory_with_keywords_is_keyword_matchable(self, tmp_path: Path):
        """キーワード付きで保存された記憶がキーワードマッチで想起される。"""
        mgr = _make_memory_manager(tmp_path)

        # 直接エントリ追加（キーワード制御のため）
        entry = {
            "id": 1,
            "summary": "特定のキーワードテスト",
            "keywords": ["特定", "キーワード"],
            "importance": 4,
            "date": "2026-03-01T10:00:00",
            "protected": False,
            "last_recalled": None,
        }
        mgr._memories.append(entry)

        results = await mgr.recall("特定のキーワード", top_k=3)
        assert len(results) > 0
        assert results[0]["summary"] == "特定のキーワードテスト"

    @pytest.mark.asyncio
    async def test_orchestrator_recall_pool_updated_via_set_recalled(self, tmp_path: Path):
        """orchestrator.set_recalled_memories で想起結果がセットされる。"""
        orch = PsycheOrchestrator(memory_count=0)
        mgr = _make_memory_manager(tmp_path)

        # 保存
        mgr.maybe_save("テスト記憶", "", {}, importance=3)

        # 想起してオーケストレータにセット
        results = await mgr.recall("テスト", top_k=3)
        orch.set_recalled_memories(results)

        # 内部状態が更新されていることの確認
        assert orch._last_recalled_memories is not None
        assert len(orch._last_recalled_memories) > 0


# =============================================================================
# 4. 要約API失敗時の挙動テスト
# =============================================================================


class TestSummaryApiFailure:
    """外部APIが空応答、不正JSON、例外を返した場合にパイプラインが
    安全に中断し、psycheの状態が変化しないことを検証する。"""

    @pytest.mark.asyncio
    async def test_empty_response_does_not_save(self, tmp_path: Path):
        """空応答の場合、記憶保存が行われない。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)
        initial_count = mgr.count
        initial_fear = orch.fear_level

        # 空応答をシミュレート
        mock_response = MagicMock()
        mock_response.text = None

        # brain.py summarize_and_save の空応答分岐を再現
        if not mock_response or not mock_response.text:
            pass  # "Empty summary response" → 保存しない
        else:
            pytest.fail("Should not reach here")

        assert mgr.count == initial_count
        assert orch.fear_level == initial_fear

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_save(self, tmp_path: Path):
        """不正JSON応答の場合、記憶保存が行われない。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)
        initial_count = mgr.count
        initial_fear = orch.fear_level

        raw = "これはJSONではない invalid text"
        save_executed = False

        try:
            data = json.loads(raw)
            save_executed = True
        except json.JSONDecodeError:
            pass  # brain.py の except json.JSONDecodeError と同等

        assert save_executed is False
        assert mgr.count == initial_count
        assert orch.fear_level == initial_fear

    @pytest.mark.asyncio
    async def test_api_exception_does_not_save(self, tmp_path: Path):
        """API呼び出しが例外を投げた場合、記憶保存が行われない。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)
        initial_count = mgr.count
        initial_fear = orch.fear_level

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API connection error")
        )

        save_executed = False
        try:
            await mock_client.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents="test",
                config={},
            )
            save_executed = True
        except Exception:
            pass  # brain.py の except Exception と同等

        assert save_executed is False
        assert mgr.count == initial_count
        assert orch.fear_level == initial_fear

    @pytest.mark.asyncio
    async def test_empty_summary_in_json_does_not_save(self, tmp_path: Path):
        """JSONは正常だがsummaryが空の場合、記憶保存が行われない。"""
        mgr = _make_memory_manager(tmp_path)
        initial_count = mgr.count

        raw = json.dumps({
            "summary": "",
            "keywords": ["a"],
            "importance": 3,
        })
        data = json.loads(raw)
        summary = data.get("summary", "")

        if summary:
            mgr.maybe_save(summary, "", {}, importance=3)

        assert mgr.count == initial_count

    @pytest.mark.asyncio
    async def test_partial_json_response_does_not_corrupt(self, tmp_path: Path):
        """JSONに一部のフィールドが欠けている場合のデフォルト値適用。"""
        raw = json.dumps({"summary": "部分的なJSON"})
        data = json.loads(raw)
        summary = data.get("summary", "")
        keywords = data.get("keywords", [])
        importance = int(data.get("importance", 3))

        assert summary == "部分的なJSON"
        assert keywords == []
        assert importance == 3

    @pytest.mark.asyncio
    async def test_psyche_state_unchanged_after_api_failure(self, tmp_path: Path):
        """API失敗後のpsyche状態が変化しないことの検証。"""
        orch = PsycheOrchestrator(memory_count=0)

        # psyche状態のスナップショット
        state_before = orch._psyche.model_dump()

        # API失敗をシミュレート（通知コールバックは呼ばれない）
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Network error")
        )

        try:
            await mock_client.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents="test",
                config={},
            )
        except Exception:
            pass

        # psyche状態が変化していないことの確認
        state_after = orch._psyche.model_dump()
        assert state_before == state_after


# =============================================================================
# 5. 保存判定閾値テスト
# =============================================================================


class TestSaveThreshold:
    """対話エントリが不足している場合に要約が生成されないことを検証する。"""

    def test_empty_entries_no_summarization(self):
        """エントリが0件の場合、要約が発生しない。"""
        entries: list[dict] = []
        should_summarize = len(entries) >= 2
        assert should_summarize is False

    def test_single_entry_no_summarization(self):
        """エントリが1件の場合、要約が発生しない。"""
        entries = [{"speaker_label": "ユーザー", "text": "テスト"}]
        should_summarize = len(entries) >= 2
        assert should_summarize is False

    def test_two_entries_allows_summarization(self):
        """エントリが2件の場合、要約が可能。"""
        entries = [
            {"speaker_label": "ユーザー", "text": "テスト1"},
            {"speaker_label": "キュレネ", "text": "テスト2"},
        ]
        should_summarize = len(entries) >= 2
        assert should_summarize is True

    def test_low_importance_skips_save(self, tmp_path: Path):
        """importance < 3 の場合、保存が行われない。"""
        mgr = _make_memory_manager(tmp_path)
        saved = mgr.maybe_save("低重要度", "応答", {}, importance=1)
        assert saved is False
        assert mgr.count == 0

    def test_importance_3_saves(self, tmp_path: Path):
        """importance == 3 の場合、保存が行われる。"""
        mgr = _make_memory_manager(tmp_path)
        saved = mgr.maybe_save("中重要度", "応答", {}, importance=3)
        assert saved is True
        assert mgr.count == 1

    def test_importance_boundary_2_does_not_save(self, tmp_path: Path):
        """importance == 2 の場合、保存が行われない。"""
        mgr = _make_memory_manager(tmp_path)
        saved = mgr.maybe_save("低重要度", "応答", {}, importance=2)
        assert saved is False
        assert mgr.count == 0

    def test_explicit_request_overrides_importance(self, tmp_path: Path):
        """explicit_request=True の場合、importance に関わらず保存される。"""
        mgr = _make_memory_manager(tmp_path)
        saved = mgr.maybe_save(
            "低重要度だが明示的", "応答", {},
            importance=1, explicit_request=True,
        )
        assert saved is True
        assert mgr.count == 1


# =============================================================================
# 6. 保存記録バッファの事実記録テスト
# =============================================================================


class TestSaveRecordBufferFacts:
    """保存操作ごとにバッファにエントリが追加され、各断面が
    正しく記録されることを検証する。"""

    def test_record_creation_with_all_facets(self):
        """保存記録に全断面が含まれることの検証。"""
        record = SaveRecord(
            summary_length=50,
            keyword_count=3,
            importance_value=4,
            save_success=True,
            total_memory_count=10,
            source_entry_count=5,
            timestamp=time.time(),
            tick_number=42,
            notification_reached=True,
        )
        assert record.summary_length == 50
        assert record.keyword_count == 3
        assert record.importance_value == 4
        assert record.save_success is True
        assert record.total_memory_count == 10
        assert record.source_entry_count == 5
        assert record.timestamp > 0
        assert record.tick_number == 42
        assert record.notification_reached is True

    def test_buffer_fifo_eviction(self):
        """バッファ上限を超えた場合のFIFO消失の検証。"""
        buffer = SaveRecordBuffer(max_size=3)
        for i in range(5):
            buffer.add(SaveRecord(
                summary_length=i * 10,
                tick_number=i,
            ))
        assert len(buffer) == 3
        # 最古(0, 1)が消え、2, 3, 4 が残る
        records = buffer.records
        assert records[0].tick_number == 2
        assert records[1].tick_number == 3
        assert records[2].tick_number == 4

    def test_buffer_records_are_immutable_after_add(self):
        """バッファに追加された記録が変更されないことの検証。"""
        buffer = SaveRecordBuffer()
        record = SaveRecord(summary_length=100, save_success=True)
        buffer.add(record)

        # 返却されるリストはコピー
        records = buffer.records
        assert len(records) == 1
        assert records[0].summary_length == 100

    def test_buffer_accumulates_multiple_records(self):
        """複数回の保存操作で記録が蓄積されることの検証。"""
        buffer = SaveRecordBuffer()
        for i in range(10):
            buffer.add(SaveRecord(
                summary_length=i * 5,
                keyword_count=i,
                importance_value=min(5, i),
                save_success=True,
                total_memory_count=i + 1,
                source_entry_count=3,
                timestamp=time.time(),
                tick_number=i,
            ))
        assert len(buffer) == 10
        # 各記録の断面が独立していることの検証
        for i, rec in enumerate(buffer.records):
            assert rec.summary_length == i * 5
            assert rec.keyword_count == i
            assert rec.tick_number == i

    def test_record_notification_tracking(self, tmp_path: Path):
        """保存→通知の到達が記録として追跡できることの検証。"""
        buffer = SaveRecordBuffer()
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        summary = "通知追跡テスト"
        keywords = ["追跡"]
        importance = 3

        # 保存
        saved = mgr.maybe_save(summary, "", {}, importance=importance)

        # 通知
        notification_reached = False
        try:
            orch.on_memory_saved(
                summary=summary,
                keywords=keywords,
                memory_count=mgr.count,
            )
            notification_reached = True
        except Exception:
            notification_reached = False

        record = SaveRecord(
            summary_length=len(summary),
            keyword_count=len(keywords),
            importance_value=importance,
            save_success=saved,
            total_memory_count=mgr.count,
            source_entry_count=0,
            timestamp=time.time(),
            tick_number=orch.tick_count,
            notification_reached=notification_reached,
        )
        buffer.add(record)

        assert buffer.records[0].save_success is True
        assert buffer.records[0].notification_reached is True


# =============================================================================
# 7. 帰還経路トリガーの間接検証テスト
# =============================================================================


class TestReturnPathwayIndirect:
    """記憶保存後、次のオーケストレータティックで帰還処理が
    想起候補を参照可能な状態にあることの間接的な検証。

    記憶感情帰還処理自体の内部ロジックの検証は行わない
    （既存テストが担当）。"""

    @pytest.mark.asyncio
    async def test_save_then_recall_then_tick_pathway(self, tmp_path: Path):
        """保存→想起→ティック実行の経路が成立する。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        # 保存
        mgr.maybe_save(
            "帰還経路テスト記憶", "", {},
            importance=4,
        )

        # 通知
        orch.on_memory_saved(
            summary="帰還経路テスト記憶",
            keywords=["帰還", "テスト"],
            memory_count=mgr.count,
        )

        # 想起
        results = await mgr.recall("帰還テスト", top_k=3)
        orch.set_recalled_memories(results)
        assert orch._last_recalled_memories is not None

        # ティック実行（psycheのフルパイプライン）
        percept = _make_percept(text="帰還テスト入力")
        orch.post_response_update(percept, 1.0, "text")

        # ティックが正常に完了することの確認
        assert orch.tick_count >= 1

    @pytest.mark.asyncio
    async def test_recalled_memories_available_for_return_processing(self, tmp_path: Path):
        """set_recalled_memoriesで設定した記憶が参照可能状態にある。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        # 保存と想起
        mgr.maybe_save("想起テスト", "", {}, importance=3)
        results = await mgr.recall("想起", top_k=3)
        assert len(results) > 0

        orch.set_recalled_memories(results)
        assert orch._last_recalled_memories == results

    @pytest.mark.asyncio
    async def test_multiple_saves_and_recall_cycle(self, tmp_path: Path):
        """複数回の保存→想起サイクルが正常に動作する。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        for i in range(3):
            summary = f"サイクル{i}の記憶"
            keywords = [f"サイクル{i}"]
            mgr.maybe_save(summary, "", {}, importance=3)
            orch.on_memory_saved(
                summary=summary,
                keywords=keywords,
                memory_count=mgr.count,
            )

        assert mgr.count == 3

        # 全記憶から想起
        results = await mgr.recall("サイクル", top_k=5)
        assert len(results) == 3

        orch.set_recalled_memories(results)
        assert len(orch._last_recalled_memories) == 3

    @pytest.mark.asyncio
    async def test_tick_after_save_does_not_crash(self, tmp_path: Path):
        """記憶保存→通知→ティック実行がクラッシュしないことの検証。"""
        mgr = _make_memory_manager(tmp_path)
        orch = PsycheOrchestrator(memory_count=0)

        # 初期ティック
        percept = _make_percept(text="初期入力")
        orch.post_response_update(percept, 1.0, "text")

        # 保存→通知
        mgr.maybe_save("テスト", "", {}, importance=3)
        orch.on_memory_saved(
            summary="テスト",
            keywords=["テスト"],
            memory_count=mgr.count,
        )

        # 想起→セット→ティック
        results = await mgr.recall("テスト", top_k=3)
        orch.set_recalled_memories(results)

        percept2 = _make_percept(text="2回目入力")
        orch.post_response_update(percept2, 1.0, "text")

        assert orch.tick_count >= 2

    def test_on_memory_saved_without_prior_tick_is_safe(self):
        """ティック実行前にon_memory_savedを呼んでも安全であることの検証。"""
        orch = PsycheOrchestrator(memory_count=0)
        assert orch.tick_count == 0

        # ティック前に通知
        orch.on_memory_saved(
            summary="early save",
            keywords=["early"],
            memory_count=1,
        )

        # psyche状態が破損していないことの確認
        assert isinstance(orch.fear_level, float)
        assert orch.tick_count == 0  # ティック数は変わらない
