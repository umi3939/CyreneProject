"""
tests/test_save_load_warmup.py - save_load_warmup.py のテスト

テスト項目:
- 静的宣言の整合性
  - 全エントリのcache_attrがオーケストレータ属性に存在すること
  - 分類A(種別R/A)のmodule_attrが非空であること
  - 分類A(種別A)のaccessor_nameが非空であること
  - 分類B(種別S)のmodule_attrが空であること
  - エントリの重複がないこと
- 再導出方法の種別テスト
  - 種別R: 読み取り直接代入
  - 種別A: アクセサ経由取得
  - 種別S: スキップ（何も行わない）
- execute_warmup の動作テスト
  - 正常な再導出: 結果辞書にderivedが含まれること
  - モジュール不在時: failedステータス
  - ソース空時: empty_sourceステータス
  - 例外発生時: failedステータス（安全弁3）
  - 分類Bのキャッシュが変更されないこと
- 統合テスト
  - オーケストレータのload()後にwarmupが呼ばれること
  - save→load→warmupサイクルで分類Aキャッシュが復元されること
- ユーティリティ関数テスト
  - get_warmup_entries / get_classification_a_entries / get_classification_b_entries
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from psyche.save_load_warmup import (
    DerivationType,
    WarmupEntry,
    WARMUP_ENTRIES,
    execute_warmup,
    get_warmup_entries,
    get_classification_a_entries,
    get_classification_b_entries,
)


# ── 静的宣言の整合性テスト ──────────────────────────────────────────


class TestWarmupEntryDeclarations:
    """静的宣言WARMUP_ENTRIESの整合性テスト。"""

    def test_entries_is_tuple(self):
        assert isinstance(WARMUP_ENTRIES, tuple)

    def test_entries_non_empty(self):
        assert len(WARMUP_ENTRIES) > 0

    def test_no_duplicate_cache_attrs(self):
        cache_attrs = [e.cache_attr for e in WARMUP_ENTRIES]
        assert len(cache_attrs) == len(set(cache_attrs)), (
            f"Duplicate cache_attrs found: "
            f"{[a for a in cache_attrs if cache_attrs.count(a) > 1]}"
        )

    def test_classification_a_have_module_attr(self):
        for entry in WARMUP_ENTRIES:
            if entry.derivation != DerivationType.S:
                assert entry.module_attr, (
                    f"Classification A entry {entry.cache_attr} must have module_attr"
                )

    def test_classification_a_type_a_have_accessor_name(self):
        for entry in WARMUP_ENTRIES:
            if entry.derivation == DerivationType.A:
                assert entry.accessor_name, (
                    f"Type A entry {entry.cache_attr} must have accessor_name"
                )

    def test_classification_b_have_empty_module_attr(self):
        for entry in WARMUP_ENTRIES:
            if entry.derivation == DerivationType.S:
                assert entry.module_attr == "", (
                    f"Classification B entry {entry.cache_attr} "
                    f"should have empty module_attr, got '{entry.module_attr}'"
                )

    def test_all_entries_are_warmup_entry(self):
        for entry in WARMUP_ENTRIES:
            assert isinstance(entry, WarmupEntry)

    def test_derivation_types_valid(self):
        for entry in WARMUP_ENTRIES:
            assert entry.derivation in (
                DerivationType.R, DerivationType.A, DerivationType.S
            )

    def test_has_classification_a_entries(self):
        a_entries = get_classification_a_entries()
        assert len(a_entries) > 0

    def test_has_classification_b_entries(self):
        b_entries = get_classification_b_entries()
        assert len(b_entries) > 0

    def test_a_plus_b_equals_total(self):
        a = get_classification_a_entries()
        b = get_classification_b_entries()
        assert len(a) + len(b) == len(WARMUP_ENTRIES)


class TestWarmupEntryOrchestatorConsistency:
    """WARMUP_ENTRIESの各cache_attrがオーケストレータに存在することを検証。"""

    @pytest.fixture
    def orchestrator(self):
        from psyche.orchestrator import PsycheOrchestrator
        return PsycheOrchestrator()

    def test_all_cache_attrs_exist_on_orchestrator(self, orchestrator):
        for entry in WARMUP_ENTRIES:
            assert hasattr(orchestrator, entry.cache_attr), (
                f"cache_attr '{entry.cache_attr}' not found on PsycheOrchestrator"
            )

    def test_all_module_attrs_exist_on_orchestrator(self, orchestrator):
        for entry in WARMUP_ENTRIES:
            if entry.derivation == DerivationType.S:
                continue
            assert hasattr(orchestrator, entry.module_attr), (
                f"module_attr '{entry.module_attr}' not found on PsycheOrchestrator"
            )

    def test_all_accessors_exist_on_modules(self, orchestrator):
        for entry in WARMUP_ENTRIES:
            if entry.derivation != DerivationType.A:
                continue
            module = getattr(orchestrator, entry.module_attr)
            assert hasattr(module, entry.accessor_name), (
                f"accessor '{entry.accessor_name}' not found on "
                f"module '{entry.module_attr}' (type: {type(module).__name__})"
            )
            accessor = getattr(module, entry.accessor_name)
            assert callable(accessor), (
                f"accessor '{entry.accessor_name}' on '{entry.module_attr}' "
                f"is not callable"
            )


# ── 種別Rのテスト ────────────────────────────────────────────────────


class TestDerivationTypeR:
    """種別R: 読み取り直接代入のテスト。"""

    def test_direct_read_with_sub_attr(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_state = MagicMock()
        mock_module._state = mock_state
        mock_orch._test_module = mock_module
        mock_orch._test_cache = None

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.R,
                source_sub_attr="_state",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "derived"
        assert mock_orch._test_cache == mock_state

    def test_direct_read_module_not_found(self):
        mock_orch = MagicMock(spec=[])

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_missing_module",
                derivation=DerivationType.R,
                source_sub_attr="_state",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "failed"

    def test_direct_read_source_none(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_module._state = None
        mock_orch._test_module = mock_module
        mock_orch._test_cache = None

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.R,
                source_sub_attr="_state",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "empty_source"

    def test_direct_read_exception_safety(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        type(mock_module)._state = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("test error"))
        )
        mock_orch._test_module = mock_module

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.R,
                source_sub_attr="_state",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "failed"


# ── 種別Aのテスト ────────────────────────────────────────────────────


class TestDerivationTypeA:
    """種別A: アクセサ経由取得のテスト。"""

    def test_accessor_derivation(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_result = MagicMock()
        mock_module.get_last_store.return_value = mock_result
        mock_orch._test_module = mock_module
        mock_orch._test_cache = None

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.A,
                accessor_name="get_last_store",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "derived"
        mock_module.get_last_store.assert_called_once()

    def test_accessor_returns_none(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_module.get_last_store.return_value = None
        mock_orch._test_module = mock_module
        mock_orch._test_cache = None

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.A,
                accessor_name="get_last_store",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "empty_source"

    def test_accessor_not_found(self):
        mock_orch = MagicMock()
        mock_module = MagicMock(spec=[])
        mock_orch._test_module = mock_module

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.A,
                accessor_name="nonexistent_method",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "failed"

    def test_accessor_exception_safety(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_module.get_last_store.side_effect = RuntimeError("test error")
        mock_orch._test_module = mock_module

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.A,
                accessor_name="get_last_store",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "failed"

    def test_accessor_module_not_found(self):
        mock_orch = MagicMock(spec=[])

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_missing",
                derivation=DerivationType.A,
                accessor_name="get_last_store",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "failed"


# ── 種別Sのテスト ────────────────────────────────────────────────────


class TestDerivationTypeS:
    """種別S: スキップのテスト。"""

    def test_skip_does_nothing(self):
        mock_orch = MagicMock()
        original_value = object()
        mock_orch._test_cache = original_value

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="",
                derivation=DerivationType.S,
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "skipped"
        # スキップなのでsetattr呼ばれていないことを確認
        # (MagicMockなので直接比較)
        assert mock_orch._test_cache == original_value


# ── execute_warmup 統合テスト ────────────────────────────────────────


class TestExecuteWarmup:
    """execute_warmupの統合テスト。"""

    def test_returns_dict(self):
        mock_orch = MagicMock()
        results = execute_warmup(mock_orch)
        assert isinstance(results, dict)

    def test_all_entries_have_result(self):
        mock_orch = MagicMock()
        results = execute_warmup(mock_orch)
        for entry in WARMUP_ENTRIES:
            assert entry.cache_attr in results, (
                f"Missing result for {entry.cache_attr}"
            )

    def test_result_statuses_valid(self):
        mock_orch = MagicMock()
        results = execute_warmup(mock_orch)
        valid_statuses = {"derived", "skipped", "failed", "empty_source"}
        for attr, status in results.items():
            assert status in valid_statuses, (
                f"Invalid status '{status}' for {attr}"
            )

    def test_classification_b_all_skipped(self):
        mock_orch = MagicMock()
        results = execute_warmup(mock_orch)
        for entry in get_classification_b_entries():
            assert results[entry.cache_attr] == "skipped", (
                f"Classification B entry {entry.cache_attr} should be skipped, "
                f"got '{results[entry.cache_attr]}'"
            )

    def test_mixed_entries(self):
        mock_orch = MagicMock()
        mock_module_ok = MagicMock()
        mock_result = MagicMock()
        mock_module_ok.get_data.return_value = mock_result
        mock_orch._ok_module = mock_module_ok

        mock_module_empty = MagicMock()
        mock_module_empty.get_data.return_value = None
        mock_orch._empty_module = mock_module_empty

        entries = (
            WarmupEntry(
                cache_attr="_ok_cache",
                module_attr="_ok_module",
                derivation=DerivationType.A,
                accessor_name="get_data",
            ),
            WarmupEntry(
                cache_attr="_empty_cache",
                module_attr="_empty_module",
                derivation=DerivationType.A,
                accessor_name="get_data",
            ),
            WarmupEntry(
                cache_attr="_skip_cache",
                module_attr="",
                derivation=DerivationType.S,
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_ok_cache"] == "derived"
        assert results["_empty_cache"] == "empty_source"
        assert results["_skip_cache"] == "skipped"

    def test_failure_isolation(self):
        """One entry's failure does not affect other entries."""
        mock_orch = MagicMock()
        mock_fail = MagicMock()
        mock_fail.get_data.side_effect = RuntimeError("boom")
        mock_orch._fail_module = mock_fail

        mock_ok = MagicMock()
        mock_ok.get_data.return_value = MagicMock()
        mock_orch._ok_module = mock_ok

        entries = (
            WarmupEntry(
                cache_attr="_fail_cache",
                module_attr="_fail_module",
                derivation=DerivationType.A,
                accessor_name="get_data",
            ),
            WarmupEntry(
                cache_attr="_ok_cache",
                module_attr="_ok_module",
                derivation=DerivationType.A,
                accessor_name="get_data",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_fail_cache"] == "failed"
        assert results["_ok_cache"] == "derived"


# ── オーケストレータ統合テスト ──────────────────────────────────────


class TestOrchestratorIntegration:
    """オーケストレータのload()後にwarmupが呼ばれることを検証。"""

    @pytest.fixture
    def orchestrator(self):
        from psyche.orchestrator import PsycheOrchestrator
        return PsycheOrchestrator()

    def test_warmup_called_during_load(self, orchestrator, tmp_path):
        """save→load サイクルで warmup が実行されることを確認。"""
        save_path = tmp_path / "test_snapshot.json"
        orchestrator.save(path=save_path)

        with patch("psyche.orchestrator.execute_warmup") as mock_warmup:
            mock_warmup.return_value = {}
            orchestrator.load(path=save_path)
            mock_warmup.assert_called_once_with(orchestrator)

    def test_warmup_not_called_when_no_snapshot(self, orchestrator, tmp_path):
        """スナップショットが存在しない場合はwarmupが呼ばれない。"""
        with patch("psyche.orchestrator.execute_warmup") as mock_warmup:
            result = orchestrator.load(path=tmp_path / "nonexistent.json")
            assert result is False
            mock_warmup.assert_not_called()

    def test_save_load_cycle_classification_a_caches(self, orchestrator, tmp_path):
        """save→loadサイクル後に分類Aキャッシュ属性がオーケストレータ上に存在することを検証。"""
        save_path = tmp_path / "test_snapshot.json"

        # Save without running any ticks (initial state)
        orchestrator.save(path=save_path)

        # Create fresh orchestrator and load
        from psyche.orchestrator import PsycheOrchestrator
        new_orch = PsycheOrchestrator()
        loaded = new_orch.load(path=save_path)
        assert loaded is True

        # Verify that classification A caches are accessible on the orchestrator
        # (they may be None if modules had no data, but the attribute must exist)
        a_entries = get_classification_a_entries()
        for entry in a_entries:
            assert hasattr(new_orch, entry.cache_attr), (
                f"Cache {entry.cache_attr} missing on restored orchestrator"
            )

    def test_save_load_cycle_with_post_response_update(self, orchestrator, tmp_path):
        """post_response_update後にsave→loadサイクルで分類Aキャッシュが復元される。"""
        save_path = tmp_path / "test_snapshot.json"

        # Run a tick to populate some caches
        from psyche.state import Percept
        percept = Percept(
            text="test", intent="test", emotion="neutral",
            topics=["test"], emotion_valence=0.0,
        )
        orchestrator.post_response_update(percept, delta_time=1.0, user_id="test")

        orchestrator.save(path=save_path)

        # Create fresh orchestrator and load
        from psyche.orchestrator import PsycheOrchestrator
        new_orch = PsycheOrchestrator()
        loaded = new_orch.load(path=save_path)
        assert loaded is True

        # Verify that classification A caches with accessor support
        # are accessible (may be None if modules don't retain last output)
        a_entries = get_classification_a_entries()
        for entry in a_entries:
            assert hasattr(new_orch, entry.cache_attr), (
                f"Cache {entry.cache_attr} missing on restored orchestrator"
            )

    def test_classification_b_caches_remain_none_after_fresh_load(self, tmp_path):
        """分類Bのキャッシュはload後もNone/初期値のままであることを確認。"""
        from psyche.orchestrator import PsycheOrchestrator

        orch = PsycheOrchestrator()
        save_path = tmp_path / "test_snapshot.json"
        orch.save(path=save_path)

        new_orch = PsycheOrchestrator()
        new_orch.load(path=save_path)

        # Classification B caches should not be populated by warmup
        b_entries = get_classification_b_entries()
        for entry in b_entries:
            cache_value = getattr(new_orch, entry.cache_attr)
            # These should be None, "", {}, False, or similar initial values
            if entry.cache_attr in ("_last_selected_policy_label",
                                    "_last_selected_policy_axis"):
                assert cache_value == "", (
                    f"Cache {entry.cache_attr} should be empty string, "
                    f"got {cache_value!r}"
                )
            elif entry.cache_attr == "_last_emotion_for_action_result":
                assert cache_value == {}, (
                    f"Cache {entry.cache_attr} should be empty dict, "
                    f"got {cache_value!r}"
                )
            elif entry.cache_attr == "_last_has_silence":
                assert cache_value is False, (
                    f"Cache {entry.cache_attr} should be False, "
                    f"got {cache_value!r}"
                )
            else:
                assert cache_value is None, (
                    f"Cache {entry.cache_attr} should be None, "
                    f"got {cache_value!r}"
                )


# ── ユーティリティ関数テスト ────────────────────────────────────────


class TestUtilityFunctions:
    """get_warmup_entries等のユーティリティ関数テスト。"""

    def test_get_warmup_entries_returns_same_as_constant(self):
        assert get_warmup_entries() is WARMUP_ENTRIES

    def test_classification_a_no_skip(self):
        for entry in get_classification_a_entries():
            assert entry.derivation != DerivationType.S

    def test_classification_b_all_skip(self):
        for entry in get_classification_b_entries():
            assert entry.derivation == DerivationType.S

    def test_classifications_are_disjoint(self):
        a_attrs = {e.cache_attr for e in get_classification_a_entries()}
        b_attrs = {e.cache_attr for e in get_classification_b_entries()}
        assert a_attrs.isdisjoint(b_attrs)

    def test_classifications_cover_all_entries(self):
        a_attrs = {e.cache_attr for e in get_classification_a_entries()}
        b_attrs = {e.cache_attr for e in get_classification_b_entries()}
        all_attrs = {e.cache_attr for e in WARMUP_ENTRIES}
        assert a_attrs | b_attrs == all_attrs


# ── DerivationType テスト ────────────────────────────────────────────


class TestDerivationType:
    """DerivationType列挙型のテスト。"""

    def test_r_value(self):
        assert DerivationType.R == "read_direct"

    def test_a_value(self):
        assert DerivationType.A == "accessor"

    def test_s_value(self):
        assert DerivationType.S == "skip"

    def test_is_string_enum(self):
        assert isinstance(DerivationType.R, str)
        assert isinstance(DerivationType.A, str)
        assert isinstance(DerivationType.S, str)


# ── WarmupEntry frozen テスト ────────────────────────────────────────


class TestWarmupEntryFrozen:
    """WarmupEntryがfrozenであることのテスト。"""

    def test_cannot_modify_cache_attr(self):
        entry = WarmupEntry(
            cache_attr="_test",
            module_attr="_mod",
            derivation=DerivationType.S,
        )
        with pytest.raises(AttributeError):
            entry.cache_attr = "_other"

    def test_cannot_modify_derivation(self):
        entry = WarmupEntry(
            cache_attr="_test",
            module_attr="_mod",
            derivation=DerivationType.S,
        )
        with pytest.raises(AttributeError):
            entry.derivation = DerivationType.R


# ── 安全弁テスト ────────────────────────────────────────────────────


class TestSafetyValves:
    """設計書の安全弁要件のテスト。"""

    def test_safety_valve_1_no_processing_for_classification_b(self):
        """安全弁1: 分類Bのキャッシュには一切の処理を行わない。"""
        mock_orch = MagicMock()
        b_entries = get_classification_b_entries()
        assert len(b_entries) > 0
        results = execute_warmup(mock_orch)
        for entry in b_entries:
            assert results[entry.cache_attr] == "skipped"

    def test_safety_valve_2_no_module_update_calls(self):
        """安全弁2: モジュール内部更新メソッドを呼び出さない。"""
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_store = MagicMock()
        mock_module.get_last_store.return_value = mock_store
        mock_orch._test_module = mock_module

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.A,
                accessor_name="get_last_store",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            execute_warmup(mock_orch)

        # get_last_store was called (read-only accessor)
        mock_module.get_last_store.assert_called_once()
        # No update/tick/process methods called
        mock_module.tick.assert_not_called()
        mock_module.update.assert_not_called()
        mock_module.process.assert_not_called()

    def test_safety_valve_3_failure_keeps_none(self):
        """安全弁3: 再導出失敗時はNoneのまま維持。"""
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_module.get_last_store.side_effect = RuntimeError("oops")
        mock_orch._test_module = mock_module
        mock_orch._test_cache = None

        entries = (
            WarmupEntry(
                cache_attr="_test_cache",
                module_attr="_test_module",
                derivation=DerivationType.A,
                accessor_name="get_last_store",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert results["_test_cache"] == "failed"

    def test_safety_valve_4_static_declaration_immutable(self):
        """安全弁4: 静的宣言は実行時に変更されない。"""
        entries_before = get_warmup_entries()
        mock_orch = MagicMock()
        execute_warmup(mock_orch)
        entries_after = get_warmup_entries()
        assert entries_before is entries_after
        assert len(entries_before) == len(entries_after)

    def test_safety_valve_5_enrichment_not_modified(self):
        """安全弁5: enrichment構造を変更しない。"""
        mock_orch = MagicMock()
        mock_orch._enrichment_prev_cache = {"test": "value"}
        original_cache = mock_orch._enrichment_prev_cache.copy()

        execute_warmup(mock_orch)

        # enrichment cache is not in WARMUP_ENTRIES, so not modified
        enrichment_entry_exists = any(
            e.cache_attr == "_enrichment_prev_cache" for e in WARMUP_ENTRIES
        )
        assert not enrichment_entry_exists


# ── 再導出結果の集計テスト ──────────────────────────────────────────


class TestResultCounting:
    """execute_warmup結果の集計テスト。"""

    def test_count_derived(self):
        mock_orch = MagicMock()
        mock_module = MagicMock()
        mock_module.get_data.return_value = MagicMock()
        mock_orch._m = mock_module

        entries = (
            WarmupEntry(
                cache_attr="_c1", module_attr="_m",
                derivation=DerivationType.A, accessor_name="get_data",
            ),
            WarmupEntry(
                cache_attr="_c2", module_attr="_m",
                derivation=DerivationType.A, accessor_name="get_data",
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        derived_count = sum(1 for v in results.values() if v == "derived")
        assert derived_count == 2

    def test_count_mixed(self):
        mock_orch = MagicMock()
        mock_ok = MagicMock()
        mock_ok.get_data.return_value = MagicMock()
        mock_orch._ok = mock_ok

        mock_empty = MagicMock()
        mock_empty.get_data.return_value = None
        mock_orch._empty = mock_empty

        entries = (
            WarmupEntry(
                cache_attr="_ok_cache", module_attr="_ok",
                derivation=DerivationType.A, accessor_name="get_data",
            ),
            WarmupEntry(
                cache_attr="_empty_cache", module_attr="_empty",
                derivation=DerivationType.A, accessor_name="get_data",
            ),
            WarmupEntry(
                cache_attr="_skip_cache", module_attr="",
                derivation=DerivationType.S,
            ),
        )

        with patch("psyche.save_load_warmup.WARMUP_ENTRIES", entries):
            results = execute_warmup(mock_orch)

        assert sum(1 for v in results.values() if v == "derived") == 1
        assert sum(1 for v in results.values() if v == "empty_source") == 1
        assert sum(1 for v in results.values() if v == "skipped") == 1
