"""
tests/test_session_recovery_check.py - セッション復帰時の状態整合性検証テスト

design_session_recovery_check.md に基づくテスト:
  検証種別A-Eの各パターンが正しく動作することを確認する。
  検証が状態に書き込まないこと、修復しないことを検証する。
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from psyche.save_load_warmup import (
    CheckType,
    ConsistencyCheckEntry,
    ConsistencyCheckResult,
    ConsistencyFinding,
    CONSISTENCY_CHECK_ENTRIES,
    execute_session_recovery_check,
    get_consistency_check_entries,
    _check_type_a,
    _check_type_b,
    _check_type_c,
    _check_type_d,
    _check_type_e,
    _resolve_module_state,
    _get_records_list,
    _get_tick_value,
)


# ── テスト用ダミーデータ構造 ───────────────────────────────────────


@dataclass
class DummyRecord:
    """ティック番号を持つダミーレコード。"""
    tick: int = 0
    value: str = ""


@dataclass
class DummyState:
    """ダミーの内部状態。テスト用にリスト・辞書フィールドを持つ。"""
    elapsed_records: list = field(default_factory=list)
    snapshot_window: list = field(default_factory=list)
    records: list = field(default_factory=list)
    sliding_window: list = field(default_factory=list)
    pathway_last_used_tick: dict = field(default_factory=dict)


class DummyModule:
    """ダミーモジュール。state属性を持つ。"""
    def __init__(self, state: Optional[DummyState] = None):
        self.state = state or DummyState()
        self._state = self.state


def make_orchestrator(
    tick_count: int = 100,
    modules: Optional[dict[str, Any]] = None,
    session_resume_tick: Optional[int] = None,
) -> MagicMock:
    """テスト用の擬似オーケストレータを構築する。"""
    orch = MagicMock()
    orch._tick_count = tick_count
    orch._session_resume_tick = session_resume_tick

    if modules:
        for attr_name, module in modules.items():
            setattr(orch, attr_name, module)

    return orch


# ══════════════════════════════════════════════════════════════════════════════
# 基本構造テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestBasicStructure:
    """基本構造・宣言の整合性テスト。"""

    def test_consistency_check_entries_is_tuple(self):
        """静的宣言テーブルがタプルであること。"""
        entries = get_consistency_check_entries()
        assert isinstance(entries, tuple)

    def test_consistency_check_entries_not_empty(self):
        """静的宣言テーブルが空でないこと。"""
        entries = get_consistency_check_entries()
        assert len(entries) > 0

    def test_all_entries_have_check_type(self):
        """全エントリが有効な検証種別を持つこと。"""
        for entry in CONSISTENCY_CHECK_ENTRIES:
            assert isinstance(entry.check_type, CheckType)

    def test_all_entries_have_description(self):
        """全エントリが説明を持つこと。"""
        for entry in CONSISTENCY_CHECK_ENTRIES:
            assert entry.description, f"Entry missing description: {entry}"

    def test_entries_are_frozen(self):
        """エントリは不変であること。"""
        entry = CONSISTENCY_CHECK_ENTRIES[0]
        with pytest.raises(AttributeError):
            entry.check_type = CheckType.B  # type: ignore[misc]

    def test_check_type_enum_values(self):
        """CheckType列挙型が5種別を持つこと。"""
        assert len(CheckType) == 5
        assert CheckType.A.value == "tick_consistency"
        assert CheckType.B.value == "window_temporal"
        assert CheckType.C.value == "freshness_premise"
        assert CheckType.D.value == "warmup_cross_check"
        assert CheckType.E.value == "cross_field_premise"


class TestConsistencyFinding:
    """検出結果データ構造のテスト。"""

    def test_finding_to_dict(self):
        """ConsistencyFinding.to_dict() が正しい辞書を返すこと。"""
        f = ConsistencyFinding(
            check_type=CheckType.A,
            field_path="_test.field",
            fact="test fact",
        )
        d = f.to_dict()
        assert d["check_type"] == "tick_consistency"
        assert d["field_path"] == "_test.field"
        assert d["fact"] == "test fact"

    def test_result_to_dict(self):
        """ConsistencyCheckResult.to_dict() が正しい構造を返すこと。"""
        result = ConsistencyCheckResult(
            restored_tick=100,
            total_fields_checked=5,
            total_patterns_applied=10,
            findings=[
                ConsistencyFinding(CheckType.A, "f1", "fact1"),
                ConsistencyFinding(CheckType.B, "f2", "fact2"),
            ],
            summary={"tick_consistency": 1, "window_temporal": 1},
        )
        d = result.to_dict()
        assert d["restored_tick"] == 100
        assert d["total_fields_checked"] == 5
        assert d["total_patterns_applied"] == 10
        assert d["total_findings"] == 2
        assert len(d["findings"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
# ヘルパー関数テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    """内部ヘルパー関数のテスト。"""

    def test_resolve_module_state_with_state(self):
        """モジュールの内部状態が正しく取得されること。"""
        state = DummyState()
        module = DummyModule(state)
        orch = make_orchestrator(modules={"_test_module": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_test_module",
            state_sub_attr="state",
        )
        result = _resolve_module_state(orch, entry)
        assert result is state

    def test_resolve_module_state_missing_module(self):
        """存在しないモジュールではNoneが返ること。"""
        orch = make_orchestrator()
        # MagicMockは全属性にMagicMockを返すため、明示的にNoneを設定
        orch._nonexistent = None
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_nonexistent",
            state_sub_attr="state",
        )
        result = _resolve_module_state(orch, entry)
        assert result is None

    def test_get_records_list_valid(self):
        """リストフィールドが正しく取得されること。"""
        state = DummyState(records=[DummyRecord(tick=1)])
        result = _get_records_list(state, "records")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_records_list_missing_field(self):
        """存在しないフィールドではNoneが返ること。"""
        state = DummyState()
        result = _get_records_list(state, "nonexistent_field")
        assert result is None

    def test_get_tick_value_from_dataclass(self):
        """データクラスレコードからティック値が取得できること。"""
        rec = DummyRecord(tick=42)
        assert _get_tick_value(rec, "tick") == 42

    def test_get_tick_value_from_dict(self):
        """辞書レコードからティック値が取得できること。"""
        rec = {"tick": 55}
        assert _get_tick_value(rec, "tick") == 55

    def test_get_tick_value_missing(self):
        """ティックフィールドがない場合はNoneが返ること。"""
        rec = {"value": "test"}
        assert _get_tick_value(rec, "tick") is None

    def test_get_tick_value_non_numeric(self):
        """ティック値が数値でない場合はNoneが返ること。"""
        rec = {"tick": "not_a_number"}
        assert _get_tick_value(rec, "tick") is None


# ══════════════════════════════════════════════════════════════════════════════
# 検証種別A: ティック番号の数値的矛盾検出テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckTypeA:
    """検証種別A: ティック番号超過・負値検出。"""

    def test_no_findings_for_valid_records(self):
        """全レコードが有効なティック番号を持つ場合、検出なし。"""
        state = DummyState(records=[
            DummyRecord(tick=10),
            DummyRecord(tick=50),
            DummyRecord(tick=100),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="test records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 0

    def test_detect_tick_exceeding_current(self):
        """レコードのティック番号が復元ティック番号を超過している場合に検出。"""
        state = DummyState(records=[
            DummyRecord(tick=50),
            DummyRecord(tick=200),  # 超過
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="test records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 1
        assert "超過" in findings[0].fact
        assert findings[0].check_type == CheckType.A

    def test_detect_negative_tick(self):
        """レコードのティック番号が負値の場合に検出。"""
        state = DummyState(records=[
            DummyRecord(tick=-5),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="test records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 1
        assert "負値" in findings[0].fact

    def test_detect_both_negative_and_exceeding(self):
        """負値と超過が同時にある場合に両方検出。"""
        state = DummyState(records=[
            DummyRecord(tick=-1),
            DummyRecord(tick=50),
            DummyRecord(tick=999),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="test records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 2

    def test_empty_records_no_findings(self):
        """レコードが空の場合、検出なし。"""
        state = DummyState(records=[])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="test records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 0

    def test_missing_module_no_findings(self):
        """モジュールが存在しない場合、検出なし。"""
        orch = make_orchestrator(tick_count=100)
        orch._nonexistent = None
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_nonexistent",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="test records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 検証種別B: 窓内レコードの時間的整合性テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckTypeB:
    """検証種別B: 窓内レコードの単調非減少・逸脱検出。"""

    def test_monotonic_non_decreasing_no_findings(self):
        """単調非減少の場合、順序違反なし。"""
        state = DummyState(records=[
            DummyRecord(tick=10),
            DummyRecord(tick=20),
            DummyRecord(tick=30),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=35, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=100,
            description="test window",
        )
        findings = _check_type_b(orch, entry, 35)
        assert len(findings) == 0

    def test_detect_non_monotonic_ticks(self):
        """単調非減少に違反する場合に検出。"""
        state = DummyState(records=[
            DummyRecord(tick=30),
            DummyRecord(tick=20),  # 違反
            DummyRecord(tick=40),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=50, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=100,
            description="test window",
        )
        findings = _check_type_b(orch, entry, 50)
        assert any("単調非減少に違反" in f.fact for f in findings)

    def test_detect_window_gap_exceeding_threshold(self):
        """最古レコードとの差が窓サイズの3倍を超える場合に検出。"""
        state = DummyState(records=[
            DummyRecord(tick=1),
            DummyRecord(tick=500),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=500, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=30,
            description="test window",
        )
        findings = _check_type_b(orch, entry, 500)
        assert any("3倍を超過" in f.fact for f in findings)

    def test_no_gap_finding_within_threshold(self):
        """窓サイズの3倍以内なら逸脱検出なし。"""
        state = DummyState(records=[
            DummyRecord(tick=70),
            DummyRecord(tick=100),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=30,
            description="test window",
        )
        findings = _check_type_b(orch, entry, 100)
        assert not any("3倍を超過" in f.fact for f in findings)

    def test_equal_consecutive_ticks_allowed(self):
        """同一ティック番号の連続は単調非減少に含まれる（許容）。"""
        state = DummyState(records=[
            DummyRecord(tick=10),
            DummyRecord(tick=10),
            DummyRecord(tick=20),
        ])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=20, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=100,
            description="test window",
        )
        findings = _check_type_b(orch, entry, 20)
        assert not any("単調非減少に違反" in f.fact for f in findings)

    def test_empty_window_no_findings(self):
        """窓が空の場合、検出なし。"""
        state = DummyState(records=[])
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=30,
            description="test window",
        )
        findings = _check_type_b(orch, entry, 100)
        assert len(findings) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 検証種別C: 鮮度減衰の前提照合テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckTypeC:
    """検証種別C: 鮮度減衰パラメータの整合性。"""

    def test_valid_pathway_ticks_no_findings(self):
        """全経路のティック値が復元ティック番号以内の場合、検出なし。"""
        state = DummyState(pathway_last_used_tick={
            "every_tick": 95,
            "every_3": 90,
            "every_5": 85,
        })
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.C,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="pathway_last_used_tick",
            description="test pathway ticks",
        )
        findings = _check_type_c(orch, entry, 100)
        assert len(findings) == 0

    def test_detect_pathway_tick_exceeding_current(self):
        """経路のティック値が復元ティック番号を超過している場合に検出。"""
        state = DummyState(pathway_last_used_tick={
            "every_tick": 95,
            "every_3": 200,  # 超過
        })
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.C,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="pathway_last_used_tick",
            description="test pathway ticks",
        )
        findings = _check_type_c(orch, entry, 100)
        assert len(findings) == 1
        assert "超過" in findings[0].fact

    def test_detect_negative_pathway_tick(self):
        """経路のティック値が負値の場合に検出。"""
        state = DummyState(pathway_last_used_tick={
            "every_tick": -10,
        })
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.C,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="pathway_last_used_tick",
            description="test pathway ticks",
        )
        findings = _check_type_c(orch, entry, 100)
        assert len(findings) == 1
        assert "負値" in findings[0].fact

    def test_empty_pathway_map_no_findings(self):
        """経路マップが空の場合、検出なし。"""
        state = DummyState(pathway_last_used_tick={})
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.C,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="pathway_last_used_tick",
            description="test pathway ticks",
        )
        findings = _check_type_c(orch, entry, 100)
        assert len(findings) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 検証種別D: ウォームアップ結果の照合テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckTypeD:
    """検証種別D: ウォームアップ失敗の検出。"""

    def test_no_findings_when_all_derived(self):
        """全てderivedの場合、検出なし。"""
        warmup_results = {
            "_last_self_image": "derived",
            "_last_coherence": "derived",
        }
        findings = _check_type_d(warmup_results)
        assert len(findings) == 0

    def test_detect_failed_warmup_entries(self):
        """failedエントリが検出されること。"""
        warmup_results = {
            "_last_self_image": "derived",
            "_last_coherence": "failed",
            "_last_narrative": "failed",
            "_last_percept": "skipped",
        }
        findings = _check_type_d(warmup_results)
        assert len(findings) == 2
        assert all(f.check_type == CheckType.D for f in findings)
        field_paths = {f.field_path for f in findings}
        assert "_last_coherence" in field_paths
        assert "_last_narrative" in field_paths

    def test_no_findings_with_empty_results(self):
        """空のウォームアップ結果では検出なし。"""
        findings = _check_type_d({})
        assert len(findings) == 0

    def test_skipped_and_empty_source_not_detected(self):
        """skippedとempty_sourceはfailureとして検出されないこと。"""
        warmup_results = {
            "_last_percept": "skipped",
            "_last_self_image": "empty_source",
        }
        findings = _check_type_d(warmup_results)
        assert len(findings) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 検証種別E: フィールド間の数値的前提照合テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckTypeE:
    """検証種別E: フィールド間整合性。"""

    def test_valid_tick_count_no_findings(self):
        """正のティック番号では検出なし。"""
        orch = make_orchestrator(tick_count=100)
        findings = _check_type_e(orch, 100)
        assert len(findings) == 0

    def test_detect_negative_tick_count(self):
        """負のティック番号が検出されること。"""
        orch = make_orchestrator(tick_count=-5)
        findings = _check_type_e(orch, -5)
        assert len(findings) == 1
        assert "負値" in findings[0].fact

    def test_detect_resume_tick_exceeding_current(self):
        """セッション復帰ティックが現在ティックを超過している場合に検出。"""
        orch = make_orchestrator(tick_count=100, session_resume_tick=200)
        findings = _check_type_e(orch, 100)
        assert len(findings) == 1
        assert "超過" in findings[0].fact

    def test_valid_resume_tick_no_findings(self):
        """セッション復帰ティックが現在ティック以下の場合、検出なし。"""
        orch = make_orchestrator(tick_count=100, session_resume_tick=50)
        findings = _check_type_e(orch, 100)
        assert len(findings) == 0

    def test_zero_tick_count_no_findings(self):
        """ティック番号0は有効。"""
        orch = make_orchestrator(tick_count=0)
        findings = _check_type_e(orch, 0)
        assert len(findings) == 0

    def test_no_resume_tick_no_findings(self):
        """復帰ティックがNoneの場合、検出なし。"""
        orch = make_orchestrator(tick_count=100, session_resume_tick=None)
        findings = _check_type_e(orch, 100)
        assert len(findings) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 統合テスト: execute_session_recovery_check
# ══════════════════════════════════════════════════════════════════════════════


class TestExecuteSessionRecoveryCheck:
    """execute_session_recovery_check() の統合テスト。"""

    def test_clean_state_no_findings(self):
        """全て正常な状態ではfinding 0件。"""
        state = DummyState(
            records=[DummyRecord(tick=10), DummyRecord(tick=20)],
            elapsed_records=[DummyRecord(tick=5), DummyRecord(tick=15)],
            snapshot_window=[DummyRecord(tick=5), DummyRecord(tick=10)],
            sliding_window=[DummyRecord(tick=5), DummyRecord(tick=15)],
            pathway_last_used_tick={"every_tick": 20},
        )
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=50, modules=modules)
        warmup_results = {"_last_self_image": "derived"}

        result = execute_session_recovery_check(orch, warmup_results)

        assert isinstance(result, ConsistencyCheckResult)
        assert result.restored_tick == 50
        assert result.total_findings == 0
        assert result.total_patterns_applied > 0

    def test_multiple_inconsistencies_detected(self):
        """複数の不整合が同時に検出されること。"""
        # ティック番号超過のレコード
        bad_state = DummyState(
            records=[DummyRecord(tick=200)],  # 超過
            elapsed_records=[DummyRecord(tick=200)],  # 超過
            snapshot_window=[DummyRecord(tick=10)],
            sliding_window=[DummyRecord(tick=10)],
            pathway_last_used_tick={"every_tick": 200},  # 超過
        )
        module = DummyModule(bad_state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)

        result = execute_session_recovery_check(orch, {})
        assert result.total_findings > 0

    def test_warmup_failures_included(self):
        """ウォームアップ失敗がD種別として含まれること。"""
        state = DummyState()
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)
        warmup_results = {
            "_last_self_image": "failed",
            "_last_coherence": "failed",
        }

        result = execute_session_recovery_check(orch, warmup_results)
        d_findings = [
            f for f in result.findings if f.check_type == CheckType.D
        ]
        assert len(d_findings) == 2

    def test_no_warmup_results_skips_d_check(self):
        """warmup_results=None の場合、D種別はスキップされること。"""
        state = DummyState()
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)

        result = execute_session_recovery_check(orch, None)
        assert result.summary.get("warmup_cross_check", 0) == 0

    def test_result_summary_has_all_check_types(self):
        """結果のsummaryが全検証種別を含むこと。"""
        state = DummyState()
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)

        result = execute_session_recovery_check(orch, {})
        for ct in CheckType:
            assert ct.value in result.summary


# ══════════════════════════════════════════════════════════════════════════════
# 安全弁テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestSafetyValves:
    """安全弁の検証。"""

    def test_no_state_modification(self):
        """検証がオーケストレータの状態を変更しないこと。"""
        state = DummyState(
            records=[DummyRecord(tick=200)],  # 不整合あり
            pathway_last_used_tick={"a": 300},  # 不整合あり
        )
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)

        # 状態のスナップショット
        original_tick = orch._tick_count
        original_records = list(state.records)
        original_pathway = dict(state.pathway_last_used_tick)

        # 検証実行
        execute_session_recovery_check(orch, {})

        # 状態が変更されていないこと
        assert orch._tick_count == original_tick
        assert state.records == original_records
        assert state.pathway_last_used_tick == original_pathway

    def test_entries_are_immutable(self):
        """宣言テーブルのエントリが不変であること。"""
        for entry in CONSISTENCY_CHECK_ENTRIES:
            with pytest.raises(AttributeError):
                entry.check_type = CheckType.A  # type: ignore[misc]

    def test_exception_in_check_does_not_propagate(self):
        """検証中の例外がexecute関数から漏れないこと。"""
        orch = MagicMock()
        orch._tick_count = 100
        orch._session_resume_tick = None

        # side_effect で例外を発生させるモジュール
        class BrokenModule:
            @property
            def state(self):
                raise RuntimeError("Broken!")

        orch._temporal_cognition = BrokenModule()
        orch._introspection_cross_section = BrokenModule()
        orch._self_action_recorder = BrokenModule()
        orch._intent_action_gap_recorder = BrokenModule()
        orch._emotional_backdrop_processor = BrokenModule()
        orch._drive_variation_processor = BrokenModule()
        orch._selection_attribution_recorder = BrokenModule()

        # 例外が漏れずに結果が返ること
        result = execute_session_recovery_check(orch, {})
        assert isinstance(result, ConsistencyCheckResult)

    def test_check_is_pure_function(self):
        """同一入力に対して同一出力を返すこと（純粋関数性）。"""
        state = DummyState(
            records=[DummyRecord(tick=200)],
        )
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)
        warmup = {"_last_self_image": "derived"}

        result1 = execute_session_recovery_check(orch, warmup)
        result2 = execute_session_recovery_check(orch, warmup)

        assert result1.total_findings == result2.total_findings
        assert result1.restored_tick == result2.restored_tick
        assert len(result1.findings) == len(result2.findings)

    def test_no_normative_judgments_in_findings(self):
        """検出結果に規範的判断語（「正常」「異常」「問題」等）が含まれないこと。"""
        state = DummyState(
            records=[DummyRecord(tick=200)],
            pathway_last_used_tick={"a": 300},
        )
        module = DummyModule(state)
        modules = {
            "_temporal_cognition": module,
            "_introspection_cross_section": module,
            "_self_action_recorder": module,
            "_intent_action_gap_recorder": module,
            "_emotional_backdrop_processor": module,
            "_drive_variation_processor": module,
            "_selection_attribution_recorder": module,
        }
        orch = make_orchestrator(tick_count=100, modules=modules)

        result = execute_session_recovery_check(orch, {})
        normative_words = ["正常", "異常", "問題", "エラー", "不正", "望ましい"]
        for finding in result.findings:
            for word in normative_words:
                assert word not in finding.fact, (
                    f"規範的語彙 '{word}' が検出結果に含まれている: {finding.fact}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 辞書レコードとの互換性テスト
# ══════════════════════════════════════════════════════════════════════════════


class TestDictRecordCompatibility:
    """辞書形式のレコードとの互換性テスト。"""

    def test_check_a_with_dict_records(self):
        """辞書形式のレコードでもティック番号が検出されること。"""
        state = DummyState()
        state.records = [
            {"tick": 50, "value": "a"},
            {"tick": 200, "value": "b"},  # 超過
        ]
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=100, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.A,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            description="dict records",
        )
        findings = _check_type_a(orch, entry, 100)
        assert len(findings) == 1

    def test_check_b_with_dict_records(self):
        """辞書形式のレコードで単調非減少も検出されること。"""
        state = DummyState()
        state.records = [
            {"tick": 30},
            {"tick": 20},  # 違反
            {"tick": 40},
        ]
        module = DummyModule(state)
        orch = make_orchestrator(tick_count=50, modules={"_mod": module})
        entry = ConsistencyCheckEntry(
            check_type=CheckType.B,
            module_attr="_mod",
            state_sub_attr="state",
            records_field="records",
            tick_field="tick",
            window_size=100,
            description="dict window",
        )
        findings = _check_type_b(orch, entry, 50)
        assert any("単調非減少に違反" in f.fact for f in findings)
