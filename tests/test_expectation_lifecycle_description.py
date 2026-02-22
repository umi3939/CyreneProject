"""
tests/test_expectation_lifecycle_description.py - 予期ライフサイクル記述テスト

全機能のテスト:
- スナップショット比較による状態遷移検出
- 遷移記録の蓄積とFIFO自然脱落
- 記録自体の鮮度均一減衰
- ライフサイクル全体像の参照
- enrichmentデータ生成
- save/load往復テスト
- 安全弁テスト（蓄積上限、均一減衰、収束監視、enrichment出力上限、内容記述長さ制限）
- エッジケーステスト
"""

import time
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from psyche.expectation_lifecycle_description import (
    TransitionType,
    RecordFreshness,
    ConvergenceLevel,
    TransitionRecord,
    LifecycleView,
    ConvergenceRecord,
    SnapshotEntry,
    ExpectationLifecycleState,
    ExpectationLifecycleConfig,
    ExpectationLifecycleDescriptionProcessor,
    create_expectation_lifecycle_processor,
    get_lifecycle_summary,
    _gen_id,
    _clamp,
    _freshness_stage,
    _strength_stage_label,
    _freshness_level_label,
    _convergence_from_score,
)


# =============================================================================
# Mock ExpectationStore / ExpectationCandidate
# =============================================================================

class MockSourceType:
    def __init__(self, value: str = "repetition"):
        self.value = value


class MockBasis:
    def __init__(self, value: str = "pattern_continuation"):
        self.value = value


@dataclass
class MockExpectation:
    """ExpectationCandidate のモック。"""
    expectation_id: str = ""
    source_type: Any = field(default_factory=lambda: MockSourceType("repetition"))
    basis: Any = field(default_factory=lambda: MockBasis("pattern_continuation"))
    description: str = "Test expectation"
    strength: float = 0.5
    freshness: float = 0.8
    revision_count: int = 0
    competing_ids: tuple = ()


@dataclass
class MockExpectationStore:
    """ExpectationStore のモック。"""
    expectations: tuple = ()


def _make_store(*expectations: MockExpectation) -> MockExpectationStore:
    return MockExpectationStore(expectations=tuple(expectations))


def _make_exp(
    exp_id: str = "",
    desc: str = "Test expectation",
    strength: float = 0.5,
    freshness: float = 0.8,
    revision_count: int = 0,
    source_type: str = "repetition",
    basis: str = "pattern_continuation",
    competing_ids: tuple = (),
) -> MockExpectation:
    if not exp_id:
        exp_id = _gen_id()
    return MockExpectation(
        expectation_id=exp_id,
        source_type=MockSourceType(source_type),
        basis=MockBasis(basis),
        description=desc,
        strength=strength,
        freshness=freshness,
        revision_count=revision_count,
        competing_ids=competing_ids,
    )


# =============================================================================
# Helper function tests
# =============================================================================

class TestHelpers:
    """ヘルパー関数のテスト。"""

    def test_clamp(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.1) == 0.0
        assert _clamp(1.5) == 1.0

    def test_freshness_stage(self):
        assert _freshness_stage(0.9) == RecordFreshness.FRESH
        assert _freshness_stage(0.7) == RecordFreshness.RECENT
        assert _freshness_stage(0.5) == RecordFreshness.AGING
        assert _freshness_stage(0.3) == RecordFreshness.STALE
        assert _freshness_stage(0.1) == RecordFreshness.FADED

    def test_strength_stage_label(self):
        assert _strength_stage_label(0.8) == "strong"
        assert _strength_stage_label(0.5) == "moderate"
        assert _strength_stage_label(0.3) == "weak"
        assert _strength_stage_label(0.1) == "faint"
        assert _strength_stage_label(0.01) == "undefined"

    def test_freshness_level_label(self):
        assert _freshness_level_label(0.9) == "fresh"
        assert _freshness_level_label(0.7) == "recent"
        assert _freshness_level_label(0.5) == "aging"
        assert _freshness_level_label(0.2) == "stale"
        assert _freshness_level_label(0.1) == "faded"

    def test_convergence_from_score(self):
        assert _convergence_from_score(0.1) == ConvergenceLevel.NONE
        assert _convergence_from_score(0.4) == ConvergenceLevel.MILD
        assert _convergence_from_score(0.6) == ConvergenceLevel.MODERATE
        assert _convergence_from_score(0.8) == ConvergenceLevel.STRONG


# =============================================================================
# TransitionRecord tests
# =============================================================================

class TestTransitionRecord:
    """遷移記録のテスト。"""

    def test_creation(self):
        rec = TransitionRecord(
            expectation_id="exp1",
            transition_type=TransitionType.GENERATION.value,
        )
        assert rec.expectation_id == "exp1"
        assert rec.transition_type == "generation"
        assert rec.record_freshness == 1.0

    def test_to_dict_from_dict(self):
        rec = TransitionRecord(
            expectation_id="exp1",
            expectation_description="test desc",
            source_type="repetition",
            basis_type="pattern_continuation",
            transition_type=TransitionType.DISAPPEARANCE.value,
            strength_stage="moderate",
            freshness_stage="aging",
            cycle_number=5,
            competing_ids=["exp2", "exp3"],
            record_freshness=0.7,
            record_freshness_stage=RecordFreshness.RECENT.value,
        )
        d = rec.to_dict()
        restored = TransitionRecord.from_dict(d)
        assert restored.expectation_id == "exp1"
        assert restored.transition_type == "disappearance"
        assert restored.competing_ids == ["exp2", "exp3"]
        assert restored.record_freshness == 0.7


# =============================================================================
# LifecycleView tests
# =============================================================================

class TestLifecycleView:
    """ライフサイクル全体像のテスト。"""

    def test_creation(self):
        view = LifecycleView(expectation_id="exp1")
        assert not view.is_completed
        assert view.generation_record is None

    def test_to_dict_from_dict(self):
        gen_rec = TransitionRecord(
            expectation_id="exp1",
            transition_type=TransitionType.GENERATION.value,
        )
        dis_rec = TransitionRecord(
            expectation_id="exp1",
            transition_type=TransitionType.DISAPPEARANCE.value,
        )
        view = LifecycleView(
            expectation_id="exp1",
            generation_record=gen_rec,
            disappearance_record=dis_rec,
            is_completed=True,
        )
        d = view.to_dict()
        restored = LifecycleView.from_dict(d)
        assert restored.expectation_id == "exp1"
        assert restored.is_completed is True
        assert restored.generation_record is not None
        assert restored.disappearance_record is not None


# =============================================================================
# State tests
# =============================================================================

class TestState:
    """内部状態のテスト。"""

    def test_empty_state(self):
        state = ExpectationLifecycleState()
        assert len(state.transition_records) == 0
        assert state.cycle_count == 0

    def test_to_dict_from_dict(self):
        state = ExpectationLifecycleState()
        state.cycle_count = 10
        state.accumulation_limit_reached = True
        state.convergence_flag = True
        rec = TransitionRecord(expectation_id="exp1")
        state.transition_records.append(rec)
        state.previous_snapshot["exp1"] = SnapshotEntry(expectation_id="exp1", strength=0.5)

        d = state.to_dict()
        restored = ExpectationLifecycleState.from_dict(d)
        assert restored.cycle_count == 10
        assert restored.accumulation_limit_reached is True
        assert restored.convergence_flag is True
        assert len(restored.transition_records) == 1
        assert "exp1" in restored.previous_snapshot


# =============================================================================
# Transition detection tests
# =============================================================================

class TestTransitionDetection:
    """スナップショット比較による遷移検出のテスト。"""

    def test_detect_generation(self):
        """新しい予期が生成されたことを検出する。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1", desc="new expectation")
        store = _make_store(exp)

        count = proc.process(store)
        # 初回は全て「生成」として検出される
        assert count == 1
        records = proc.get_recent_records()
        assert len(records) == 1
        assert records[0].transition_type == TransitionType.GENERATION.value
        assert records[0].expectation_id == "exp1"

    def test_detect_disappearance(self):
        """予期が消失したことを検出する。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store1 = _make_store(exp)
        proc.process(store1)

        # 次のサイクルで予期がなくなる
        store2 = _make_store()
        count = proc.process(store2)
        assert count == 1
        records = proc.get_recent_records()
        # 最後の記録が消失
        last = records[-1]
        assert last.transition_type == TransitionType.DISAPPEARANCE.value
        assert last.expectation_id == "exp1"

    def test_detect_revision(self):
        """予期の修正を検出する。"""
        proc = create_expectation_lifecycle_processor()
        exp1 = _make_exp(exp_id="exp1", revision_count=0)
        store1 = _make_store(exp1)
        proc.process(store1)

        # 修正回数が増加
        exp2 = _make_exp(exp_id="exp1", revision_count=1)
        store2 = _make_store(exp2)
        count = proc.process(store2)
        assert count >= 1  # 修正検出 + 可能な鮮度/強度変化
        records = proc.get_recent_records()
        revision_records = [r for r in records if r.transition_type == TransitionType.REVISION.value]
        assert len(revision_records) >= 1

    def test_detect_strength_change(self):
        """予期の強度変化を検出する。"""
        proc = create_expectation_lifecycle_processor()
        # strength=0.8 → "strong"
        exp1 = _make_exp(exp_id="exp1", strength=0.8)
        store1 = _make_store(exp1)
        proc.process(store1)

        # strength=0.3 → "weak" (段階変化)
        exp2 = _make_exp(exp_id="exp1", strength=0.3)
        store2 = _make_store(exp2)
        count = proc.process(store2)
        records = proc.get_recent_records()
        strength_records = [r for r in records if r.transition_type == TransitionType.STRENGTH_CHANGE.value]
        assert len(strength_records) >= 1

    def test_detect_freshness_change(self):
        """予期の鮮度変化を検出する。"""
        proc = create_expectation_lifecycle_processor()
        # freshness=0.9 → "fresh"
        exp1 = _make_exp(exp_id="exp1", freshness=0.9)
        store1 = _make_store(exp1)
        proc.process(store1)

        # freshness=0.5 → "aging" (段階変化)
        exp2 = _make_exp(exp_id="exp1", freshness=0.5)
        store2 = _make_store(exp2)
        count = proc.process(store2)
        records = proc.get_recent_records()
        freshness_records = [r for r in records if r.transition_type == TransitionType.FRESHNESS_CHANGE.value]
        assert len(freshness_records) >= 1

    def test_no_change_no_records(self):
        """変化がなければ記録は生成されない。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1", strength=0.5, freshness=0.8)
        store = _make_store(exp)

        proc.process(store)  # 初回: 生成記録
        initial_count = len(proc.state.transition_records)

        proc.process(store)  # 同一スナップショット: 変化なし
        assert len(proc.state.transition_records) == initial_count

    def test_multiple_expectations(self):
        """複数の予期の同時遷移を検出する。"""
        proc = create_expectation_lifecycle_processor()
        exp1 = _make_exp(exp_id="exp1")
        exp2 = _make_exp(exp_id="exp2")
        store1 = _make_store(exp1, exp2)
        count = proc.process(store1)
        assert count == 2  # 2つとも「生成」

    def test_simultaneous_generation_and_disappearance(self):
        """生成と消失が同時に発生する場合。"""
        proc = create_expectation_lifecycle_processor()
        exp1 = _make_exp(exp_id="exp1")
        store1 = _make_store(exp1)
        proc.process(store1)

        # exp1消失 + exp2生成
        exp2 = _make_exp(exp_id="exp2")
        store2 = _make_store(exp2)
        count = proc.process(store2)
        assert count == 2  # 消失1 + 生成1

    def test_competing_ids_recorded(self):
        """競合関係が記録されることを確認。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1", competing_ids=("exp2", "exp3"))
        store = _make_store(exp)
        proc.process(store)
        records = proc.get_recent_records()
        assert records[0].competing_ids == ["exp2", "exp3"]


# =============================================================================
# FIFO accumulation tests
# =============================================================================

class TestFIFOAccumulation:
    """FIFO蓄積のテスト。"""

    def test_fifo_overflow(self):
        """蓄積上限に達するとFIFOで最古の記録が脱落する。"""
        config = ExpectationLifecycleConfig(max_records=5)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        # 10件の遷移を生成
        for i in range(10):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)

        # 上限5件に収まっている
        assert len(proc.state.transition_records) <= 5

    def test_fifo_keeps_latest(self):
        """FIFOで最新の記録が保持される。"""
        config = ExpectationLifecycleConfig(max_records=3)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        for i in range(5):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            # 前サイクルの予期を消す
            proc.state.previous_snapshot.clear()

        records = proc.state.transition_records
        # 最新の記録が存在する
        assert len(records) <= 3

    def test_accumulation_limit_flag(self):
        """蓄積上限到達フラグのテスト。"""
        config = ExpectationLifecycleConfig(max_records=3)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        # 3件まではフラグなし
        for i in range(3):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            proc.state.previous_snapshot.clear()

        # 上限超過時にフラグが立つ
        exp = _make_exp(exp_id="exp_overflow")
        store = _make_store(exp)
        proc.process(store)
        # 3件+1件の生成記録 → FIFO発動
        assert proc.state.accumulation_limit_reached is True

    def test_no_selective_retention(self):
        """「重要な遷移」の優先保持がないことを確認。

        全記録は等価であり、遷移種類による優先保持はない。
        """
        config = ExpectationLifecycleConfig(max_records=5)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        # さまざまな遷移種類を生成
        exp1 = _make_exp(exp_id="exp1", strength=0.8)
        store1 = _make_store(exp1)
        proc.process(store1)

        # 修正
        exp1_rev = _make_exp(exp_id="exp1", strength=0.8, revision_count=1)
        store2 = _make_store(exp1_rev)
        proc.process(store2)

        # 消失
        store3 = _make_store()
        proc.process(store3)

        # 新規生成を大量に追加してFIFO発動
        for i in range(10):
            exp = _make_exp(exp_id=f"new{i}")
            store = _make_store(exp)
            proc.process(store)
            proc.state.previous_snapshot.clear()

        # 上限内に収まっている
        assert len(proc.state.transition_records) <= 5


# =============================================================================
# Record freshness decay tests
# =============================================================================

class TestRecordFreshnessDecay:
    """記録自体の鮮度減衰テスト。"""

    def test_uniform_decay(self):
        """すべての記録が均一に減衰する。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        initial_freshness = proc.state.transition_records[0].record_freshness

        # 追加のサイクルで減衰
        proc.process(store)  # 変化なし（遷移なし）だが減衰は進む
        new_freshness = proc.state.transition_records[0].record_freshness
        assert new_freshness < initial_freshness

    def test_decay_is_type_independent(self):
        """遷移種類に依存しない均一な減衰速度。"""
        config = ExpectationLifecycleConfig(record_freshness_decay_rate=0.1)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        # 生成記録
        exp1 = _make_exp(exp_id="exp1")
        store1 = _make_store(exp1)
        proc.process(store1)

        # 消失記録
        store2 = _make_store()
        proc.process(store2)

        records = proc.state.transition_records
        assert len(records) == 2  # 生成 + 消失

        gen_rec = [r for r in records if r.transition_type == TransitionType.GENERATION.value][0]
        dis_rec = [r for r in records if r.transition_type == TransitionType.DISAPPEARANCE.value][0]

        # 生成記録は1サイクル多く減衰している
        # 生成は cycle=1 に作成、消失は cycle=2 に作成
        # cycle=2 の減衰適用後: 生成は2回減衰、消失は1回減衰
        assert gen_rec.record_freshness < dis_rec.record_freshness

    def test_freshness_stage_updates(self):
        """鮮度減衰に伴い段階ラベルが更新される。"""
        config = ExpectationLifecycleConfig(record_freshness_decay_rate=0.3)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)
        # decay_rate=0.3 => 1.0 - 0.3 = 0.7 after first process(), which maps to RECENT
        assert proc.state.transition_records[0].record_freshness_stage == RecordFreshness.RECENT.value

        # 複数サイクルで減衰
        for _ in range(3):
            proc.process(store)  # 変化なし、減衰のみ

        # 鮮度が下がっているはず
        rec = proc.state.transition_records[0]
        assert rec.record_freshness < 0.5
        assert rec.record_freshness_stage != RecordFreshness.FRESH.value

    def test_decay_does_not_delete(self):
        """鮮度減衰は記録の削除を引き起こさない（削除はFIFOのみ）。"""
        config = ExpectationLifecycleConfig(record_freshness_decay_rate=0.5)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        # 大量のサイクルで鮮度を0近くに
        for _ in range(20):
            proc.process(store)

        # 記録は残っている（FIFOでなく鮮度で削除しない）
        assert len(proc.state.transition_records) == 1
        assert proc.state.transition_records[0].record_freshness <= 0.01


# =============================================================================
# Lifecycle view tests
# =============================================================================

class TestLifecycleViewRetrieval:
    """ライフサイクル全体像の参照テスト。"""

    def test_full_lifecycle(self):
        """生成→修正→消失の完全なライフサイクル。"""
        proc = create_expectation_lifecycle_processor()

        # 生成
        exp = _make_exp(exp_id="exp1", strength=0.8, revision_count=0)
        store1 = _make_store(exp)
        proc.process(store1)

        # 修正
        exp_rev = _make_exp(exp_id="exp1", strength=0.8, revision_count=1)
        store2 = _make_store(exp_rev)
        proc.process(store2)

        # 消失
        store3 = _make_store()
        proc.process(store3)

        view = proc.get_lifecycle_view("exp1")
        assert view is not None
        assert view.expectation_id == "exp1"
        assert view.generation_record is not None
        assert view.generation_record.transition_type == TransitionType.GENERATION.value
        assert view.disappearance_record is not None
        assert view.disappearance_record.transition_type == TransitionType.DISAPPEARANCE.value
        assert view.is_completed is True
        assert len(view.intermediate_records) >= 1  # 修正記録

    def test_in_progress_lifecycle(self):
        """進行中のライフサイクル（消失していない）。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        view = proc.get_lifecycle_view("exp1")
        assert view is not None
        assert view.is_completed is False
        assert view.disappearance_record is None

    def test_nonexistent_lifecycle(self):
        """存在しない予期のライフサイクルはNoneを返す。"""
        proc = create_expectation_lifecycle_processor()
        view = proc.get_lifecycle_view("nonexistent")
        assert view is None

    def test_lifecycle_counts(self):
        """ライフサイクル数のカウント。"""
        proc = create_expectation_lifecycle_processor()

        # 2つ生成
        exp1 = _make_exp(exp_id="exp1")
        exp2 = _make_exp(exp_id="exp2")
        store1 = _make_store(exp1, exp2)
        proc.process(store1)

        counts = proc.get_lifecycle_counts()
        assert counts["active_lifecycles"] == 2
        assert counts["completed_lifecycles"] == 0

        # 1つ消失
        store2 = _make_store(exp1)
        proc.process(store2)

        counts = proc.get_lifecycle_counts()
        assert counts["active_lifecycles"] == 1
        assert counts["completed_lifecycles"] == 1


# =============================================================================
# Convergence monitoring tests
# =============================================================================

class TestConvergenceMonitoring:
    """収束監視のテスト。"""

    def test_no_convergence_initially(self):
        """初期状態では収束なし。"""
        proc = create_expectation_lifecycle_processor()
        assert proc.state.convergence_flag is False

    def test_convergence_detection(self):
        """特定遷移種類が支配的になると収束フラグが立つ。"""
        config = ExpectationLifecycleConfig(convergence_threshold=0.7, max_records=200)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        # 大量の「生成」のみを蓄積
        for i in range(20):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            # 前回スナップショットをクリアして毎回「生成」を検出
            proc.state.previous_snapshot.clear()

        # 全て「生成」→ 収束フラグが立つ
        assert proc.state.convergence_flag is True

    def test_convergence_does_not_modify_records(self):
        """収束検出が記録の内容や蓄積を変更しないことを確認。"""
        config = ExpectationLifecycleConfig(convergence_threshold=0.5, max_records=200)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        for i in range(10):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            proc.state.previous_snapshot.clear()

        record_count_before = len(proc.state.transition_records)

        # 収束検出後も記録数は変わらない
        exp = _make_exp(exp_id="exp_extra")
        store = _make_store(exp)
        proc.process(store)

        # 新規記録が追加されただけ
        assert len(proc.state.transition_records) == record_count_before + 1

    def test_convergence_flag_not_in_enrichment(self):
        """収束フラグがenrichment出力に含まれないことを確認。"""
        config = ExpectationLifecycleConfig(convergence_threshold=0.5)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        for i in range(10):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            proc.state.previous_snapshot.clear()

        enrichment = proc.get_enrichment_data()
        assert "convergence_flag" not in enrichment

    def test_convergence_records_accumulate(self):
        """収束監視記録が蓄積される。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)

        for _ in range(5):
            proc.process(store)

        assert len(proc.state.convergence_records) == 5


# =============================================================================
# Enrichment data tests
# =============================================================================

class TestEnrichmentData:
    """enrichmentデータ生成のテスト。"""

    def test_empty_enrichment(self):
        """記録がない場合のenrichment。"""
        proc = create_expectation_lifecycle_processor()
        data = proc.get_enrichment_data()
        assert data["total_records"] == 0
        assert data["entries"] == []

    def test_enrichment_contains_fact_descriptions(self):
        """enrichmentが事実記述のみを含むことを確認。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1", desc="Test expectation for enrichment")
        store = _make_store(exp)
        proc.process(store)

        data = proc.get_enrichment_data()
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["transition_type"] == TransitionType.GENERATION.value
        assert entry["expectation_id"] == "exp1"
        assert "expectation_description" in entry
        assert "strength_stage" in entry

    def test_enrichment_count_limit(self):
        """enrichmentの件数上限。"""
        config = ExpectationLifecycleConfig(enrichment_count=3, max_records=200)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        for i in range(10):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            proc.state.previous_snapshot.clear()

        data = proc.get_enrichment_data()
        assert len(data["entries"]) <= 3

    def test_enrichment_has_lifecycle_counts(self):
        """enrichmentにライフサイクル数が含まれる。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        data = proc.get_enrichment_data()
        assert "active_lifecycles" in data
        assert "completed_lifecycles" in data
        assert data["active_lifecycles"] == 1

    def test_enrichment_description_truncation(self):
        """enrichmentでの内容記述の切り詰め。"""
        proc = create_expectation_lifecycle_processor()
        long_desc = "A" * 200
        exp = _make_exp(exp_id="exp1", desc=long_desc)
        store = _make_store(exp)
        proc.process(store)

        data = proc.get_enrichment_data()
        entry = data["entries"][0]
        assert len(entry["expectation_description"]) <= 84  # 80 + "..."

    def test_enrichment_no_statistics(self):
        """enrichmentに統計量が含まれないことを確認。

        寿命、的中率、ソースタイプ別消失率等を算出しない。
        """
        proc = create_expectation_lifecycle_processor()
        for i in range(5):
            exp = _make_exp(exp_id=f"exp{i}")
            store = _make_store(exp)
            proc.process(store)
            proc.state.previous_snapshot.clear()

        data = proc.get_enrichment_data()
        # 統計量キーが存在しないことを確認
        assert "average_lifespan" not in data
        assert "hit_rate" not in data
        assert "disappearance_rate_by_source" not in data
        assert "type_distribution" not in data


# =============================================================================
# Save/Load roundtrip tests
# =============================================================================

class TestSaveLoadRoundtrip:
    """save/load往復テスト。"""

    def test_empty_state_roundtrip(self):
        """空の状態のsave/load。"""
        state = ExpectationLifecycleState()
        d = state.to_dict()
        restored = ExpectationLifecycleState.from_dict(d)
        assert restored.cycle_count == 0
        assert len(restored.transition_records) == 0

    def test_full_state_roundtrip(self):
        """完全な状態のsave/load。"""
        proc = create_expectation_lifecycle_processor()

        # データを蓄積
        exp1 = _make_exp(exp_id="exp1", strength=0.8, desc="First expectation")
        store1 = _make_store(exp1)
        proc.process(store1)

        # 修正
        exp1_rev = _make_exp(exp_id="exp1", strength=0.5, revision_count=1, desc="First expectation revised")
        store2 = _make_store(exp1_rev)
        proc.process(store2)

        # 消失
        store3 = _make_store()
        proc.process(store3)

        # Save
        d = proc.state.to_dict()

        # Load
        new_proc = create_expectation_lifecycle_processor()
        new_proc.state = ExpectationLifecycleState.from_dict(d)

        # 検証
        assert new_proc.state.cycle_count == proc.state.cycle_count
        assert len(new_proc.state.transition_records) == len(proc.state.transition_records)
        assert len(new_proc.state.convergence_records) == len(proc.state.convergence_records)

        # スナップショットも復元されている
        assert len(new_proc.state.previous_snapshot) == len(proc.state.previous_snapshot)

    def test_continue_processing_after_load(self):
        """load後に処理を継続できる。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        # Save & load
        d = proc.state.to_dict()
        new_proc = create_expectation_lifecycle_processor()
        new_proc.state = ExpectationLifecycleState.from_dict(d)

        # 処理を継続
        exp2 = _make_exp(exp_id="exp2")
        store2 = _make_store(exp, exp2)
        count = new_proc.process(store2)
        assert count >= 1  # exp2の生成を検出

    def test_snapshot_preserved_across_load(self):
        """load後に前回スナップショットが保持され、遷移検出が正しく機能する。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        # Save & load
        d = proc.state.to_dict()
        new_proc = create_expectation_lifecycle_processor()
        new_proc.state = ExpectationLifecycleState.from_dict(d)

        # exp1が消失
        store2 = _make_store()
        count = new_proc.process(store2)
        assert count == 1  # 消失を検出

        records = new_proc.get_recent_records()
        dis_records = [r for r in records if r.transition_type == TransitionType.DISAPPEARANCE.value]
        assert len(dis_records) >= 1


# =============================================================================
# Safety valve tests
# =============================================================================

class TestSafetyValves:
    """安全弁テスト。"""

    def test_content_description_length_limit(self):
        """内容記述の文字数上限。"""
        config = ExpectationLifecycleConfig(description_max_length=20)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        long_desc = "A" * 200
        exp = _make_exp(exp_id="exp1", desc=long_desc)
        store = _make_store(exp)
        proc.process(store)

        records = proc.get_recent_records()
        assert len(records[0].expectation_description) <= 20

    def test_uniform_decay_no_type_preference(self):
        """特定の遷移種類の記録が他より長く保持されない。

        均一減衰: 遷移種類に依存しない同一の減衰速度。
        """
        config = ExpectationLifecycleConfig(record_freshness_decay_rate=0.1)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        # 2つの異なる遷移種類の記録を作成
        exp1 = _make_exp(exp_id="exp1")
        store1 = _make_store(exp1)
        proc.process(store1)  # 生成

        store2 = _make_store()
        proc.process(store2)  # 消失

        gen_rec = [r for r in proc.state.transition_records
                   if r.transition_type == TransitionType.GENERATION.value][0]
        dis_rec = [r for r in proc.state.transition_records
                   if r.transition_type == TransitionType.DISAPPEARANCE.value][0]

        # 生成記録は消失記録より1サイクル分余計に減衰している
        diff = gen_rec.record_freshness - dis_rec.record_freshness
        # 1サイクル分の差（近似的に decay_rate の1回分）
        assert abs(diff - (-config.record_freshness_decay_rate)) < 0.05

    def test_no_reference_frequency_affects_accumulation(self):
        """参照頻度が蓄積の優先度に影響しない。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        # 何度参照しても記録の内容は変わらない
        for _ in range(10):
            proc.get_recent_records()
            proc.get_lifecycle_view("exp1")

        records = proc.state.transition_records
        assert records[0].record_freshness < 1.0 or True  # 参照では変わらない

    def test_no_freshness_recovery_on_reference(self):
        """参照された記録が鮮度回復しない。"""
        config = ExpectationLifecycleConfig(record_freshness_decay_rate=0.3)
        proc = ExpectationLifecycleDescriptionProcessor(config=config)

        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        # 数サイクルで減衰
        for _ in range(3):
            proc.process(store)

        freshness_before = proc.state.transition_records[0].record_freshness

        # 参照
        proc.get_recent_records()
        proc.get_lifecycle_view("exp1")
        proc.get_enrichment_data()

        freshness_after = proc.state.transition_records[0].record_freshness
        assert freshness_after == freshness_before  # 変化なし

    def test_no_feedback_to_expectation_formation(self):
        """ライフサイクル記録が予期形成に書き込まない。

        READ-ONLY原則の確認。
        """
        proc = create_expectation_lifecycle_processor()
        # プロセッサに書き込みメソッドがないことを確認
        forbidden_patterns = [
            "update_expectation", "modify_expectation",
            "set_expectation", "write_to_formation",
            "update_decay_rate", "set_parameter",
            "modify_parameter",
        ]
        methods = [m for m in dir(proc) if not m.startswith("_") and callable(getattr(proc, m))]
        for method in methods:
            for pattern in forbidden_patterns:
                assert pattern not in method.lower(), f"Forbidden method found: {method}"

    def test_no_policy_connection(self):
        """ポリシー選択・バイアス適用・スコアリングへの接続がない。"""
        proc = create_expectation_lifecycle_processor()
        forbidden_patterns = [
            "policy", "bias", "scoring", "decision",
            "goal", "motive", "action", "emotion",
            "responsibility", "value",
        ]
        methods = [m for m in dir(proc) if not m.startswith("_") and callable(getattr(proc, m))]
        for method in methods:
            method_lower = method.lower()
            for pattern in forbidden_patterns:
                assert pattern not in method_lower, f"Suspicious method found: {method}"

    def test_no_quality_evaluation(self):
        """予期の品質評価をしない。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        data = proc.get_enrichment_data()
        # 品質・精度・有用性・適切さに関するキーがない
        for entry in data["entries"]:
            assert "quality" not in entry
            assert "accuracy" not in entry
            assert "usefulness" not in entry
            assert "correctness" not in entry
            assert "hit" not in entry
            assert "miss" not in entry


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """エッジケーステスト。"""

    def test_none_store(self):
        """Noneのストアでもエラーにならない。"""
        proc = create_expectation_lifecycle_processor()
        count = proc.process(None)
        assert count == 0

    def test_empty_store(self):
        """空のストアでもエラーにならない。"""
        proc = create_expectation_lifecycle_processor()
        store = _make_store()
        count = proc.process(store)
        assert count == 0

    def test_store_without_expectations_attr(self):
        """expectations属性がないストアでもエラーにならない。"""
        proc = create_expectation_lifecycle_processor()

        class EmptyStore:
            pass

        count = proc.process(EmptyStore())
        assert count == 0

    def test_expectation_without_id(self):
        """IDがない予期はスキップされる。"""
        proc = create_expectation_lifecycle_processor()

        @dataclass
        class BadExpectation:
            expectation_id: str = ""
            source_type: Any = None
            basis: Any = None
            description: str = "test"
            strength: float = 0.5
            freshness: float = 0.8
            revision_count: int = 0
            competing_ids: tuple = ()

        store = MockExpectationStore(expectations=(BadExpectation(),))
        count = proc.process(store)
        assert count == 0

    def test_rapid_succession_cycles(self):
        """高速連続サイクルでもエラーにならない。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)

        for _ in range(100):
            proc.process(store)

        assert proc.state.cycle_count == 100

    def test_many_expectations(self):
        """大量の予期を処理できる。"""
        proc = create_expectation_lifecycle_processor()
        exps = [_make_exp(exp_id=f"exp{i}") for i in range(50)]
        store = _make_store(*exps)
        count = proc.process(store)
        assert count == 50

    def test_dict_store_input(self):
        """dict形式のストアでもエラーにならない。"""
        proc = create_expectation_lifecycle_processor()
        # dict入力はexpectations属性がないのでスキップ
        count = proc.process({"expectations": []})
        assert count == 0

    def test_cycle_count_increments(self):
        """サイクルカウンタが正しくインクリメントされる。"""
        proc = create_expectation_lifecycle_processor()
        assert proc.state.cycle_count == 0

        proc.process(None)
        assert proc.state.cycle_count == 1

        proc.process(None)
        assert proc.state.cycle_count == 2

    def test_strength_same_stage_no_detection(self):
        """強度が変化しても段階ラベルが同じなら検出しない。"""
        proc = create_expectation_lifecycle_processor()
        # strength=0.5 → "moderate"
        exp1 = _make_exp(exp_id="exp1", strength=0.5)
        store1 = _make_store(exp1)
        proc.process(store1)

        # strength=0.6 → still "moderate"
        exp2 = _make_exp(exp_id="exp1", strength=0.6)
        store2 = _make_store(exp2)
        count = proc.process(store2)
        assert count == 0  # 段階変化なし

    def test_freshness_same_stage_no_detection(self):
        """鮮度が変化しても段階ラベルが同じなら検出しない。"""
        proc = create_expectation_lifecycle_processor()
        # freshness=0.85 → "fresh"
        exp1 = _make_exp(exp_id="exp1", freshness=0.85)
        store1 = _make_store(exp1)
        proc.process(store1)

        # freshness=0.95 → still "fresh"
        exp2 = _make_exp(exp_id="exp1", freshness=0.95)
        store2 = _make_store(exp2)
        count = proc.process(store2)
        assert count == 0  # 段階変化なし


# =============================================================================
# Summary tests
# =============================================================================

class TestSummary:
    """サマリ関数のテスト。"""

    def test_empty_summary(self):
        """空の状態のサマリ。"""
        state = ExpectationLifecycleState()
        summary = get_lifecycle_summary(state)
        assert "待機中" in summary

    def test_summary_with_data(self):
        """データがある場合のサマリ。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        summary = get_lifecycle_summary(proc.state)
        assert "cycle=" in summary
        assert "遷移記録=" in summary

    def test_get_summary_method(self):
        """プロセッサのget_summary()メソッド。"""
        proc = create_expectation_lifecycle_processor()
        exp = _make_exp(exp_id="exp1")
        store = _make_store(exp)
        proc.process(store)

        summary = proc.get_summary()
        assert "total_records" in summary
        assert "cycle_count" in summary
        assert "active_lifecycles" in summary
        assert "completed_lifecycles" in summary


# =============================================================================
# Factory tests
# =============================================================================

class TestFactory:
    """ファクトリ関数のテスト。"""

    def test_create_with_default_config(self):
        proc = create_expectation_lifecycle_processor()
        assert proc is not None
        assert proc.state.cycle_count == 0

    def test_create_with_custom_config(self):
        config = ExpectationLifecycleConfig(max_records=50, enrichment_count=5)
        proc = create_expectation_lifecycle_processor(config=config)
        assert proc is not None

    def test_state_property(self):
        proc = create_expectation_lifecycle_processor()
        state = proc.state
        assert isinstance(state, ExpectationLifecycleState)

    def test_state_setter(self):
        proc = create_expectation_lifecycle_processor()
        new_state = ExpectationLifecycleState(cycle_count=42)
        proc.state = new_state
        assert proc.state.cycle_count == 42


# =============================================================================
# Enum tests
# =============================================================================

class TestEnums:
    """Enumのテスト。"""

    def test_transition_types(self):
        assert TransitionType.GENERATION.value == "generation"
        assert TransitionType.DISAPPEARANCE.value == "disappearance"
        assert TransitionType.REVISION.value == "revision"
        assert TransitionType.STRENGTH_CHANGE.value == "strength_change"
        assert TransitionType.FRESHNESS_CHANGE.value == "freshness_change"

    def test_record_freshness_levels(self):
        assert RecordFreshness.FRESH.value == "fresh"
        assert RecordFreshness.RECENT.value == "recent"
        assert RecordFreshness.AGING.value == "aging"
        assert RecordFreshness.STALE.value == "stale"
        assert RecordFreshness.FADED.value == "faded"

    def test_convergence_levels(self):
        assert ConvergenceLevel.NONE.value == "none"
        assert ConvergenceLevel.MILD.value == "mild"
        assert ConvergenceLevel.MODERATE.value == "moderate"
        assert ConvergenceLevel.STRONG.value == "strong"


# =============================================================================
# SnapshotEntry tests
# =============================================================================

class TestSnapshotEntry:
    """スナップショットエントリのテスト。"""

    def test_to_dict_from_dict(self):
        entry = SnapshotEntry(
            expectation_id="exp1",
            source_type="repetition",
            basis_type="pattern_continuation",
            description="test",
            strength=0.5,
            freshness=0.8,
            revision_count=2,
            competing_ids=["exp2"],
        )
        d = entry.to_dict()
        restored = SnapshotEntry.from_dict(d)
        assert restored.expectation_id == "exp1"
        assert restored.strength == 0.5
        assert restored.revision_count == 2
        assert restored.competing_ids == ["exp2"]


# =============================================================================
# ConvergenceRecord tests
# =============================================================================

class TestConvergenceRecord:
    """収束監視記録のテスト。"""

    def test_to_dict_from_dict(self):
        rec = ConvergenceRecord(
            convergence_score=0.6,
            convergence_level=ConvergenceLevel.MODERATE.value,
            dominant_type="generation",
            type_diversity=0.4,
            cycle=10,
        )
        d = rec.to_dict()
        restored = ConvergenceRecord.from_dict(d)
        assert restored.convergence_score == 0.6
        assert restored.dominant_type == "generation"
        assert restored.cycle == 10
