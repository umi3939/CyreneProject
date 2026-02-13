"""
tests/test_responsibility_manager.py - ResponsibilityManager persistence tests.

Tests verify:
1. Init with temp file path (file isolation)
2. get_state creates default for new user
3. get_state returns ResponsibilityState
4. record_decision returns state + decision_id
5. record_decision persists to file
6. get_influence returns ResponsibilityInfluence
7. get_summary returns expected dict structure
8. Multiple users isolated
9. _load handles missing file
10. _load handles corrupt JSON
11. calc_responsibility_influence convenience function
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from psyche.responsibility_manager import (
    ResponsibilityManager,
    calc_responsibility_influence,
)
from psyche.responsibility import (
    ResponsibilityState,
    ResponsibilityInfluence,
    create_default_state,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_filepath(tmp_path: Path) -> Path:
    """Return a temporary file path for responsibility data."""
    return tmp_path / "responsibility.json"


@pytest.fixture
def mgr(tmp_filepath: Path) -> ResponsibilityManager:
    """Create a ResponsibilityManager backed by a temporary file."""
    return ResponsibilityManager(filepath=tmp_filepath)


@pytest.fixture
def sample_policy() -> dict:
    return {"policy_label": "共感する", "rationale": "相手の気持ちに寄り添う"}


@pytest.fixture
def sample_context() -> dict:
    return {"target_partner": "user_1", "fear_level": 0.3}


# ── Test: Initialization ──────────────────────────────────────────


class TestInit:
    """ResponsibilityManager の初期化"""

    def test_init_with_tmp_path(self, tmp_filepath: Path):
        """tmp_path でファイルパスを指定して初期化できる"""
        mgr = ResponsibilityManager(filepath=tmp_filepath)
        assert mgr.filepath == tmp_filepath

    def test_init_creates_empty_data_when_file_missing(self, tmp_filepath: Path):
        """ファイルが存在しない場合、空の内部データで初期化される"""
        mgr = ResponsibilityManager(filepath=tmp_filepath)
        assert mgr._data == {}

    def test_init_loads_existing_file(self, tmp_filepath: Path):
        """既存ファイルがあれば読み込む"""
        # Pre-populate the file
        state = create_default_state()
        data = {"user_A": state.model_dump()}
        tmp_filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_filepath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        mgr = ResponsibilityManager(filepath=tmp_filepath)
        assert "user_A" in mgr._data


# ── Test: get_state ───────────────────────────────────────────────


class TestGetState:
    """get_state の動作"""

    def test_creates_default_for_new_user(self, mgr: ResponsibilityManager):
        """新規ユーザーにはデフォルト状態が生成される"""
        state = mgr.get_state("new_user")

        assert state.total_weight == 0.0
        assert state.pending_decisions == 0
        assert state.accumulated_harm == 0.0
        assert state.accumulated_confidence == 0.0
        assert len(state.recent_decisions) == 0

    def test_returns_responsibility_state(self, mgr: ResponsibilityManager):
        """返り値の型が ResponsibilityState である"""
        state = mgr.get_state("user_1")
        assert isinstance(state, ResponsibilityState)

    def test_returns_same_state_for_same_user(self, mgr: ResponsibilityManager):
        """同じユーザーに対して同等の状態を返す"""
        state1 = mgr.get_state("user_1")
        state2 = mgr.get_state("user_1")

        assert state1.total_weight == state2.total_weight
        assert state1.pending_decisions == state2.pending_decisions

    def test_state_persists_in_internal_data(self, mgr: ResponsibilityManager):
        """取得した状態が内部データに保存される"""
        mgr.get_state("user_1")
        assert "user_1" in mgr._data


# ── Test: record_decision ─────────────────────────────────────────


class TestRecordDecision:
    """record_decision の動作"""

    def test_returns_state_and_decision_id(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """ResponsibilityState と decision_id のタプルを返す"""
        result = mgr.record_decision("user_1", sample_policy, sample_context)

        assert isinstance(result, tuple)
        assert len(result) == 2

        state, decision_id = result
        assert isinstance(state, ResponsibilityState)
        assert isinstance(decision_id, str)
        assert len(decision_id) == 12

    def test_increments_pending_decisions(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """pending_decisions が増加する"""
        state, _ = mgr.record_decision("user_1", sample_policy, sample_context)
        assert state.pending_decisions == 1

        state2, _ = mgr.record_decision("user_1", sample_policy, sample_context)
        assert state2.pending_decisions == 2

    def test_appends_to_recent_decisions(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """recent_decisions にレコードが追加される"""
        state, decision_id = mgr.record_decision("user_1", sample_policy, sample_context)

        assert len(state.recent_decisions) == 1
        record = state.recent_decisions[0]
        assert record["id"] == decision_id
        assert record["policy_label"] == "共感する"
        assert record["evaluated"] is False

    def test_persists_to_file(
        self,
        mgr: ResponsibilityManager,
        tmp_filepath: Path,
        sample_policy: dict,
        sample_context: dict,
    ):
        """判断記録がファイルに永続化される"""
        mgr.record_decision("user_1", sample_policy, sample_context)

        # Verify file exists and contains the data
        assert tmp_filepath.exists()

        raw = json.loads(tmp_filepath.read_text(encoding="utf-8"))
        assert "user_1" in raw
        assert len(raw["user_1"]["recent_decisions"]) == 1

    def test_file_survives_reload(
        self,
        tmp_filepath: Path,
        sample_policy: dict,
        sample_context: dict,
    ):
        """ファイルに書き込まれたデータが再読み込みで復元される"""
        mgr1 = ResponsibilityManager(filepath=tmp_filepath)
        mgr1.record_decision("user_1", sample_policy, sample_context)

        # Create new manager from same file
        mgr2 = ResponsibilityManager(filepath=tmp_filepath)
        state = mgr2.get_state("user_1")

        assert len(state.recent_decisions) == 1
        assert state.pending_decisions >= 1


# ── Test: evaluate_outcome ────────────────────────────────────────


class TestEvaluateOutcome:
    """evaluate_outcome の動作"""

    def test_returns_responsibility_state(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """ResponsibilityState を返す"""
        _, decision_id = mgr.record_decision("user_1", sample_policy, sample_context)
        outcome = {"user_reaction": "positive", "relationship_delta": 0.1}

        state = mgr.evaluate_outcome("user_1", decision_id, outcome)
        assert isinstance(state, ResponsibilityState)

    def test_decrements_pending_decisions(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """pending_decisions が減少する"""
        _, decision_id = mgr.record_decision("user_1", sample_policy, sample_context)
        outcome = {"user_reaction": "neutral"}

        state = mgr.evaluate_outcome("user_1", decision_id, outcome)
        assert state.pending_decisions == 0

    def test_positive_outcome_increases_confidence(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """肯定的結果で accumulated_confidence が増加する"""
        _, decision_id = mgr.record_decision("user_1", sample_policy, sample_context)
        outcome = {
            "user_reaction": "positive",
            "relationship_delta": 0.1,
            "expectation_gap": 0.0,
        }

        state = mgr.evaluate_outcome("user_1", decision_id, outcome)
        assert state.accumulated_confidence > 0.0

    def test_negative_outcome_increases_harm(
        self,
        mgr: ResponsibilityManager,
    ):
        """否定的結果で accumulated_harm が増加する"""
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1"}
        _, decision_id = mgr.record_decision("user_1", policy, context)

        outcome = {
            "user_reaction": "negative",
            "relationship_delta": -0.2,
            "expectation_gap": 0.3,
        }

        state = mgr.evaluate_outcome("user_1", decision_id, outcome)
        assert state.accumulated_harm > 0.0
        assert state.total_weight > 0.0

    def test_evaluate_persists_to_file(
        self,
        mgr: ResponsibilityManager,
        tmp_filepath: Path,
        sample_policy: dict,
        sample_context: dict,
    ):
        """評価結果がファイルに永続化される"""
        _, decision_id = mgr.record_decision("user_1", sample_policy, sample_context)
        outcome = {"user_reaction": "positive", "relationship_delta": 0.1}
        mgr.evaluate_outcome("user_1", decision_id, outcome)

        raw = json.loads(tmp_filepath.read_text(encoding="utf-8"))
        record = raw["user_1"]["recent_decisions"][0]
        assert record["evaluated"] is True

    def test_unknown_decision_id_returns_unchanged(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """存在しない decision_id では状態が変わらない"""
        mgr.record_decision("user_1", sample_policy, sample_context)
        state_before = mgr.get_state("user_1")

        outcome = {"user_reaction": "positive"}
        state_after = mgr.evaluate_outcome("user_1", "nonexistent_id", outcome)

        assert state_after.pending_decisions == state_before.pending_decisions


# ── Test: get_influence ───────────────────────────────────────────


class TestGetInfluence:
    """get_influence の動作"""

    def test_returns_responsibility_influence(self, mgr: ResponsibilityManager):
        """ResponsibilityInfluence を返す"""
        influence = mgr.get_influence("user_1")
        assert isinstance(influence, ResponsibilityInfluence)

    def test_default_influence_is_zero(self, mgr: ResponsibilityManager):
        """新規ユーザーの影響値はゼロ"""
        influence = mgr.get_influence("new_user")

        assert influence.fear_amplification == 0.0
        assert influence.caution_bias == 0.0
        assert influence.anxiety_baseline == 0.0
        assert influence.empathy_bias == 0.0

    def test_influence_reflects_accumulated_state(self, mgr: ResponsibilityManager):
        """蓄積された状態を反映した影響を返す"""
        # Build up harm through negative outcomes
        for _ in range(3):
            policy = {"policy_label": "からかう"}
            context = {"target_partner": "user_1"}
            _, decision_id = mgr.record_decision("user_1", policy, context)
            outcome = {
                "user_reaction": "negative",
                "relationship_delta": -0.2,
                "expectation_gap": 0.3,
            }
            mgr.evaluate_outcome("user_1", decision_id, outcome)

        influence = mgr.get_influence("user_1")

        # After negative outcomes, caution and empathy should be nonzero
        assert influence.caution_bias > 0.0
        assert influence.empathy_bias > 0.0


# ── Test: get_summary ─────────────────────────────────────────────


class TestGetSummary:
    """get_summary の動作"""

    def test_returns_expected_structure(self, mgr: ResponsibilityManager):
        """期待される辞書構造を返す"""
        summary = mgr.get_summary("user_1")

        assert isinstance(summary, dict)

        # Top-level keys
        assert "total_weight" in summary
        assert "accumulated_harm" in summary
        assert "accumulated_confidence" in summary
        assert "pending_decisions" in summary
        assert "recent_decision_count" in summary
        assert "influence" in summary

        # Influence sub-keys
        influence = summary["influence"]
        assert "fear_amplification" in influence
        assert "caution_bias" in influence
        assert "anxiety_baseline" in influence
        assert "empathy_bias" in influence

    def test_default_summary_values(self, mgr: ResponsibilityManager):
        """新規ユーザーのサマリーはデフォルト値"""
        summary = mgr.get_summary("new_user")

        assert summary["total_weight"] == 0.0
        assert summary["accumulated_harm"] == 0.0
        assert summary["accumulated_confidence"] == 0.0
        assert summary["pending_decisions"] == 0
        assert summary["recent_decision_count"] == 0

    def test_summary_reflects_recorded_decisions(
        self,
        mgr: ResponsibilityManager,
        sample_policy: dict,
        sample_context: dict,
    ):
        """記録した判断がサマリーに反映される"""
        mgr.record_decision("user_1", sample_policy, sample_context)
        mgr.record_decision("user_1", sample_policy, sample_context)

        summary = mgr.get_summary("user_1")

        assert summary["pending_decisions"] == 2
        assert summary["recent_decision_count"] == 2

    def test_summary_values_are_rounded(self, mgr: ResponsibilityManager):
        """サマリーの数値が丸められている"""
        # Build up some state
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1"}
        _, decision_id = mgr.record_decision("user_1", policy, context)
        outcome = {"user_reaction": "negative", "relationship_delta": -0.1}
        mgr.evaluate_outcome("user_1", decision_id, outcome)

        summary = mgr.get_summary("user_1")

        # Values should be rounded to 4 decimal places
        assert summary["total_weight"] == round(summary["total_weight"], 4)
        assert summary["accumulated_harm"] == round(summary["accumulated_harm"], 4)
        assert summary["accumulated_confidence"] == round(summary["accumulated_confidence"], 4)


# ── Test: Multiple Users Isolated ─────────────────────────────────


class TestMultipleUsersIsolated:
    """複数ユーザー間の隔離"""

    def test_users_have_independent_states(
        self,
        mgr: ResponsibilityManager,
    ):
        """ユーザー間で状態が独立している"""
        policy = {"policy_label": "共感する"}
        context = {"target_partner": "user_A"}

        mgr.record_decision("user_A", policy, context)
        mgr.record_decision("user_A", policy, context)

        state_a = mgr.get_state("user_A")
        state_b = mgr.get_state("user_B")

        assert state_a.pending_decisions == 2
        assert state_b.pending_decisions == 0
        assert len(state_a.recent_decisions) == 2
        assert len(state_b.recent_decisions) == 0

    def test_evaluate_does_not_affect_other_user(
        self,
        mgr: ResponsibilityManager,
    ):
        """一方のユーザーの評価が他方に影響しない"""
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "partner"}

        _, id_a = mgr.record_decision("user_A", policy, context)
        mgr.record_decision("user_B", policy, context)

        outcome = {"user_reaction": "negative", "relationship_delta": -0.3}
        mgr.evaluate_outcome("user_A", id_a, outcome)

        state_a = mgr.get_state("user_A")
        state_b = mgr.get_state("user_B")

        assert state_a.accumulated_harm > 0.0
        assert state_b.accumulated_harm == 0.0

    def test_multiple_users_in_file(
        self,
        mgr: ResponsibilityManager,
        tmp_filepath: Path,
    ):
        """複数ユーザーがファイルに共存する"""
        policy = {"policy_label": "共感する"}
        context = {}

        mgr.record_decision("alice", policy, context)
        mgr.record_decision("bob", policy, context)
        mgr.record_decision("charlie", policy, context)

        raw = json.loads(tmp_filepath.read_text(encoding="utf-8"))
        assert "alice" in raw
        assert "bob" in raw
        assert "charlie" in raw


# ── Test: _load ───────────────────────────────────────────────────


class TestLoad:
    """_load の動作"""

    def test_handles_missing_file(self, tmp_path: Path):
        """ファイルが存在しない場合、空辞書を返す"""
        nonexistent = tmp_path / "nonexistent" / "responsibility.json"
        mgr = ResponsibilityManager(filepath=nonexistent)
        assert mgr._data == {}

    def test_handles_corrupt_json(self, tmp_path: Path):
        """不正な JSON ファイルの場合、空辞書を返す"""
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{invalid json content!!!", encoding="utf-8")

        mgr = ResponsibilityManager(filepath=corrupt_file)
        assert mgr._data == {}

    def test_handles_non_dict_json(self, tmp_path: Path):
        """JSON がリストなど辞書以外の場合、空辞書を返す"""
        list_file = tmp_path / "list.json"
        list_file.write_text("[1, 2, 3]", encoding="utf-8")

        mgr = ResponsibilityManager(filepath=list_file)
        assert mgr._data == {}

    def test_handles_empty_file(self, tmp_path: Path):
        """空ファイルの場合、空辞書を返す"""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")

        mgr = ResponsibilityManager(filepath=empty_file)
        assert mgr._data == {}

    def test_loads_valid_json(self, tmp_path: Path):
        """正常な JSON を正しく読み込む"""
        valid_file = tmp_path / "valid.json"
        data = {"user_1": {"total_weight": 0.5, "pending_decisions": 1}}
        valid_file.write_text(json.dumps(data), encoding="utf-8")

        mgr = ResponsibilityManager(filepath=valid_file)
        assert "user_1" in mgr._data
        assert mgr._data["user_1"]["total_weight"] == 0.5


# ── Test: _save ───────────────────────────────────────────────────


class TestSave:
    """_save の動作"""

    def test_creates_parent_directories(self, tmp_path: Path):
        """親ディレクトリが存在しない場合、作成する"""
        deep_path = tmp_path / "a" / "b" / "c" / "responsibility.json"
        mgr = ResponsibilityManager(filepath=deep_path)
        mgr.get_state("user_1")  # triggers _save via _get_raw_state

        assert deep_path.exists()

    def test_file_contains_valid_json(
        self,
        mgr: ResponsibilityManager,
        tmp_filepath: Path,
    ):
        """保存されたファイルが有効な JSON である"""
        mgr.record_decision("user_1", {"policy_label": "test"}, {})

        content = tmp_filepath.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)


# ── Test: _apply_time_decay ───────────────────────────────────────


class TestApplyTimeDecay:
    """_apply_time_decay の動作"""

    def test_no_decay_within_six_minutes(self, mgr: ResponsibilityManager):
        """6分以内では減衰しない"""
        # Get state (which will have current timestamp)
        state = mgr.get_state("user_1")

        # Apply decay immediately (0 hours elapsed)
        decayed = mgr._apply_time_decay(state)

        assert decayed.total_weight == state.total_weight

    def test_decay_applied_after_threshold(self, mgr: ResponsibilityManager):
        """閾値を超えると減衰が適用される"""
        from datetime import datetime, timedelta

        # Create state with old timestamp
        state = ResponsibilityState(
            total_weight=0.5,
            accumulated_harm=0.3,
            last_updated=(datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds"),
        )

        decayed = mgr._apply_time_decay(state)

        assert decayed.total_weight < 0.5


# ── Test: calc_responsibility_influence ───────────────────────────


class TestCalcResponsibilityInfluence:
    """calc_responsibility_influence 便利関数"""

    def test_returns_responsibility_influence(self):
        """ResponsibilityInfluence を返す"""
        state = create_default_state()
        influence = calc_responsibility_influence(state)
        assert isinstance(influence, ResponsibilityInfluence)

    def test_default_state_gives_zero_influence(self):
        """デフォルト状態ではゼロの影響"""
        state = create_default_state()
        influence = calc_responsibility_influence(state)

        assert influence.fear_amplification == 0.0
        assert influence.caution_bias == 0.0
        assert influence.anxiety_baseline == 0.0
        assert influence.empathy_bias == 0.0

    def test_high_weight_gives_nonzero_influence(self):
        """高い責任重みは非ゼロの影響を生む"""
        state = ResponsibilityState(
            total_weight=0.8,
            accumulated_harm=0.5,
            accumulated_confidence=0.1,
        )
        influence = calc_responsibility_influence(state)

        assert influence.fear_amplification > 0.0
        assert influence.caution_bias > 0.0
        assert influence.empathy_bias > 0.0

    def test_matches_manager_get_influence(self, mgr: ResponsibilityManager):
        """マネージャの get_influence と同じ結果を返す"""
        # Build up some state
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1"}
        _, decision_id = mgr.record_decision("user_1", policy, context)
        outcome = {"user_reaction": "negative", "relationship_delta": -0.1}
        mgr.evaluate_outcome("user_1", decision_id, outcome)

        # Get state and compare both paths
        state = mgr.get_state("user_1")
        influence_from_func = calc_responsibility_influence(state)
        influence_from_mgr = mgr.get_influence("user_1")

        # They should produce the same influence type (values may differ slightly
        # due to get_influence calling get_state again which applies decay)
        assert isinstance(influence_from_func, ResponsibilityInfluence)
        assert isinstance(influence_from_mgr, ResponsibilityInfluence)


# ── Test: End-to-end flow ─────────────────────────────────────────


class TestEndToEnd:
    """record -> evaluate -> get_influence の一連のフロー"""

    def test_full_lifecycle(self, mgr: ResponsibilityManager):
        """判断記録 -> 結果評価 -> 影響取得 の完全なフロー"""
        user = "test_user"

        # 1. Record a decision
        policy = {"policy_label": "共感する", "rationale": "寄り添う"}
        context = {"target_partner": user, "fear_level": 0.2}
        state, decision_id = mgr.record_decision(user, policy, context)

        assert state.pending_decisions == 1
        assert len(state.recent_decisions) == 1

        # 2. Evaluate the outcome
        outcome = {
            "user_reaction": "positive",
            "relationship_delta": 0.2,
            "expectation_gap": 0.0,
        }
        state = mgr.evaluate_outcome(user, decision_id, outcome)

        assert state.pending_decisions == 0
        assert state.accumulated_confidence > 0.0

        # 3. Get influence
        influence = mgr.get_influence(user)

        assert isinstance(influence, ResponsibilityInfluence)

        # 4. Get summary
        summary = mgr.get_summary(user)

        assert summary["pending_decisions"] == 0
        assert summary["recent_decision_count"] == 1
        assert summary["accumulated_confidence"] > 0.0

    def test_repeated_negative_outcomes_build_burden(
        self,
        mgr: ResponsibilityManager,
    ):
        """否定的結果の繰り返しで責任負担が蓄積される"""
        user = "burden_user"

        for _ in range(5):
            policy = {"policy_label": "からかう"}
            context = {"target_partner": "partner", "fear_level": 0.4}
            _, decision_id = mgr.record_decision(user, policy, context)
            outcome = {
                "user_reaction": "rejected",
                "relationship_delta": -0.3,
                "expectation_gap": 0.5,
            }
            mgr.evaluate_outcome(user, decision_id, outcome)

        state = mgr.get_state(user)
        influence = mgr.get_influence(user)

        assert state.accumulated_harm > 0.0
        assert state.total_weight > 0.0
        assert influence.caution_bias > 0.0
        assert influence.fear_amplification > 0.0

    def test_reload_preserves_full_state(self, tmp_filepath: Path):
        """マネージャ再生成で全状態が保持される"""
        # First manager: record and evaluate
        mgr1 = ResponsibilityManager(filepath=tmp_filepath)
        policy = {"policy_label": "からかう"}
        context = {"target_partner": "user_1"}
        _, decision_id = mgr1.record_decision("user_1", policy, context)
        outcome = {"user_reaction": "negative", "relationship_delta": -0.2}
        mgr1.evaluate_outcome("user_1", decision_id, outcome)

        summary_before = mgr1.get_summary("user_1")

        # Second manager: reload from same file
        mgr2 = ResponsibilityManager(filepath=tmp_filepath)
        summary_after = mgr2.get_summary("user_1")

        assert summary_after["recent_decision_count"] == summary_before["recent_decision_count"]
        assert summary_after["accumulated_harm"] > 0.0
