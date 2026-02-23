"""
tests/test_persistence_integrity.py - 永続化整合性検査のテスト

tools/persistence_integrity.py の6種類の汎用検証パターンと
ユーティリティ関数、コマンドラインインターフェース、
および persistence.py のload時自動検証をテストする。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from tools.persistence_integrity import (
    ACCUMULATION_LIMIT_PATTERNS,
    REFERENCE_EXISTENCE_PATTERNS,
    REQUIRED_FIELD_PATTERNS,
    TIMESTAMP_ORDER_PATTERNS,
    TYPE_STRUCTURE_PATTERNS,
    VERSION_FIELD_PATTERNS,
    _MISSING,
    _check_accumulation_limits,
    _check_reference_existence,
    _check_required_fields,
    _check_timestamp_order,
    _check_type_structure,
    _check_version_fields,
    _extract_ids,
    _extract_target_ids,
    _is_empty_value,
    _is_numeric,
    _resolve_path,
    check_integrity,
    main,
)


# ── ヘルパー ─────────────────────────────────────────────────

def _make_minimal_save_dict(version: int = 42) -> dict[str, Any]:
    """テスト用の最小限の永続化辞書を生成する。"""
    data: dict[str, Any] = {
        "version": version,
        "tick_count": 5,
        "psyche": {
            "emotions": {"joy": 0.5, "anger": 0.0},
            "drives": {"social": 0.5, "curiosity": 0.5},
            "mood": {"valence": 0.1, "arousal": 0.3},
            "fear_index": 0.1,
            "loss_aversion": 0.3,
            "last_updated": "2026-01-15T10:00:00",
        },
        "loop_state": {
            "memory": {
                "entries": [],
                "max_entries": 10,
                "last_update_time": 100.0,
            },
            "last_loop_time": 100.0,
        },
        "dynamics": {
            "phase": "neutral",
            "phase_entered_at": 90.0,
            "intensity_history": [0.5, 0.3],
            "max_history_length": 10,
        },
        "amplitude": {},
        "value_orientation": {},
        "self_ref_state": {},
        "last_self_view": {},
        "tendency_awareness": {},
        "last_diff_summary": {},
        "last_strain": {},
        "last_self_image": {},
        "last_coherence": {},
        "last_narrative": {},
        "last_episodes": {
            "episodes": [
                {"episode_id": "ep-001", "summary": "test episode 1"},
                {"episode_id": "ep-002", "summary": "test episode 2"},
            ],
            "links": [
                {
                    "from_episode_id": "ep-001",
                    "to_episode_id": "ep-002",
                    "link_type": "temporal",
                    "strength": 0.5,
                },
            ],
            "total_episodes_recorded": 2,
        },
        "last_bindings": {},
        "last_trace": {},
        "last_consumption": {},
        "last_expectations": {},
        "last_motives": {},
        "last_other_model": {},
        "input_supply": {},
        # Version 5 fields
        "tendency_state": {},
        "vector_state": {},
        "candidate_state": {},
        "transient_goal_state": {},
        "stability_valve": {},
        # Version 6 fields
        "dispersion_state": {},
        "context_sensitivity_state": {},
        "last_coupling": {},
        # Version 7 fields
        "policy_expansion_state": {},
        # Version 8 fields
        "memory_integration_state": {},
        # Version 9 fields
        "real_feed_state": {},
        # Version 10 fields
        "text_dialogue_state": {},
        # Version 11 fields
        "spontaneous_state": {},
        # Version 12 fields
        "vo_validation_state": {},
        # Version 13 fields
        "forgetting_fixation_state": {},
        # Version 14 fields
        "action_result_state": {},
        # Version 15 fields
        "dialogue_learning_state": {},
        # Version 16 fields
        "meta_emotion_state": {},
        # Version 17 fields
        "self_action_perception_state": {},
        # Version 18 fields
        "expectation_action_diff_log": [],
        # Version 19 fields
        "intent_action_gap_state": {},
        # Version 20 fields
        "temporal_cognition_state": {},
        # Version 21 fields
        "multi_path_recall_state": {},
        # Version 22 fields
        "introspection_cross_section_state": {},
        "perceptual_context_state": {},
        # Version 23 fields
        "selection_attribution_state": {},
        # Version 24 fields
        "reference_frequency_state": {},
        # Version 25 fields
        "persistent_commitment_state": {},
        # Version 26 fields
        "stabilization_description_state": {},
        # Version 27 fields
        "behavioral_diversity_state": {},
        # Version 28 fields
        "spontaneous_recall_state": {},
        # Version 29 fields
        "internal_contradiction_state": {},
        # Version 30 fields
        "interaction_accumulation_state": {},
        # Version 31 fields
        "emotional_backdrop_state": {},
        # Version 32 fields
        "situational_self_presentation_state": {},
        # Version 33 fields
        "drive_variation_state": {},
        # Version 34 fields
        "expectation_lifecycle_state": {},
        # Version 35 fields
        "input_pathway_balance_state": {},
        # Version 36 fields
        "responsibility_temporal_trace_state": {},
        # Version 37 fields
        "emotion_cooccurrence_state": {},
        # Version 38 fields
        "other_boundary_accumulation_state": {},
        # Version 39 fields
        "forgetting_recall_balance_state": {},
        # Version 40 fields
        "attention_distribution_state": {},
        # Version 41 fields
        "goal_hierarchy_propagation_state": {},
        # Version 42 fields
        "hypothesis_observation_pairing_state": {},
    }
    return data


# ── _resolve_path テスト ──────────────────────────────────────

class TestResolvePath:
    """_resolve_path ユーティリティのテスト。"""

    def test_single_level(self) -> None:
        data = {"version": 42}
        assert _resolve_path(data, "version") == 42

    def test_nested_path(self) -> None:
        data = {"psyche": {"emotions": {"joy": 0.5}}}
        assert _resolve_path(data, "psyche.emotions.joy") == 0.5

    def test_missing_key_returns_sentinel(self) -> None:
        data = {"psyche": {"emotions": {}}}
        result = _resolve_path(data, "psyche.nonexistent")
        assert isinstance(result, type(_MISSING))

    def test_missing_intermediate_returns_sentinel(self) -> None:
        data = {"psyche": {}}
        result = _resolve_path(data, "psyche.emotions.joy")
        assert isinstance(result, type(_MISSING))

    def test_none_value_at_path(self) -> None:
        data = {"field": None}
        assert _resolve_path(data, "field") is None

    def test_empty_dict_value(self) -> None:
        data = {"field": {}}
        assert _resolve_path(data, "field") == {}

    def test_list_value(self) -> None:
        data = {"field": [1, 2, 3]}
        assert _resolve_path(data, "field") == [1, 2, 3]

    def test_deeply_nested(self) -> None:
        data = {"a": {"b": {"c": {"d": 99}}}}
        assert _resolve_path(data, "a.b.c.d") == 99

    def test_missing_root_key(self) -> None:
        data = {"existing": 1}
        result = _resolve_path(data, "nonexistent")
        assert isinstance(result, type(_MISSING))

    def test_non_dict_intermediate(self) -> None:
        data = {"field": "string_value"}
        result = _resolve_path(data, "field.subfield")
        assert isinstance(result, type(_MISSING))


# ── _extract_ids テスト ───────────────────────────────────────

class TestExtractIds:
    """_extract_ids ユーティリティのテスト。"""

    def test_list_field_extraction(self) -> None:
        data = [
            {"episode_id": "ep-001", "other": "x"},
            {"episode_id": "ep-002", "other": "y"},
        ]
        result = _extract_ids(data, "list_field:episode_id")
        assert result == ["ep-001", "ep-002"]

    def test_list_field_with_missing_field(self) -> None:
        data = [
            {"episode_id": "ep-001"},
            {"other": "no_id"},
            {"episode_id": "ep-003"},
        ]
        result = _extract_ids(data, "list_field:episode_id")
        assert result == ["ep-001", "ep-003"]

    def test_list_of_strings(self) -> None:
        data = ["id-1", "id-2", "id-3"]
        result = _extract_ids(data, "list_of_strings")
        assert result == ["id-1", "id-2", "id-3"]

    def test_list_of_strings_with_non_strings(self) -> None:
        data = ["id-1", 42, "id-3", None]
        result = _extract_ids(data, "list_of_strings")
        assert result == ["id-1", "id-3"]

    def test_keys_extraction(self) -> None:
        data = {"key_a": 1, "key_b": 2}
        result = _extract_ids(data, "keys")
        assert set(result) == {"key_a", "key_b"}

    def test_missing_sentinel_returns_empty(self) -> None:
        result = _extract_ids(_MISSING, "list_field:episode_id")
        assert result == []

    def test_none_returns_empty(self) -> None:
        result = _extract_ids(None, "list_field:episode_id")
        assert result == []

    def test_non_list_for_list_field(self) -> None:
        result = _extract_ids("not a list", "list_field:episode_id")
        assert result == []

    def test_unknown_method_returns_empty(self) -> None:
        result = _extract_ids([1, 2, 3], "unknown_method")
        assert result == []

    def test_non_dict_for_keys(self) -> None:
        result = _extract_ids([1, 2], "keys")
        assert result == []


# ── _extract_target_ids テスト ────────────────────────────────

class TestExtractTargetIds:
    """_extract_target_ids ユーティリティのテスト。"""

    def test_basic_extraction(self) -> None:
        data = [
            {"episode_id": "ep-001"},
            {"episode_id": "ep-002"},
        ]
        result = _extract_target_ids(data, "episode_id")
        assert result == {"ep-001", "ep-002"}

    def test_missing_id_field(self) -> None:
        data = [
            {"episode_id": "ep-001"},
            {"other_field": "val"},
        ]
        result = _extract_target_ids(data, "episode_id")
        assert result == {"ep-001"}

    def test_missing_sentinel(self) -> None:
        result = _extract_target_ids(_MISSING, "episode_id")
        assert result == set()

    def test_none_input(self) -> None:
        result = _extract_target_ids(None, "episode_id")
        assert result == set()

    def test_non_list_input(self) -> None:
        result = _extract_target_ids({"ep": 1}, "episode_id")
        assert result == set()

    def test_non_string_id_values(self) -> None:
        data = [
            {"episode_id": "ep-001"},
            {"episode_id": 42},
        ]
        result = _extract_target_ids(data, "episode_id")
        assert result == {"ep-001"}


# ── _is_empty_value テスト ────────────────────────────────────

class TestIsEmptyValue:
    """_is_empty_value ユーティリティのテスト。"""

    def test_none_is_empty(self) -> None:
        assert _is_empty_value(None) is True

    def test_missing_is_empty(self) -> None:
        assert _is_empty_value(_MISSING) is True

    def test_empty_dict_is_empty(self) -> None:
        assert _is_empty_value({}) is True

    def test_empty_list_is_empty(self) -> None:
        assert _is_empty_value([]) is True

    def test_non_empty_dict_is_not_empty(self) -> None:
        assert _is_empty_value({"key": "val"}) is False

    def test_non_empty_list_is_not_empty(self) -> None:
        assert _is_empty_value([1]) is False

    def test_zero_is_not_empty(self) -> None:
        assert _is_empty_value(0) is False

    def test_string_is_not_empty(self) -> None:
        assert _is_empty_value("text") is False

    def test_false_is_not_empty(self) -> None:
        assert _is_empty_value(False) is False


# ── _is_numeric テスト ────────────────────────────────────────

class TestIsNumeric:
    """_is_numeric ユーティリティのテスト。"""

    def test_int(self) -> None:
        assert _is_numeric(42) is True

    def test_float(self) -> None:
        assert _is_numeric(3.14) is True

    def test_string(self) -> None:
        assert _is_numeric("42") is False

    def test_none(self) -> None:
        assert _is_numeric(None) is False

    def test_bool_is_numeric(self) -> None:
        # Pythonではboolはintのサブクラスだが、仕様上True
        assert _is_numeric(True) is True


# ── パターン1: 参照先存在確認テスト ──────────────────────────

class TestReferenceExistence:
    """パターン1: 参照先存在確認のテスト。"""

    def test_valid_references_no_findings(self) -> None:
        data = _make_minimal_save_dict()
        findings = _check_reference_existence(data)
        assert len(findings) == 0

    def test_broken_from_reference(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["links"] = [
            {
                "from_episode_id": "ep-BROKEN",
                "to_episode_id": "ep-002",
                "link_type": "temporal",
                "strength": 0.5,
            },
        ]
        findings = _check_reference_existence(data)
        assert len(findings) >= 1
        assert any("ep-BROKEN" in f["fact"] for f in findings)

    def test_broken_to_reference(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["links"] = [
            {
                "from_episode_id": "ep-001",
                "to_episode_id": "ep-MISSING",
                "link_type": "temporal",
                "strength": 0.5,
            },
        ]
        findings = _check_reference_existence(data)
        assert len(findings) >= 1
        assert any("ep-MISSING" in f["fact"] for f in findings)

    def test_no_links_no_findings(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["links"] = []
        findings = _check_reference_existence(data)
        assert len(findings) == 0

    def test_missing_episodes_field_skips(self) -> None:
        data = _make_minimal_save_dict()
        del data["last_episodes"]
        findings = _check_reference_existence(data)
        assert len(findings) == 0

    def test_multiple_broken_references(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["links"] = [
            {
                "from_episode_id": "ep-BAD1",
                "to_episode_id": "ep-BAD2",
                "link_type": "temporal",
                "strength": 0.5,
            },
            {
                "from_episode_id": "ep-BAD3",
                "to_episode_id": "ep-001",
                "link_type": "causal",
                "strength": 0.3,
            },
        ]
        findings = _check_reference_existence(data)
        # ep-BAD1, ep-BAD2, ep-BAD3 の3件が検出されるはず
        assert len(findings) == 3

    def test_all_findings_have_pattern_field(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["links"] = [
            {
                "from_episode_id": "ep-BROKEN",
                "to_episode_id": "ep-002",
                "link_type": "temporal",
                "strength": 0.5,
            },
        ]
        findings = _check_reference_existence(data)
        for f in findings:
            assert f["pattern"] == "reference_existence"
            assert "field_path" in f
            assert "fact" in f


# ── パターン2: 蓄積構造上限確認テスト ────────────────────────

class TestAccumulationLimits:
    """パターン2: 蓄積構造の構造的制約確認のテスト。"""

    def test_within_limit_no_findings(self) -> None:
        data = _make_minimal_save_dict()
        findings = _check_accumulation_limits(data)
        assert len(findings) == 0

    def test_episodes_exceed_limit(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["episodes"] = [
            {"episode_id": f"ep-{i:04d}", "summary": f"episode {i}"}
            for i in range(250)
        ]
        findings = _check_accumulation_limits(data)
        episode_findings = [
            f for f in findings if "last_episodes.episodes" in f["field_path"]
        ]
        assert len(episode_findings) == 1
        assert "250" in episode_findings[0]["fact"]
        assert "200" in episode_findings[0]["fact"]

    def test_intensity_history_exceed_limit(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["intensity_history"] = [0.5] * 15
        findings = _check_accumulation_limits(data)
        history_findings = [
            f for f in findings if "intensity_history" in f["field_path"]
        ]
        assert len(history_findings) == 1

    def test_memory_entries_exceed_limit(self) -> None:
        data = _make_minimal_save_dict()
        data["loop_state"]["memory"]["entries"] = [
            {"source_text": f"text {i}", "timestamp": float(i)}
            for i in range(15)
        ]
        findings = _check_accumulation_limits(data)
        entry_findings = [
            f for f in findings if "loop_state.memory.entries" in f["field_path"]
        ]
        assert len(entry_findings) == 1

    def test_exact_limit_no_finding(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["episodes"] = [
            {"episode_id": f"ep-{i:04d}", "summary": f"episode {i}"}
            for i in range(200)
        ]
        findings = _check_accumulation_limits(data)
        episode_findings = [
            f for f in findings if "last_episodes.episodes" in f["field_path"]
        ]
        assert len(episode_findings) == 0

    def test_missing_field_skips(self) -> None:
        data = _make_minimal_save_dict()
        del data["dynamics"]
        findings = _check_accumulation_limits(data)
        history_findings = [
            f for f in findings if "intensity_history" in f["field_path"]
        ]
        assert len(history_findings) == 0

    def test_action_result_pairs_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["action_result_state"] = {
            "pairs": [{"id": f"p-{i}"} for i in range(250)],
        }
        findings = _check_accumulation_limits(data)
        pair_findings = [
            f for f in findings if "action_result_state.pairs" in f["field_path"]
        ]
        assert len(pair_findings) == 1

    def test_all_findings_have_pattern_field(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["intensity_history"] = [0.5] * 15
        findings = _check_accumulation_limits(data)
        for f in findings:
            assert f["pattern"] == "accumulation_limit"


# ── パターン3: 時刻順序矛盾確認テスト ────────────────────────

class TestTimestampOrder:
    """パターン3: 時刻順序の矛盾確認のテスト。"""

    def test_normal_order_no_findings(self) -> None:
        data = _make_minimal_save_dict()
        # phase_entered_at (90.0) < last_loop_time (100.0) : 正常
        findings = _check_timestamp_order(data)
        assert len(findings) == 0

    def test_reversed_order_detected(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["phase_entered_at"] = 200.0
        data["loop_state"]["last_loop_time"] = 100.0
        findings = _check_timestamp_order(data)
        assert len(findings) == 1
        assert findings[0]["pattern"] == "timestamp_order"
        assert "200" in findings[0]["fact"]

    def test_equal_timestamps_no_finding(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["phase_entered_at"] = 100.0
        data["loop_state"]["last_loop_time"] = 100.0
        findings = _check_timestamp_order(data)
        assert len(findings) == 0

    def test_missing_earlier_field_skips(self) -> None:
        data = _make_minimal_save_dict()
        del data["dynamics"]
        findings = _check_timestamp_order(data)
        assert len(findings) == 0

    def test_missing_later_field_skips(self) -> None:
        data = _make_minimal_save_dict()
        del data["loop_state"]
        findings = _check_timestamp_order(data)
        assert len(findings) == 0

    def test_string_timestamps_normal_order(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["phase_entered_at"] = "2026-01-01T00:00:00"
        data["loop_state"]["last_loop_time"] = "2026-01-02T00:00:00"
        findings = _check_timestamp_order(data)
        assert len(findings) == 0

    def test_string_timestamps_reversed(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["phase_entered_at"] = "2026-01-03T00:00:00"
        data["loop_state"]["last_loop_time"] = "2026-01-01T00:00:00"
        findings = _check_timestamp_order(data)
        assert len(findings) == 1

    def test_mixed_types_skip(self) -> None:
        """数値と文字列が混在する場合はスキップ。"""
        data = _make_minimal_save_dict()
        data["dynamics"]["phase_entered_at"] = 100.0
        data["loop_state"]["last_loop_time"] = "2026-01-01T00:00:00"
        findings = _check_timestamp_order(data)
        # 型不一致なので非数値パスに進むが文字列でもないのでスキップ
        assert len(findings) == 0


# ── パターン4: 必須フィールド存在確認テスト ──────────────────

class TestRequiredFields:
    """パターン4: 必須フィールド存在確認のテスト。"""

    def test_all_present_no_findings(self) -> None:
        data = _make_minimal_save_dict()
        findings = _check_required_fields(data)
        assert len(findings) == 0

    def test_missing_version(self) -> None:
        data = _make_minimal_save_dict()
        del data["version"]
        findings = _check_required_fields(data)
        version_findings = [
            f for f in findings if f["field_path"] == "version"
        ]
        assert len(version_findings) == 1

    def test_missing_tick_count(self) -> None:
        data = _make_minimal_save_dict()
        del data["tick_count"]
        findings = _check_required_fields(data)
        tick_findings = [
            f for f in findings if f["field_path"] == "tick_count"
        ]
        assert len(tick_findings) == 1

    def test_empty_psyche(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"] = {}
        findings = _check_required_fields(data)
        psyche_findings = [
            f for f in findings if f["field_path"] == "psyche"
        ]
        assert len(psyche_findings) == 1

    def test_none_psyche(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"] = None
        findings = _check_required_fields(data)
        psyche_findings = [
            f for f in findings if f["field_path"] == "psyche"
        ]
        assert len(psyche_findings) == 1

    def test_missing_emotions(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"]["emotions"] = {}
        findings = _check_required_fields(data)
        emotion_findings = [
            f for f in findings if f["field_path"] == "psyche.emotions"
        ]
        assert len(emotion_findings) == 1

    def test_missing_drives(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"]["drives"] = {}
        findings = _check_required_fields(data)
        drive_findings = [
            f for f in findings if f["field_path"] == "psyche.drives"
        ]
        assert len(drive_findings) == 1

    def test_missing_mood(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"]["mood"] = {}
        findings = _check_required_fields(data)
        mood_findings = [
            f for f in findings if f["field_path"] == "psyche.mood"
        ]
        assert len(mood_findings) == 1

    def test_all_findings_have_correct_pattern(self) -> None:
        data = _make_minimal_save_dict()
        del data["version"]
        findings = _check_required_fields(data)
        for f in findings:
            assert f["pattern"] == "required_field"

    def test_zero_version_is_not_empty(self) -> None:
        """version=0でも空ではない。"""
        data = _make_minimal_save_dict()
        data["version"] = 0
        findings = _check_required_fields(data)
        version_findings = [
            f for f in findings if f["field_path"] == "version"
        ]
        assert len(version_findings) == 0


# ── パターン5: バージョン整合確認テスト ──────────────────────

class TestVersionFields:
    """パターン5: バージョン整合確認のテスト。"""

    def test_full_version_42_no_findings(self) -> None:
        data = _make_minimal_save_dict(version=42)
        findings = _check_version_fields(data)
        assert len(findings) == 0

    def test_version_1_minimal(self) -> None:
        data = {
            "version": 1,
            "tick_count": 0,
            "psyche": {},
            "loop_state": {},
            "dynamics": {},
        }
        findings = _check_version_fields(data)
        # Version 1 only requires basic fields
        assert len(findings) == 0

    def test_version_5_missing_fields(self) -> None:
        data = {
            "version": 5,
            "tick_count": 0,
            "psyche": {},
            "loop_state": {},
            "dynamics": {},
        }
        findings = _check_version_fields(data)
        # Should detect missing v5 fields
        v5_fields = VERSION_FIELD_PATTERNS[5]
        missing_count = sum(
            1 for f in v5_fields if f not in data
        )
        version_findings = [
            f for f in findings if f["pattern"] == "version_consistency"
        ]
        assert len(version_findings) == missing_count

    def test_version_42_missing_one_field(self) -> None:
        data = _make_minimal_save_dict(version=42)
        del data["hypothesis_observation_pairing_state"]
        findings = _check_version_fields(data)
        assert len(findings) >= 1
        assert any("hypothesis_observation_pairing_state" in f["fact"] for f in findings)

    def test_no_version_field(self) -> None:
        data = {"tick_count": 0, "psyche": {}}
        findings = _check_version_fields(data)
        assert len(findings) == 1
        assert "バージョン番号" in findings[0]["fact"]

    def test_non_numeric_version(self) -> None:
        data = {"version": "not_a_number"}
        findings = _check_version_fields(data)
        assert len(findings) == 1
        assert "数値でない" in findings[0]["fact"]

    def test_version_none(self) -> None:
        data = {"version": None}
        findings = _check_version_fields(data)
        assert len(findings) >= 1

    def test_lower_version_does_not_require_higher_fields(self) -> None:
        """バージョン10の辞書にはバージョン11以降のフィールドは要求しない。"""
        data = _make_minimal_save_dict(version=10)
        # v11以降のフィールドを削除
        for ver in range(11, 43):
            if ver in VERSION_FIELD_PATTERNS:
                for field in VERSION_FIELD_PATTERNS[ver]:
                    data.pop(field, None)
        findings = _check_version_fields(data)
        assert len(findings) == 0

    def test_findings_include_added_version(self) -> None:
        """検出結果にフィールド追加バージョンが含まれる。"""
        data = _make_minimal_save_dict(version=42)
        del data["spontaneous_state"]
        findings = _check_version_fields(data)
        spon_findings = [
            f for f in findings if "spontaneous_state" in f["fact"]
        ]
        assert len(spon_findings) == 1
        assert "バージョン11" in spon_findings[0]["fact"]


# ── check_integrity 統合テスト ────────────────────────────────

class TestCheckIntegrity:
    """check_integrity メイン関数の統合テスト。"""

    def test_clean_dict_returns_zero_findings(self) -> None:
        data = _make_minimal_save_dict()
        result = check_integrity(data)
        assert result["total_findings"] == 0
        assert len(result["findings"]) == 0

    def test_result_structure(self) -> None:
        data = _make_minimal_save_dict()
        result = check_integrity(data)
        assert "basic_info" in result
        assert "findings" in result
        assert "summary" in result
        assert "total_findings" in result

    def test_basic_info_contents(self) -> None:
        data = _make_minimal_save_dict()
        result = check_integrity(data)
        info = result["basic_info"]
        assert info["version"] == 42
        assert info["top_level_field_count"] > 0
        assert info["pattern_count"] > 0

    def test_summary_contains_all_patterns(self) -> None:
        data = _make_minimal_save_dict()
        result = check_integrity(data)
        summary = result["summary"]
        assert "reference_existence" in summary
        assert "accumulation_limit" in summary
        assert "timestamp_order" in summary
        assert "required_field" in summary
        assert "version_consistency" in summary
        assert "type_structure" in summary

    def test_original_dict_not_modified(self) -> None:
        data = _make_minimal_save_dict()
        original = copy.deepcopy(data)
        check_integrity(data)
        assert data == original

    def test_multiple_findings_from_different_patterns(self) -> None:
        data = _make_minimal_save_dict()
        # パターン1: 壊れた参照
        data["last_episodes"]["links"] = [
            {
                "from_episode_id": "ep-BROKEN",
                "to_episode_id": "ep-002",
                "link_type": "temporal",
                "strength": 0.5,
            },
        ]
        # パターン2: 上限超過
        data["dynamics"]["intensity_history"] = [0.5] * 15
        # パターン3: 時刻逆転
        data["dynamics"]["phase_entered_at"] = 999.0
        data["loop_state"]["last_loop_time"] = 100.0

        result = check_integrity(data)
        assert result["total_findings"] >= 3
        assert result["summary"]["reference_existence"] >= 1
        assert result["summary"]["accumulation_limit"] >= 1
        assert result["summary"]["timestamp_order"] >= 1

    def test_findings_are_list_of_dicts(self) -> None:
        data = _make_minimal_save_dict()
        del data["version"]
        result = check_integrity(data)
        assert isinstance(result["findings"], list)
        for f in result["findings"]:
            assert isinstance(f, dict)
            assert "pattern" in f
            assert "fact" in f

    def test_empty_dict_input(self) -> None:
        result = check_integrity({})
        # 必須フィールドが欠けているので検出件数 > 0
        assert result["total_findings"] > 0
        # versionもないのでversion_consistencyの検出もある
        assert result["summary"]["required_field"] > 0

    def test_total_findings_matches_findings_list(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["intensity_history"] = [0.5] * 15
        result = check_integrity(data)
        assert result["total_findings"] == len(result["findings"])

    def test_total_findings_matches_summary_sum(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"]["intensity_history"] = [0.5] * 15
        result = check_integrity(data)
        summary_total = sum(result["summary"].values())
        assert result["total_findings"] == summary_total

    def test_deeply_modified_dict_not_affecting_original(self) -> None:
        """deepcopyにより深いネスト内の変更も元辞書に影響しないことを確認。"""
        data = _make_minimal_save_dict()
        episodes = data["last_episodes"]["episodes"]
        original_ep_count = len(episodes)
        check_integrity(data)
        assert len(data["last_episodes"]["episodes"]) == original_ep_count


# ── コマンドラインテスト ──────────────────────────────────────

class TestCommandLine:
    """コマンドラインインターフェースのテスト。"""

    def test_valid_file(self, tmp_path: Path) -> None:
        data = _make_minimal_save_dict()
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps(data), encoding="utf-8")

        exit_code = main([str(snapshot)])
        assert exit_code == 0

    def test_missing_file(self) -> None:
        exit_code = main(["/nonexistent/path/to/file.json"])
        assert exit_code == 1

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")

        exit_code = main([str(bad_file)])
        assert exit_code == 1

    def test_non_dict_json(self, tmp_path: Path) -> None:
        list_file = tmp_path / "list.json"
        list_file.write_text("[1, 2, 3]", encoding="utf-8")

        exit_code = main([str(list_file)])
        assert exit_code == 1

    def test_output_to_file(self, tmp_path: Path) -> None:
        data = _make_minimal_save_dict()
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps(data), encoding="utf-8")

        output_file = tmp_path / "report.json"
        exit_code = main([str(snapshot), "--output", str(output_file)])
        assert exit_code == 0
        assert output_file.exists()

        result = json.loads(output_file.read_text(encoding="utf-8"))
        assert "basic_info" in result
        assert "findings" in result
        assert "summary" in result

    def test_output_to_nested_dir(self, tmp_path: Path) -> None:
        data = _make_minimal_save_dict()
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps(data), encoding="utf-8")

        output_file = tmp_path / "sub" / "dir" / "report.json"
        exit_code = main([str(snapshot), "--output", str(output_file)])
        assert exit_code == 0
        assert output_file.exists()

    def test_stdout_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        data = _make_minimal_save_dict()
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps(data), encoding="utf-8")

        exit_code = main([str(snapshot)])
        assert exit_code == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["total_findings"] == 0


# ── パターン定義自体のテスト ──────────────────────────────────

class TestPatternDefinitions:
    """検証パターン定義の構造的整合性テスト。"""

    def test_reference_patterns_have_required_keys(self) -> None:
        for p in REFERENCE_EXISTENCE_PATTERNS:
            assert "source_path" in p
            assert "id_extraction" in p
            assert "target_path" in p
            assert "target_id_field" in p
            assert "description" in p

    def test_accumulation_patterns_have_required_keys(self) -> None:
        for p in ACCUMULATION_LIMIT_PATTERNS:
            assert "field_path" in p
            assert "limit" in p
            assert "description" in p
            assert isinstance(p["limit"], int)
            assert p["limit"] > 0

    def test_timestamp_patterns_have_required_keys(self) -> None:
        for p in TIMESTAMP_ORDER_PATTERNS:
            assert "earlier_path" in p
            assert "later_path" in p
            assert "description" in p

    def test_required_field_patterns_have_required_keys(self) -> None:
        for p in REQUIRED_FIELD_PATTERNS:
            assert "field_path" in p
            assert "description" in p

    def test_version_patterns_are_sorted(self) -> None:
        versions = list(VERSION_FIELD_PATTERNS.keys())
        assert versions == sorted(versions)

    def test_version_patterns_have_non_empty_fields(self) -> None:
        for ver, fields in VERSION_FIELD_PATTERNS.items():
            assert isinstance(ver, int)
            assert len(fields) > 0
            for f in fields:
                assert isinstance(f, str)
                assert len(f) > 0


# ── エッジケーステスト ─────────────────────────────────────────

class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_very_large_version_number(self) -> None:
        data = _make_minimal_save_dict(version=9999)
        result = check_integrity(data)
        # versionは数値なので必須フィールドチェックは通過するが、
        # version 42以降のフィールドがないと検出される場合がある
        # ただし全フィールドが存在するので0件であるべき
        assert result["total_findings"] == 0

    def test_version_zero(self) -> None:
        data = {"version": 0, "tick_count": 0, "psyche": {"emotions": {"joy": 0.5}, "drives": {"social": 0.5}, "mood": {"valence": 0.0, "arousal": 0.0}}}
        result = check_integrity(data)
        # バージョン0は有効なバージョン番号だがv1のフィールドも不要
        assert isinstance(result["total_findings"], int)

    def test_negative_tick_count(self) -> None:
        data = _make_minimal_save_dict()
        data["tick_count"] = -1
        result = check_integrity(data)
        # tick_count は存在するので必須フィールドチェックは通過
        required_findings = [
            f for f in result["findings"] if f["field_path"] == "tick_count"
        ]
        assert len(required_findings) == 0

    def test_non_list_where_list_expected(self) -> None:
        data = _make_minimal_save_dict()
        data["last_episodes"]["episodes"] = "not a list"
        data["last_episodes"]["links"] = "not a list either"
        # エラーを起こさずにスキップすることを確認
        result = check_integrity(data)
        assert isinstance(result, dict)

    def test_nested_none_values(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"] = None
        result = check_integrity(data)
        assert isinstance(result, dict)

    def test_float_version(self) -> None:
        data = _make_minimal_save_dict()
        data["version"] = 42.0
        result = check_integrity(data)
        # float versionは有効として扱われる
        version_findings = [
            f for f in result["findings"]
            if f.get("pattern") == "version_consistency"
            and "バージョン番号" in f.get("fact", "")
        ]
        assert len(version_findings) == 0

    def test_concurrent_multiple_issues(self) -> None:
        """複数パターンから同時に検出されるケース。"""
        data = _make_minimal_save_dict()
        # 壊れた参照
        data["last_episodes"]["links"].append({
            "from_episode_id": "ep-GHOST",
            "to_episode_id": "ep-001",
            "link_type": "temporal",
            "strength": 0.5,
        })
        # 上限超過
        data["last_episodes"]["episodes"] = [
            {"episode_id": f"ep-{i}", "summary": f"ep {i}"}
            for i in range(300)
        ]
        # 時刻逆転
        data["dynamics"]["phase_entered_at"] = 9999.0
        # 必須フィールド欠損
        data["psyche"]["emotions"] = {}
        # バージョンフィールド欠損
        del data["hypothesis_observation_pairing_state"]

        result = check_integrity(data)
        assert result["summary"]["reference_existence"] >= 1
        assert result["summary"]["accumulation_limit"] >= 1
        assert result["summary"]["timestamp_order"] >= 1
        assert result["summary"]["required_field"] >= 1
        assert result["summary"]["version_consistency"] >= 1

    def test_missing_sentinel_repr(self) -> None:
        assert repr(_MISSING) == "<MISSING>"


# ── 状態非保持テスト ──────────────────────────────────────────

class TestStatelessness:
    """ツールが状態を保持しないことの確認テスト。"""

    def test_repeated_calls_independent(self) -> None:
        """連続呼び出しの結果が互いに独立であることを確認。"""
        data1 = _make_minimal_save_dict()
        data1["last_episodes"]["links"] = [
            {
                "from_episode_id": "ep-BROKEN",
                "to_episode_id": "ep-002",
                "link_type": "temporal",
                "strength": 0.5,
            },
        ]
        result1 = check_integrity(data1)

        data2 = _make_minimal_save_dict()
        result2 = check_integrity(data2)

        # 1回目の呼び出しの結果が2回目に影響しない
        assert result1["total_findings"] > 0
        assert result2["total_findings"] == 0

    def test_result_does_not_reference_input(self) -> None:
        """結果辞書が入力辞書への参照を持たないことを確認。"""
        data = _make_minimal_save_dict()
        result = check_integrity(data)
        # 入力を変更しても結果に影響しない
        data["version"] = 999
        assert result["basic_info"]["version"] == 42

    def test_three_consecutive_calls(self) -> None:
        """3回連続で異なる辞書を渡して結果が独立であることを確認。"""
        results = []
        for i in range(3):
            data = _make_minimal_save_dict()
            if i == 1:
                data["dynamics"]["intensity_history"] = [0.5] * 15
            result = check_integrity(data)
            results.append(result)

        assert results[0]["total_findings"] == 0
        assert results[1]["total_findings"] > 0
        assert results[2]["total_findings"] == 0


# ── パターン6: 型構造基本確認テスト ──────────────────────────

class TestTypeStructure:
    """パターン6: 型構造の基本確認のテスト。"""

    def test_valid_types_no_findings(self) -> None:
        data = _make_minimal_save_dict()
        findings = _check_type_structure(data)
        assert len(findings) == 0

    def test_version_wrong_type(self) -> None:
        data = _make_minimal_save_dict()
        data["version"] = "not_a_number"
        findings = _check_type_structure(data)
        version_findings = [
            f for f in findings if f["field_path"] == "version"
        ]
        assert len(version_findings) == 1
        assert "numeric" in version_findings[0]["fact"]

    def test_tick_count_wrong_type(self) -> None:
        data = _make_minimal_save_dict()
        data["tick_count"] = "string_tick"
        findings = _check_type_structure(data)
        tick_findings = [
            f for f in findings if f["field_path"] == "tick_count"
        ]
        assert len(tick_findings) == 1

    def test_psyche_wrong_type(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"] = [1, 2, 3]
        findings = _check_type_structure(data)
        psyche_findings = [
            f for f in findings if f["field_path"] == "psyche"
        ]
        assert len(psyche_findings) == 1
        assert "dict" in psyche_findings[0]["fact"]

    def test_emotions_wrong_type(self) -> None:
        data = _make_minimal_save_dict()
        data["psyche"]["emotions"] = "not_a_dict"
        findings = _check_type_structure(data)
        emotion_findings = [
            f for f in findings if f["field_path"] == "psyche.emotions"
        ]
        assert len(emotion_findings) == 1

    def test_loop_state_wrong_type(self) -> None:
        data = _make_minimal_save_dict()
        data["loop_state"] = "wrong"
        findings = _check_type_structure(data)
        ls_findings = [
            f for f in findings if f["field_path"] == "loop_state"
        ]
        assert len(ls_findings) == 1

    def test_dynamics_wrong_type(self) -> None:
        data = _make_minimal_save_dict()
        data["dynamics"] = [1, 2]
        findings = _check_type_structure(data)
        dyn_findings = [
            f for f in findings if f["field_path"] == "dynamics"
        ]
        assert len(dyn_findings) == 1

    def test_missing_field_skips(self) -> None:
        """存在しないフィールドはスキップする（パターン4/5で検出済み）。"""
        data = {"version": 1}
        findings = _check_type_structure(data)
        # version is numeric -> OK, but psyche etc. are missing -> skipped
        assert all(f["field_path"] != "psyche" for f in findings)

    def test_none_field_skips(self) -> None:
        """Noneフィールドはスキップする（パターン4で検出済み）。"""
        data = _make_minimal_save_dict()
        data["psyche"] = None
        findings = _check_type_structure(data)
        psyche_findings = [
            f for f in findings if f["field_path"] == "psyche"
        ]
        assert len(psyche_findings) == 0

    def test_all_findings_have_correct_pattern(self) -> None:
        data = _make_minimal_save_dict()
        data["version"] = "bad"
        data["psyche"] = "bad"
        findings = _check_type_structure(data)
        for f in findings:
            assert f["pattern"] == "type_structure"
            assert "field_path" in f
            assert "fact" in f

    def test_int_version_ok(self) -> None:
        data = _make_minimal_save_dict()
        data["version"] = 42
        findings = _check_type_structure(data)
        version_findings = [
            f for f in findings if f["field_path"] == "version"
        ]
        assert len(version_findings) == 0

    def test_float_version_ok(self) -> None:
        data = _make_minimal_save_dict()
        data["version"] = 42.0
        findings = _check_type_structure(data)
        version_findings = [
            f for f in findings if f["field_path"] == "version"
        ]
        assert len(version_findings) == 0

    def test_integration_in_check_integrity(self) -> None:
        """check_integrity経由でパターン6が実行されることを確認。"""
        data = _make_minimal_save_dict()
        data["version"] = "bad_version"
        result = check_integrity(data)
        assert result["summary"]["type_structure"] >= 1
        type_findings = [
            f for f in result["findings"]
            if f["pattern"] == "type_structure"
        ]
        assert len(type_findings) >= 1


# ── パターン2追加分: 新規蓄積上限テスト ─────────────────────

class TestNewAccumulationLimits:
    """パターン2に追加された新規蓄積上限パターンのテスト。"""

    def test_spontaneous_recall_history_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["spontaneous_recall_state"] = {
            "recent_recall_history": [f"id-{i}" for i in range(500)],
        }
        findings = _check_accumulation_limits(data)
        sr_findings = [
            f for f in findings
            if "spontaneous_recall_state.recent_recall_history" in f["field_path"]
        ]
        assert len(sr_findings) == 1

    def test_internal_contradiction_window_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["internal_contradiction_state"] = {
            "contradiction_window": [{"id": i} for i in range(60)],
        }
        findings = _check_accumulation_limits(data)
        ic_findings = [
            f for f in findings
            if "internal_contradiction_state.contradiction_window" in f["field_path"]
        ]
        assert len(ic_findings) == 1

    def test_perceptual_context_summaries_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["perceptual_context_state"] = {
            "summaries": [{"id": i} for i in range(60)],
        }
        findings = _check_accumulation_limits(data)
        pc_findings = [
            f for f in findings
            if "perceptual_context_state.summaries" in f["field_path"]
        ]
        assert len(pc_findings) == 1

    def test_persistent_commitment_items_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["persistent_commitment_state"] = {
            "items": [{"id": i} for i in range(10)],
        }
        findings = _check_accumulation_limits(data)
        pc_findings = [
            f for f in findings
            if "persistent_commitment_state.items" in f["field_path"]
        ]
        assert len(pc_findings) == 1

    def test_self_action_perception_records_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["self_action_perception_state"] = {
            "records": [{"id": i} for i in range(60)],
        }
        findings = _check_accumulation_limits(data)
        sap_findings = [
            f for f in findings
            if "self_action_perception_state.records" in f["field_path"]
        ]
        assert len(sap_findings) == 1

    def test_intent_action_gap_records_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["intent_action_gap_state"] = {
            "records": [{"id": i} for i in range(60)],
        }
        findings = _check_accumulation_limits(data)
        iag_findings = [
            f for f in findings
            if "intent_action_gap_state.records" in f["field_path"]
        ]
        assert len(iag_findings) == 1

    def test_temporal_cognition_elapsed_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["temporal_cognition_state"] = {
            "elapsed_records": [{"id": i} for i in range(150)],
        }
        findings = _check_accumulation_limits(data)
        tc_findings = [
            f for f in findings
            if "temporal_cognition_state.elapsed_records" in f["field_path"]
        ]
        assert len(tc_findings) == 1

    def test_stabilization_history_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["stabilization_description_state"] = {
            "history": [{"id": i} for i in range(40)],
        }
        findings = _check_accumulation_limits(data)
        sd_findings = [
            f for f in findings
            if "stabilization_description_state.history" in f["field_path"]
        ]
        assert len(sd_findings) == 1

    def test_behavioral_diversity_history_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["behavioral_diversity_state"] = {
            "history": [{"id": i} for i in range(40)],
        }
        findings = _check_accumulation_limits(data)
        bd_findings = [
            f for f in findings
            if "behavioral_diversity_state.history" in f["field_path"]
        ]
        assert len(bd_findings) == 1

    def test_emotional_backdrop_window_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["emotional_backdrop_state"] = {
            "sliding_window": [{"id": i} for i in range(40)],
        }
        findings = _check_accumulation_limits(data)
        eb_findings = [
            f for f in findings
            if "emotional_backdrop_state.sliding_window" in f["field_path"]
        ]
        assert len(eb_findings) == 1

    def test_selection_attribution_records_exceed(self) -> None:
        data = _make_minimal_save_dict()
        data["selection_attribution_state"] = {
            "records": [{"id": i} for i in range(60)],
        }
        findings = _check_accumulation_limits(data)
        sa_findings = [
            f for f in findings
            if "selection_attribution_state.records" in f["field_path"]
        ]
        assert len(sa_findings) == 1

    def test_within_limit_no_findings(self) -> None:
        """上限以内の場合は検出なし。"""
        data = _make_minimal_save_dict()
        data["self_action_perception_state"] = {
            "records": [{"id": i} for i in range(50)],
        }
        findings = _check_accumulation_limits(data)
        sap_findings = [
            f for f in findings
            if "self_action_perception_state.records" in f["field_path"]
        ]
        assert len(sap_findings) == 0

    def test_all_new_patterns_have_required_keys(self) -> None:
        """追加された全パターンが必須キーを持つことを確認。"""
        for p in ACCUMULATION_LIMIT_PATTERNS:
            assert "field_path" in p
            assert "limit" in p
            assert "description" in p
            assert isinstance(p["limit"], int)
            assert p["limit"] > 0


# ── パターン6定義テスト ──────────────────────────────────────

class TestTypeStructurePatternDefinitions:
    """型構造パターン定義の構造的整合性テスト。"""

    def test_patterns_have_required_keys(self) -> None:
        for p in TYPE_STRUCTURE_PATTERNS:
            assert "field_path" in p
            assert "expected_type" in p
            assert "description" in p
            assert p["expected_type"] in ("dict", "list", "numeric")


# ── persistence.py load時自動検証テスト ─────────────────────

class TestPersistenceAutoVerification:
    """persistence.py のload時自動検証のテスト。"""

    def test_integrity_check_enabled_by_default(self, tmp_path: Path) -> None:
        """デフォルトで整合性検証が有効であることを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path)
        assert mgr._integrity_check is True

    def test_integrity_check_disabled_by_flag(self, tmp_path: Path) -> None:
        """構成フラグで無効化できることを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=False)
        assert mgr._integrity_check is False

    def test_integrity_check_env_override_true(self, tmp_path: Path) -> None:
        """環境変数で有効化を上書きできることを確認。"""
        from psyche.persistence import PersistenceManager
        with patch.dict(os.environ, {"CYRENE_INTEGRITY_CHECK": "1"}):
            mgr = PersistenceManager(directory=tmp_path, integrity_check=False)
            assert mgr._integrity_check is True

    def test_integrity_check_env_override_false(self, tmp_path: Path) -> None:
        """環境変数で無効化を上書きできることを確認。"""
        from psyche.persistence import PersistenceManager
        with patch.dict(os.environ, {"CYRENE_INTEGRITY_CHECK": "0"}):
            mgr = PersistenceManager(directory=tmp_path, integrity_check=True)
            assert mgr._integrity_check is False

    def test_integrity_check_env_true_variants(self, tmp_path: Path) -> None:
        """環境変数の様々なtrue値を確認。"""
        from psyche.persistence import PersistenceManager
        for val in ("1", "true", "True", "TRUE", "yes", "Yes", "YES"):
            with patch.dict(os.environ, {"CYRENE_INTEGRITY_CHECK": val}):
                mgr = PersistenceManager(directory=tmp_path, integrity_check=False)
                assert mgr._integrity_check is True, f"Failed for env value: {val}"

    def test_load_calls_integrity_check_when_enabled(
        self, tmp_path: Path,
    ) -> None:
        """loadが整合性検証を呼び出すことを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=True)

        # 最小限のスナップショットファイルを作成
        data = _make_minimal_save_dict()
        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(mgr, '_run_integrity_check') as mock_check:
            mgr.load()
            mock_check.assert_called_once()

    def test_load_skips_integrity_check_when_disabled(
        self, tmp_path: Path,
    ) -> None:
        """loadが整合性検証をスキップすることを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=False)

        data = _make_minimal_save_dict()
        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(mgr, '_run_integrity_check') as mock_check:
            mgr.load()
            mock_check.assert_not_called()

    def test_integrity_check_exception_does_not_break_load(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """整合性検証で例外が発生してもloadが失敗しないことを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=True)

        data = _make_minimal_save_dict()
        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(json.dumps(data), encoding="utf-8")

        with patch('tools.persistence_integrity.check_integrity',
                   side_effect=RuntimeError("test explosion")):
            with caplog.at_level(logging.WARNING):
                # load should NOT raise - exception is absorbed
                result = mgr.load()
                # Result may be None due to Snapshot.from_dict expecting specific format,
                # but the key point is no exception was raised
                assert "Integrity check failed with exception" in caplog.text

    def test_run_integrity_check_logs_findings(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """検証結果がログに出力されることを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=True)

        # 壊れたデータを作成
        data = _make_minimal_save_dict()
        data["dynamics"]["intensity_history"] = [0.5] * 15  # 上限10を超過

        with caplog.at_level(logging.WARNING):
            mgr._run_integrity_check(data)

        assert "finding(s) detected" in caplog.text

    def test_run_integrity_check_logs_clean(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """問題なしの場合のログを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=True)

        data = _make_minimal_save_dict()

        with caplog.at_level(logging.INFO):
            mgr._run_integrity_check(data)

        assert "Integrity check passed" in caplog.text

    def test_run_integrity_check_does_not_modify_data(
        self, tmp_path: Path,
    ) -> None:
        """整合性検証が入力データを変更しないことを確認。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=True)

        data = _make_minimal_save_dict()
        original = copy.deepcopy(data)
        mgr._run_integrity_check(data)
        assert data == original

    def test_load_result_unaffected_by_integrity_findings(
        self, tmp_path: Path,
    ) -> None:
        """整合性検証の結果がloadの戻り値に影響しないことを確認。
        検証結果にかかわらず、loadの成否判定は既存仕様のまま。"""
        from psyche.persistence import PersistenceManager
        mgr = PersistenceManager(directory=tmp_path, integrity_check=True)

        # intensity_historyが上限超過するデータでもloadの成否は変わらない
        data = _make_minimal_save_dict()
        data["dynamics"]["intensity_history"] = [0.5] * 15

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(mgr, '_run_integrity_check') as mock_check:
            mgr.load()
            # 検証は呼ばれるが、結果がloadの戻り値を変えない
            mock_check.assert_called_once()
