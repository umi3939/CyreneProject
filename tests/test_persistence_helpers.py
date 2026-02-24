"""
tests/test_persistence_helpers.py - persistence_helpers.py のテスト

テスト項目:
- 段階A: マイグレーションチェーンの系統化
  - マイグレーション定義テーブルの整合性
  - バージョン番号の一意性・単調増加
  - フィールド追加の一意性（重複なし）
  - get_fields_added_in_version / get_all_known_field_keys / get_version_for_field
- 段階B: 共通保存・復元ヘルパー
  - save_field / load_field の各インターフェース種別のテスト
  - save_fields / load_fields の統合テスト
  - エラー耐性（不正な属性パスなど）
  - 出力同一性検証（既存メソッドとの完全一致）
- 段階C: セマンティックグルーピング
  - グループの非空性
  - グルーピングユーティリティ
  - フィールド定義の整合性検証
- フィールド定義 FIELD_DEFINITIONS の検証
  - 全フィールドのキー名がマイグレーションチェーンに含まれること
  - 全フィールドのキー名に重複がないこと
  - 外部関数参照の完備性
  - orchestratorとの出力同一性
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from psyche.persistence_helpers import (
    FieldDef,
    SaveInterface,
    LoadInterface,
    SemanticGroup,
    MigrationEntry,
    MIGRATION_CHAIN,
    CURRENT_VERSION,
    get_migration_chain,
    get_fields_added_in_version,
    get_all_known_field_keys,
    get_version_for_field,
    save_field,
    save_fields,
    load_field,
    load_fields,
    get_fields_by_group,
    get_group_keys,
    get_all_field_keys,
    validate_field_defs,
)

from psyche.orchestrator import (
    PsycheOrchestrator,
    FIELD_DEFINITIONS,
)
from psyche.state import Percept


# ── Helpers ─────────────────────────────────────────────────────


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


# ── 段階A: マイグレーションチェーン テスト ──────────────────────


class TestMigrationChain:
    """段階A: マイグレーションチェーンの系統化テスト。"""

    def test_migration_chain_not_empty(self):
        """マイグレーションチェーンが空でない。"""
        assert len(MIGRATION_CHAIN) > 0

    def test_migration_chain_returns_same(self):
        """get_migration_chain() が MIGRATION_CHAIN と同一。"""
        assert get_migration_chain() is MIGRATION_CHAIN

    def test_current_version_matches_latest(self):
        """CURRENT_VERSION が最新マイグレーションのバージョンと一致。"""
        latest = max(entry.version for entry in MIGRATION_CHAIN)
        assert CURRENT_VERSION == latest

    def test_version_numbers_unique(self):
        """バージョン番号に重複がない。"""
        versions = [entry.version for entry in MIGRATION_CHAIN]
        assert len(versions) == len(set(versions))

    def test_version_numbers_monotonic(self):
        """バージョン番号が単調増加。"""
        versions = [entry.version for entry in MIGRATION_CHAIN]
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1], (
                f"Version {versions[i]} is not greater than {versions[i - 1]}"
            )

    def test_field_keys_unique_across_versions(self):
        """フィールドキー名がバージョン間で重複しない。"""
        all_keys: list[str] = []
        for entry in MIGRATION_CHAIN:
            all_keys.extend(entry.added_fields)
        assert len(all_keys) == len(set(all_keys)), (
            f"Duplicate keys found: "
            f"{[k for k in all_keys if all_keys.count(k) > 1]}"
        )

    def test_get_fields_added_in_version(self):
        """get_fields_added_in_version が正しいフィールドを返す。"""
        v1_fields = get_fields_added_in_version(1)
        assert "psyche" in v1_fields
        assert "tick_count" in v1_fields

    def test_get_fields_added_in_nonexistent_version(self):
        """存在しないバージョンに対しては空タプルを返す。"""
        result = get_fields_added_in_version(9999)
        assert result == ()

    def test_get_all_known_field_keys(self):
        """全既知フィールドキーが取得できる。"""
        all_keys = get_all_known_field_keys()
        assert "psyche" in all_keys
        assert "loop_state" in all_keys
        assert "hypothesis_observation_pairing_state" in all_keys
        assert len(all_keys) >= 60  # 現在66フィールド

    def test_get_version_for_field(self):
        """フィールドの導入バージョンが正しく返される。"""
        assert get_version_for_field("psyche") == 1
        assert get_version_for_field("tendency_state") == 5
        assert get_version_for_field("hypothesis_observation_pairing_state") == 42

    def test_get_version_for_nonexistent_field(self):
        """存在しないフィールドには None を返す。"""
        assert get_version_for_field("nonexistent_field") is None

    def test_no_removed_or_renamed_fields_currently(self):
        """現状、廃止・改名フィールドが存在しない。"""
        for entry in MIGRATION_CHAIN:
            assert entry.removed_fields == (), (
                f"Version {entry.version} has removed fields"
            )
            assert entry.renamed_fields == {}, (
                f"Version {entry.version} has renamed fields"
            )

    def test_migration_entries_are_frozen(self):
        """MigrationEntry が frozen dataclass である。"""
        entry = MIGRATION_CHAIN[0]
        with pytest.raises(AttributeError):
            entry.version = 999  # type: ignore


# ── 段階B: 共通保存・復元ヘルパー テスト ────────────────────────


class TestSaveLoadHelpers:
    """段階B: 共通保存・復元ヘルパーのテスト。"""

    def test_save_fields_produces_all_keys(self):
        """save_fields が FIELD_DEFINITIONS の全キーを含む辞書を生成する。"""
        orch = PsycheOrchestrator()
        result = save_fields(orch, FIELD_DEFINITIONS)
        expected_keys = {f.key for f in FIELD_DEFINITIONS}
        assert set(result.keys()) == expected_keys

    def test_save_load_roundtrip_basic(self, tmp_path):
        """基本的なsave→loadラウンドトリップ。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True

    def test_save_load_roundtrip_with_ticks(self, tmp_path):
        """ティック実行後のsave→loadラウンドトリップ。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(5):
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        assert orch2.tick_count == 5

    def test_save_produces_valid_json_with_version(self, tmp_path):
        """save() が正しいバージョン番号のJSONを出力する。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.save()
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        assert data["version"] == CURRENT_VERSION
        assert "save_timestamp" in data
        assert "tick_count" in data

    def test_save_contains_all_field_definition_keys(self, tmp_path):
        """save() が FIELD_DEFINITIONS の全キーを含む。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        orch.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        for fdef in FIELD_DEFINITIONS:
            assert fdef.key in data, f"Missing save field: {fdef.key}"

    def test_load_nonexistent_returns_false(self, tmp_path):
        """存在しないファイルの load() は False を返す。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        result = orch.load()
        assert result is False

    def test_save_field_error_handling(self):
        """save_field がエラー時に空辞書を返す。"""
        fdef = FieldDef(
            key="nonexistent",
            attr_path="_nonexistent_attr",
            save_interface=SaveInterface.TO_DICT,
            load_interface=LoadInterface.DIRECT_ASSIGN,
            version=1,
        )
        orch = PsycheOrchestrator()
        key, value = save_field(orch, fdef)
        assert key == "nonexistent"
        assert value == {}

    def test_load_field_missing_key_returns_false(self):
        """load_field がデータにキーがない場合 False を返す。"""
        fdef = FieldDef(
            key="nonexistent",
            attr_path="_psyche",
            save_interface=SaveInterface.TO_DICT,
            load_interface=LoadInterface.DIRECT_ASSIGN,
            version=1,
        )
        orch = PsycheOrchestrator()
        result = load_field(orch, fdef, {})
        assert result is False

    def test_load_field_error_handling(self):
        """load_field がエラー時に False を返す。"""
        fdef = FieldDef(
            key="test_key",
            attr_path="_nonexistent_attr",
            save_interface=SaveInterface.TO_DICT,
            load_interface=LoadInterface.DIRECT_ASSIGN,
            version=1,
        )
        orch = PsycheOrchestrator()
        result = load_field(orch, fdef, {"test_key": {"some": "data"}})
        assert result is False

    def test_load_fields_returns_count(self, tmp_path):
        """load_fields が復元したフィールド数を返す。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        orch1.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        orch2 = PsycheOrchestrator()
        count = load_fields(orch2, FIELD_DEFINITIONS, data)
        # 初期状態でもpsycheなどのフィールドは保存されるので count > 0
        assert count > 0


# ── 段階B: 出力同一性検証テスト ──────────────────────────────────


class TestOutputIdentity:
    """共通ヘルパーの出力が既存メソッドと同一であることを検証する。"""

    def test_save_output_contains_all_expected_keys(self, tmp_path):
        """save() の出力が期待される全キーを含む。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        percept = _make_percept()
        for _ in range(5):
            orch.post_response_update(percept, delta_time=1.0)
        orch.save()

        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # 旧バージョンのテストで検証していた全キーが存在すること
        expected_keys = [
            # Core (v1)
            "version", "tick_count", "psyche", "loop_state", "dynamics",
            # v4
            "amplitude", "value_orientation", "self_ref_state",
            "last_self_view", "tendency_awareness", "last_diff_summary",
            "last_strain", "last_self_image", "last_coherence",
            "last_narrative", "last_episodes", "last_bindings",
            "last_trace", "last_consumption",
            "last_expectations", "last_motives",
            "last_other_model", "input_supply",
            # v5
            "tendency_state", "vector_state", "candidate_state",
            "transient_goal_state", "stability_valve",
            # v6
            "dispersion_state", "context_sensitivity_state", "last_coupling",
            # v7-v13
            "policy_expansion_state", "memory_integration_state",
            "real_feed_state", "text_dialogue_state",
            "spontaneous_state", "vo_validation_state",
            "forgetting_fixation_state",
            # v14-v23
            "action_result_state", "dialogue_learning_state",
            "meta_emotion_state", "self_action_perception_state",
            "expectation_action_diff_log", "intent_action_gap_state",
            "temporal_cognition_state", "multi_path_recall_state",
            "introspection_cross_section_state", "perceptual_context_state",
            "selection_attribution_state",
            # v24-v42
            "reference_frequency_state", "persistent_commitment_state",
            "stabilization_description_state", "behavioral_diversity_state",
            "spontaneous_recall_state", "internal_contradiction_state",
            "interaction_accumulation_state", "emotional_backdrop_state",
            "situational_self_presentation_state", "drive_variation_state",
            "expectation_lifecycle_state", "input_pathway_balance_state",
            "responsibility_temporal_trace_state", "emotion_cooccurrence_state",
            "other_boundary_accumulation_state",
            "forgetting_recall_balance_state", "attention_distribution_state",
            "goal_hierarchy_propagation_state",
            "hypothesis_observation_pairing_state",
        ]
        for key in expected_keys:
            assert key in data, f"Missing save field: {key}"
        assert data["version"] == CURRENT_VERSION

    def test_roundtrip_json_match(self, tmp_path):
        """save → load → save で JSON が一致する（全フィールド復元確認）。

        meta_emotion_state 等は load 時に apply_session_decay() が適用されるため
        差分が生じうる。それ以外の全フィールドが完全一致することを検証する。
        """
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        orch1 = PsycheOrchestrator(data_dir=dir_a, memory_count=10)
        emotions = ["happy", "sad", "loving", "angry", "neutral",
                     "surprised", "teasing", "scared", "happy", "sad"]
        valences = [0.7, -0.6, 0.8, -0.5, 0.0,
                    0.3, 0.4, -0.4, 0.6, -0.3]
        for i in range(20):
            idx = i % len(emotions)
            percept = _make_percept(
                emotion=emotions[idx],
                valence=valences[idx],
                text=f"テスト入力{i}",
            )
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.save()

        json_a = json.loads(
            (dir_a / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=dir_b, memory_count=10)
        orch2.load(path=dir_a / "psyche_snapshot.json")
        orch2.save()

        json_b = json.loads(
            (dir_b / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # session_decay が適用されるフィールドと psyche (fear_index) を除外
        skip_keys = {
            "meta_emotion_state", "emotional_backdrop_state",
            "drive_variation_state", "emotion_cooccurrence_state",
            "other_boundary_accumulation_state",
            "situational_self_presentation_state",
            "psyche", "save_timestamp",
        }
        for key in json_a:
            if key in skip_keys:
                continue
            assert json_a[key] == json_b[key], (
                f"Roundtrip mismatch on field '{key}'"
            )

    def test_version_in_saved_data(self, tmp_path):
        """保存データのバージョンが CURRENT_VERSION。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.save()
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        assert data["version"] == CURRENT_VERSION


# ── 段階C: セマンティックグルーピング テスト ─────────────────────


class TestSemanticGrouping:
    """段階C: セマンティックグルーピングのテスト。"""

    def test_all_groups_represented(self):
        """全セマンティックグループが少なくとも1つのフィールドを持つ。"""
        for group in SemanticGroup:
            fields = get_fields_by_group(FIELD_DEFINITIONS, group)
            assert len(fields) > 0, f"Group {group.value} has no fields"

    def test_get_fields_by_group_core(self):
        """コアグループのフィールドが正しく取得できる。"""
        core_fields = get_fields_by_group(FIELD_DEFINITIONS, SemanticGroup.CORE)
        core_keys = [f.key for f in core_fields]
        assert "psyche" in core_keys
        assert "loop_state" in core_keys
        assert "dynamics" in core_keys

    def test_get_fields_by_group_memory(self):
        """記憶グループのフィールドが正しく取得できる。"""
        memory_fields = get_fields_by_group(FIELD_DEFINITIONS, SemanticGroup.MEMORY)
        memory_keys = [f.key for f in memory_fields]
        assert "last_episodes" in memory_keys
        assert "last_bindings" in memory_keys

    def test_get_group_keys(self):
        """get_group_keys がキー名リストを返す。"""
        keys = get_group_keys(FIELD_DEFINITIONS, SemanticGroup.CORE)
        assert isinstance(keys, list)
        assert all(isinstance(k, str) for k in keys)
        assert "psyche" in keys

    def test_get_all_field_keys(self):
        """get_all_field_keys が全キー名を返す。"""
        keys = get_all_field_keys(FIELD_DEFINITIONS)
        assert len(keys) == len(FIELD_DEFINITIONS)

    def test_semantic_group_enum_values(self):
        """SemanticGroup の値が文字列。"""
        for group in SemanticGroup:
            assert isinstance(group.value, str)

    def test_grouping_does_not_affect_save_load(self, tmp_path):
        """グルーピングが保存・復元の成否に影響しない。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        orch1.save()
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # グループごとにフィールドが正しく保存されている
        for group in SemanticGroup:
            group_fields = get_fields_by_group(FIELD_DEFINITIONS, group)
            for fdef in group_fields:
                assert fdef.key in data, (
                    f"Field {fdef.key} (group={group.value}) missing from saved data"
                )


# ── FIELD_DEFINITIONS 整合性テスト ───────────────────────────────


class TestFieldDefinitionsIntegrity:
    """FIELD_DEFINITIONS の整合性テスト。"""

    def test_no_duplicate_keys(self):
        """フィールドキーに重複がない。"""
        keys = [f.key for f in FIELD_DEFINITIONS]
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"
        )

    def test_all_keys_in_migration_chain(self):
        """全フィールドキーがマイグレーションチェーンに含まれる。"""
        migration_keys = get_all_known_field_keys()
        for fdef in FIELD_DEFINITIONS:
            assert fdef.key in migration_keys, (
                f"Field {fdef.key} not in migration chain"
            )

    def test_migration_chain_covers_field_defs(self):
        """マイグレーションチェーンの全キーがFIELD_DEFINITIONS に含まれる
        （tick_count を除く）。"""
        migration_keys = get_all_known_field_keys()
        field_def_keys = {f.key for f in FIELD_DEFINITIONS}
        for mk in migration_keys:
            if mk == "tick_count":
                continue  # tick_count は特殊処理
            assert mk in field_def_keys, (
                f"Migration key {mk} not in FIELD_DEFINITIONS"
            )

    def test_external_save_has_save_func(self):
        """EXTERNAL_SAVE のフィールドに save_func が設定されている。"""
        for fdef in FIELD_DEFINITIONS:
            if fdef.save_interface == SaveInterface.EXTERNAL_SAVE:
                assert fdef.save_func is not None, (
                    f"Field {fdef.key}: EXTERNAL_SAVE but no save_func"
                )

    def test_external_load_has_load_func(self):
        """EXTERNAL_LOAD のフィールドに load_func が設定されている。"""
        for fdef in FIELD_DEFINITIONS:
            if fdef.load_interface == LoadInterface.EXTERNAL_LOAD:
                assert fdef.load_func is not None, (
                    f"Field {fdef.key}: EXTERNAL_LOAD but no load_func"
                )

    def test_state_attr_has_load_type(self):
        """STATE_ATTR / PRIVATE_STATE_ATTR / DIRECT_ASSIGN に load_type がある。"""
        requires_type = {
            LoadInterface.STATE_ATTR,
            LoadInterface.PRIVATE_STATE_ATTR,
            LoadInterface.DIRECT_ASSIGN,
        }
        for fdef in FIELD_DEFINITIONS:
            if fdef.load_interface in requires_type:
                assert fdef.load_type is not None, (
                    f"Field {fdef.key}: {fdef.load_interface.value} but no load_type"
                )

    def test_validate_field_defs_no_issues(self):
        """validate_field_defs が問題を検出しない。"""
        issues = validate_field_defs(FIELD_DEFINITIONS)
        assert issues == [], f"Field def validation issues: {issues}"

    def test_version_matches_migration(self):
        """各フィールドの version がマイグレーションチェーンと一致。"""
        for fdef in FIELD_DEFINITIONS:
            migration_version = get_version_for_field(fdef.key)
            assert migration_version is not None, (
                f"Field {fdef.key} not found in migration chain"
            )
            assert fdef.version == migration_version, (
                f"Field {fdef.key}: version {fdef.version} != "
                f"migration version {migration_version}"
            )

    def test_field_count_matches_expected(self):
        """FIELD_DEFINITIONS のフィールド数が期待値と一致。"""
        # tick_count を除いたマイグレーションフィールド数と一致するはず
        migration_keys = get_all_known_field_keys()
        expected_count = len(migration_keys) - 1  # tick_count を除外
        assert len(FIELD_DEFINITIONS) == expected_count, (
            f"Expected {expected_count} field defs, got {len(FIELD_DEFINITIONS)}"
        )


# ── 後方互換性テスト ────────────────────────────────────────────


class TestBackwardCompatibility:
    """既存の保存データとの後方互換性テスト。"""

    def test_load_empty_data_graceful(self):
        """空のデータでも load_fields がクラッシュしない。"""
        orch = PsycheOrchestrator()
        count = load_fields(orch, FIELD_DEFINITIONS, {})
        assert count == 0

    def test_load_partial_data_graceful(self, tmp_path):
        """一部のフィールドのみ含むデータからの復元。"""
        # v1 のデータのみを含むJSON
        minimal_data = {
            "version": 1,
            "tick_count": 3,
            "psyche": PsycheOrchestrator()._psyche.to_dict(),
        }
        save_path = tmp_path / "psyche_snapshot.json"
        save_path.write_text(
            json.dumps(minimal_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch.load()
        assert loaded is True
        assert orch.tick_count == 3

    def test_load_unknown_fields_ignored(self, tmp_path):
        """未知のフィールドが含まれていても問題なく復元できる。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        orch1.save()

        # 未知のフィールドを追加
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )
        data["future_unknown_field"] = {"some": "data"}
        (tmp_path / "psyche_snapshot.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True

    def test_corrupted_json_returns_false(self, tmp_path):
        """壊れたJSONファイルの場合 load() が False を返す。"""
        save_path = tmp_path / "psyche_snapshot.json"
        save_path.write_text("not valid json {{{", encoding="utf-8")

        orch = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch.load()
        assert loaded is False


# ── session_decay 検証テスト ────────────────────────────────────


class TestSessionDecay:
    """session_decay フラグの適用テスト。"""

    def test_session_decay_fields_identified(self):
        """session_decay=True のフィールドが存在する。"""
        decay_fields = [f for f in FIELD_DEFINITIONS if f.session_decay]
        assert len(decay_fields) > 0
        decay_keys = {f.key for f in decay_fields}
        # 既知の session_decay フィールド
        assert "meta_emotion_state" in decay_keys
        assert "emotional_backdrop_state" in decay_keys
        assert "drive_variation_state" in decay_keys
        assert "emotion_cooccurrence_state" in decay_keys
        assert "other_boundary_accumulation_state" in decay_keys
        assert "situational_self_presentation_state" in decay_keys

    def test_validate_on_load_fields_identified(self):
        """validate_on_load=True のフィールドが存在する。"""
        validate_fields = [f for f in FIELD_DEFINITIONS if f.validate_on_load]
        assert len(validate_fields) == 1
        assert validate_fields[0].key == "persistent_commitment_state"


# ── FieldDef dataclass テスト ───────────────────────────────────


class TestFieldDefDataclass:
    """FieldDef dataclass のテスト。"""

    def test_field_def_frozen(self):
        """FieldDef が frozen であり変更不可。"""
        fdef = FieldDef(
            key="test",
            attr_path="_test",
            save_interface=SaveInterface.TO_DICT,
            load_interface=LoadInterface.DIRECT_ASSIGN,
        )
        with pytest.raises(AttributeError):
            fdef.key = "changed"  # type: ignore

    def test_field_def_defaults(self):
        """FieldDef のデフォルト値が正しい。"""
        fdef = FieldDef(
            key="test",
            attr_path="_test",
            save_interface=SaveInterface.TO_DICT,
            load_interface=LoadInterface.DIRECT_ASSIGN,
        )
        assert fdef.load_type is None
        assert fdef.session_decay is False
        assert fdef.validate_on_load is False
        assert fdef.version == 1
        assert fdef.group == SemanticGroup.CORE
        assert fdef.save_func is None
        assert fdef.load_func is None
        assert fdef.state_sub_attr is None
        assert fdef.nullable_check is True

    def test_save_interface_enum(self):
        """SaveInterface enum の値。"""
        assert SaveInterface.TO_DICT == "to_dict"
        assert SaveInterface.SAVE_METHOD == "save"
        assert SaveInterface.EXTERNAL_SAVE == "external_save"
        assert SaveInterface.RAW == "raw"

    def test_load_interface_enum(self):
        """LoadInterface enum の値。"""
        assert LoadInterface.STATE_ATTR == "state_attr"
        assert LoadInterface.PRIVATE_STATE_ATTR == "private_state_attr"
        assert LoadInterface.DIRECT_ASSIGN == "direct_assign"
        assert LoadInterface.LOAD_METHOD == "load_method"
        assert LoadInterface.EXTERNAL_LOAD == "external_load"
        assert LoadInterface.RAW == "raw"


# ── validate_field_defs ユーティリティテスト ──────────────────────


class TestValidateFieldDefs:
    """validate_field_defs ユーティリティのテスト。"""

    def test_duplicate_key_detected(self):
        """重複キーが検出される。"""
        defs = [
            FieldDef(key="dup", attr_path="_a",
                     save_interface=SaveInterface.RAW,
                     load_interface=LoadInterface.RAW),
            FieldDef(key="dup", attr_path="_b",
                     save_interface=SaveInterface.RAW,
                     load_interface=LoadInterface.RAW),
        ]
        issues = validate_field_defs(defs)
        assert any("Duplicate key" in issue for issue in issues)

    def test_missing_save_func_detected(self):
        """EXTERNAL_SAVE で save_func なしが検出される。"""
        defs = [
            FieldDef(key="ext", attr_path="_a",
                     save_interface=SaveInterface.EXTERNAL_SAVE,
                     load_interface=LoadInterface.RAW),
        ]
        issues = validate_field_defs(defs)
        assert any("EXTERNAL_SAVE" in issue for issue in issues)

    def test_missing_load_func_detected(self):
        """EXTERNAL_LOAD で load_func なしが検出される。"""
        defs = [
            FieldDef(key="ext", attr_path="_a",
                     save_interface=SaveInterface.RAW,
                     load_interface=LoadInterface.EXTERNAL_LOAD),
        ]
        issues = validate_field_defs(defs)
        assert any("EXTERNAL_LOAD" in issue for issue in issues)

    def test_missing_load_type_detected(self):
        """DIRECT_ASSIGN で load_type なしが検出される。"""
        defs = [
            FieldDef(key="da", attr_path="_a",
                     save_interface=SaveInterface.TO_DICT,
                     load_interface=LoadInterface.DIRECT_ASSIGN),
        ]
        issues = validate_field_defs(defs)
        assert any("load_type" in issue for issue in issues)
