"""
psyche/persistence_helpers.py - Save/Load構造圧縮ヘルパー

design_save_load_compression.md に基づく3段階の構造改善:
  段階A: マイグレーションチェーンの系統化
  段階B: 共通保存・復元ヘルパー
  段階C: セマンティックグルーピングの宣言

既存の保存データとの後方互換性を完全に維持する。
永続化辞書のキー名・構造・JSONフォーマットは一切変更しない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type

logger = logging.getLogger(__name__)


# ── 段階C: セマンティックグループ定義 ──────────────────────────────

class SemanticGroup(str, Enum):
    """永続化フィールドの意味的カテゴリ。

    保守上の便宜であり、構造的意味を持たない。
    グループに属さないフィールドの追加を禁止しない。
    グルーピングの定義がフィールドの保存・復元の成否に影響しない。
    """
    CORE = "core"                       # 感情・ループ・ダイナミクス等の基底状態
    SELF_RECOGNITION = "self_recognition"  # 自己モデル・差分認知・自己像等
    MEMORY = "memory"                   # エピソード・バインディング・統合・忘却等
    OTHER_MODEL = "other_model"         # 他者観測・入力供給・対話学習等
    DESCRIPTION_COGNITION = "description_cognition"  # 各種断面記述モジュール


# ── 段階B: 保存・復元インターフェース種別 ──────────────────────────

class SaveInterface(str, Enum):
    """保存インターフェース種別。"""
    TO_DICT = "to_dict"           # obj.to_dict() を呼ぶ
    SAVE_METHOD = "save"          # obj.save() を呼ぶ
    EXTERNAL_SAVE = "external_save"  # 外部ヘルパー関数を使用
    RAW = "raw"                   # 直接辞書として保存


class LoadInterface(str, Enum):
    """復元インターフェース種別。"""
    STATE_ATTR = "state_attr"               # Type.from_dict() → obj.state = result
    PRIVATE_STATE_ATTR = "private_state_attr"  # Type.from_dict() → obj._state = result
    DIRECT_ASSIGN = "direct_assign"         # Type.from_dict() → self.attr = result
    LOAD_METHOD = "load_method"             # obj.load(data) を呼ぶ
    EXTERNAL_LOAD = "external_load"         # 外部ヘルパー関数を使用
    RAW = "raw"                             # 直接辞書として復元


# ── 段階B: フィールド定義 ──────────────────────────────────────────

@dataclass(frozen=True)
class FieldDef:
    """永続化対象フィールドの定義。

    Attributes:
        key: 永続化辞書内のキー名
        attr_path: オーケストレータ属性へのアクセス経路 (e.g. "_psyche", "_loop_state")
        save_interface: 保存インターフェース種別
        load_interface: 復元インターフェース種別
        load_type: from_dict() に渡す型 (load_method/external_load の場合は None)
        session_decay: apply_session_decay() の呼び出し有無
        validate_on_load: validate_on_load() の呼び出し有無
        version: このフィールドが導入されたバージョン番号
        group: セマンティックグループ名
        save_func: 外部保存関数参照 (EXTERNAL_SAVE の場合のみ)
        load_func: 外部復元関数参照 (EXTERNAL_LOAD の場合のみ)
        state_sub_attr: オブジェクトの state 属性からのサブパス (e.g. "state" → obj.state)
                        attr_path が "_tendency_sys" で state_sub_attr が "state" なら
                        save時: self._tendency_sys.state.to_dict()
                        load時: self._tendency_sys._state = Type.from_dict(data)
        nullable_check: True ならば save 時に None チェック (if obj else {}) する
    """
    key: str
    attr_path: str
    save_interface: SaveInterface
    load_interface: LoadInterface
    load_type: Optional[Type] = None
    session_decay: bool = False
    validate_on_load: bool = False
    version: int = 1
    group: SemanticGroup = SemanticGroup.CORE
    save_func: Optional[Callable] = None
    load_func: Optional[Callable] = None
    state_sub_attr: Optional[str] = None
    nullable_check: bool = True


# ── 段階A: マイグレーション定義テーブル ────────────────────────────

@dataclass(frozen=True)
class MigrationEntry:
    """バージョン間のマイグレーション定義。

    Attributes:
        version: バージョン番号
        added_fields: 追加されたフィールドのキー名リスト
        removed_fields: 廃止されたフィールドのキー名リスト (現状は空)
        renamed_fields: キー名変換規則 (旧キー名 → 新キー名) の辞書 (現状は空)
    """
    version: int
    added_fields: tuple[str, ...] = ()
    removed_fields: tuple[str, ...] = ()
    renamed_fields: dict[str, str] = field(default_factory=dict)


# マイグレーション定義テーブル: 各バージョンで追加されたフィールドを宣言
MIGRATION_CHAIN: tuple[MigrationEntry, ...] = (
    MigrationEntry(
        version=1,
        added_fields=(
            "psyche", "loop_state", "dynamics", "tick_count",
        ),
    ),
    MigrationEntry(
        version=4,
        added_fields=(
            "amplitude", "value_orientation",
            "self_ref_state", "last_self_view", "tendency_awareness",
            "last_diff_summary", "last_strain", "last_self_image",
            "last_coherence", "last_narrative",
            "last_episodes", "last_bindings",
            "last_trace", "last_consumption",
            "last_expectations", "last_motives",
            "last_other_model", "input_supply",
        ),
    ),
    MigrationEntry(
        version=5,
        added_fields=(
            "tendency_state", "vector_state", "candidate_state",
            "transient_goal_state", "stability_valve",
        ),
    ),
    MigrationEntry(
        version=6,
        added_fields=(
            "dispersion_state", "context_sensitivity_state", "last_coupling",
        ),
    ),
    MigrationEntry(
        version=7,
        added_fields=("policy_expansion_state",),
    ),
    MigrationEntry(
        version=8,
        added_fields=("memory_integration_state",),
    ),
    MigrationEntry(
        version=9,
        added_fields=("real_feed_state",),
    ),
    MigrationEntry(
        version=10,
        added_fields=("text_dialogue_state",),
    ),
    MigrationEntry(
        version=11,
        added_fields=("spontaneous_state",),
    ),
    MigrationEntry(
        version=12,
        added_fields=("vo_validation_state",),
    ),
    MigrationEntry(
        version=13,
        added_fields=("forgetting_fixation_state",),
    ),
    MigrationEntry(
        version=14,
        added_fields=("action_result_state",),
    ),
    MigrationEntry(
        version=15,
        added_fields=("dialogue_learning_state",),
    ),
    MigrationEntry(
        version=16,
        added_fields=("meta_emotion_state",),
    ),
    MigrationEntry(
        version=17,
        added_fields=("self_action_perception_state",),
    ),
    MigrationEntry(
        version=18,
        added_fields=("expectation_action_diff_log",),
    ),
    MigrationEntry(
        version=19,
        added_fields=("intent_action_gap_state",),
    ),
    MigrationEntry(
        version=20,
        added_fields=("temporal_cognition_state",),
    ),
    MigrationEntry(
        version=21,
        added_fields=("multi_path_recall_state",),
    ),
    MigrationEntry(
        version=22,
        added_fields=(
            "introspection_cross_section_state",
            "perceptual_context_state",
        ),
    ),
    MigrationEntry(
        version=23,
        added_fields=("selection_attribution_state",),
    ),
    MigrationEntry(
        version=24,
        added_fields=("reference_frequency_state",),
    ),
    MigrationEntry(
        version=25,
        added_fields=("persistent_commitment_state",),
    ),
    MigrationEntry(
        version=26,
        added_fields=("stabilization_description_state",),
    ),
    MigrationEntry(
        version=27,
        added_fields=("behavioral_diversity_state",),
    ),
    MigrationEntry(
        version=28,
        added_fields=("spontaneous_recall_state",),
    ),
    MigrationEntry(
        version=29,
        added_fields=("internal_contradiction_state",),
    ),
    MigrationEntry(
        version=30,
        added_fields=("interaction_accumulation_state",),
    ),
    MigrationEntry(
        version=31,
        added_fields=("emotional_backdrop_state",),
    ),
    MigrationEntry(
        version=32,
        added_fields=("situational_self_presentation_state",),
    ),
    MigrationEntry(
        version=33,
        added_fields=("drive_variation_state",),
    ),
    MigrationEntry(
        version=34,
        added_fields=("expectation_lifecycle_state",),
    ),
    MigrationEntry(
        version=35,
        added_fields=("input_pathway_balance_state",),
    ),
    MigrationEntry(
        version=36,
        added_fields=("responsibility_temporal_trace_state",),
    ),
    MigrationEntry(
        version=37,
        added_fields=("emotion_cooccurrence_state",),
    ),
    MigrationEntry(
        version=38,
        added_fields=("other_boundary_accumulation_state",),
    ),
    MigrationEntry(
        version=39,
        added_fields=("forgetting_recall_balance_state",),
    ),
    MigrationEntry(
        version=40,
        added_fields=("attention_distribution_state",),
    ),
    MigrationEntry(
        version=41,
        added_fields=("goal_hierarchy_propagation_state",),
    ),
    MigrationEntry(
        version=42,
        added_fields=("hypothesis_observation_pairing_state",),
    ),
    MigrationEntry(
        version=43,
        added_fields=("memory_emotion_return_state",),
    ),
    MigrationEntry(
        version=44,
        added_fields=("other_hypothesis_emotion_return_state",),
    ),
    MigrationEntry(
        version=45,
        added_fields=("return_pathway_history",),
    ),
)

# 現在のバージョン番号
CURRENT_VERSION: int = 45


# ── 段階A: マイグレーション処理 ────────────────────────────────────

def get_migration_chain() -> tuple[MigrationEntry, ...]:
    """マイグレーション定義テーブルを返す。"""
    return MIGRATION_CHAIN


def get_fields_added_in_version(version: int) -> tuple[str, ...]:
    """指定バージョンで追加されたフィールドのキー名リストを返す。"""
    for entry in MIGRATION_CHAIN:
        if entry.version == version:
            return entry.added_fields
    return ()


def get_all_known_field_keys() -> set[str]:
    """マイグレーション定義テーブルに含まれる全フィールドキー名を返す。"""
    keys: set[str] = set()
    for entry in MIGRATION_CHAIN:
        keys.update(entry.added_fields)
    return keys


def get_version_for_field(field_key: str) -> Optional[int]:
    """指定フィールドが導入されたバージョン番号を返す。該当なしなら None。"""
    for entry in MIGRATION_CHAIN:
        if field_key in entry.added_fields:
            return entry.version
    return None


# ── 段階B: 共通保存ヘルパー ────────────────────────────────────────

def save_field(orchestrator: Any, fdef: FieldDef) -> tuple[str, Any]:
    """フィールド定義に従い、1フィールドを保存用辞書エントリとして返す。

    Returns:
        (key, value) のタプル
    """
    try:
        obj = getattr(orchestrator, fdef.attr_path)

        # state_sub_attr がある場合、サブ属性を辿る
        if fdef.state_sub_attr and obj is not None:
            obj = getattr(obj, fdef.state_sub_attr, obj)

        if fdef.save_interface == SaveInterface.TO_DICT:
            if fdef.nullable_check and obj is None:
                return (fdef.key, {})
            value = obj.to_dict() if obj is not None else {}
            return (fdef.key, value)

        elif fdef.save_interface == SaveInterface.SAVE_METHOD:
            if fdef.nullable_check and obj is None:
                return (fdef.key, {})
            # obj 自体が save() を持つ場合 (state_sub_attr なし)
            target = getattr(orchestrator, fdef.attr_path)
            value = target.save() if target is not None else {}
            return (fdef.key, value)

        elif fdef.save_interface == SaveInterface.EXTERNAL_SAVE:
            if fdef.save_func is None:
                logger.warning("No save_func for external_save field %s", fdef.key)
                return (fdef.key, {})
            target_obj = getattr(orchestrator, fdef.attr_path)
            if fdef.nullable_check and target_obj is None:
                return (fdef.key, {})
            value = fdef.save_func(target_obj)
            return (fdef.key, value)

        elif fdef.save_interface == SaveInterface.RAW:
            if fdef.nullable_check and obj is None:
                return (fdef.key, {})
            return (fdef.key, obj)

        else:
            logger.warning("Unknown save_interface for field %s: %s", fdef.key, fdef.save_interface)
            return (fdef.key, {})

    except Exception as e:
        logger.warning("Error saving field %s: %s", fdef.key, e)
        return (fdef.key, {})


def save_fields(orchestrator: Any, field_defs: list[FieldDef]) -> dict[str, Any]:
    """フィールド定義リストに従い、全フィールドを保存用辞書として返す。"""
    result: dict[str, Any] = {}
    for fdef in field_defs:
        key, value = save_field(orchestrator, fdef)
        result[key] = value
    return result


# ── 段階B: 共通復元ヘルパー ────────────────────────────────────────

def load_field(orchestrator: Any, fdef: FieldDef, data: dict) -> bool:
    """フィールド定義に従い、1フィールドを復元する。

    Returns:
        True if the field was found and loaded, False otherwise.
    """
    if fdef.key not in data:
        return False

    raw_value = data[fdef.key]
    # Noneは「データなし」として扱う
    if raw_value is None:
        return False
    # dict系ロードインターフェースでは空dictは「未初期化状態」を意味するためスキップ
    # (RAWインターフェースでは空dictも有効なデータとして復元)
    if (isinstance(raw_value, dict) and not raw_value
            and fdef.load_interface != LoadInterface.RAW):
        return False

    try:
        field_data = data[fdef.key]

        if fdef.load_interface == LoadInterface.STATE_ATTR:
            # Type.from_dict() → obj.state = result
            if fdef.load_type is None:
                logger.warning("No load_type for state_attr field %s", fdef.key)
                return False
            restored = fdef.load_type.from_dict(field_data)
            obj = getattr(orchestrator, fdef.attr_path)
            obj.state = restored
            if fdef.session_decay:
                obj.state.apply_session_decay()
            if fdef.validate_on_load:
                obj.validate_on_load()
            return True

        elif fdef.load_interface == LoadInterface.PRIVATE_STATE_ATTR:
            # Type.from_dict() → obj._state = result
            if fdef.load_type is None:
                logger.warning("No load_type for private_state_attr field %s", fdef.key)
                return False
            restored = fdef.load_type.from_dict(field_data)
            obj = getattr(orchestrator, fdef.attr_path)
            obj._state = restored
            if fdef.session_decay:
                obj.state.apply_session_decay()
            if fdef.validate_on_load:
                obj.validate_on_load()
            return True

        elif fdef.load_interface == LoadInterface.DIRECT_ASSIGN:
            # Type.from_dict() → self.attr = result
            if fdef.load_type is None:
                logger.warning("No load_type for direct_assign field %s", fdef.key)
                return False
            restored = fdef.load_type.from_dict(field_data)
            setattr(orchestrator, fdef.attr_path, restored)
            if fdef.session_decay:
                restored_obj = getattr(orchestrator, fdef.attr_path)
                if hasattr(restored_obj, 'apply_session_decay'):
                    restored_obj.apply_session_decay()
            return True

        elif fdef.load_interface == LoadInterface.LOAD_METHOD:
            # obj.load(data) を呼ぶ
            obj = getattr(orchestrator, fdef.attr_path)
            obj.load(field_data)
            if fdef.session_decay:
                if hasattr(obj, 'state') and hasattr(obj.state, 'apply_session_decay'):
                    obj.state.apply_session_decay()
            return True

        elif fdef.load_interface == LoadInterface.EXTERNAL_LOAD:
            # 外部ヘルパー関数を使用
            if fdef.load_func is None:
                logger.warning("No load_func for external_load field %s", fdef.key)
                return False
            restored = fdef.load_func(field_data)
            setattr(orchestrator, fdef.attr_path, restored)
            return True

        elif fdef.load_interface == LoadInterface.RAW:
            # 直接辞書として復元
            setattr(orchestrator, fdef.attr_path, field_data)
            return True

        else:
            logger.warning("Unknown load_interface for field %s: %s", fdef.key, fdef.load_interface)
            return False

    except Exception as e:
        logger.warning("Error loading field %s: %s", fdef.key, e)
        return False


def load_fields(orchestrator: Any, field_defs: list[FieldDef], data: dict) -> int:
    """フィールド定義リストに従い、全フィールドを復元する。

    Returns:
        復元に成功したフィールド数
    """
    loaded_count = 0
    for fdef in field_defs:
        if load_field(orchestrator, fdef, data):
            loaded_count += 1
    return loaded_count


# ── 段階C: グルーピングユーティリティ ──────────────────────────────

def get_fields_by_group(
    field_defs: list[FieldDef],
    group: SemanticGroup,
) -> list[FieldDef]:
    """指定グループに属するフィールド定義を返す。"""
    return [f for f in field_defs if f.group == group]


def get_group_keys(
    field_defs: list[FieldDef],
    group: SemanticGroup,
) -> list[str]:
    """指定グループに属するフィールドのキー名リストを返す。"""
    return [f.key for f in field_defs if f.group == group]


def get_all_field_keys(field_defs: list[FieldDef]) -> list[str]:
    """全フィールドのキー名リストを返す。"""
    return [f.key for f in field_defs]


def validate_field_defs(field_defs: list[FieldDef]) -> list[str]:
    """フィールド定義リストの整合性を検証する。

    Returns:
        問題の記述リスト（空なら問題なし）
    """
    issues: list[str] = []

    # キー名の重複チェック
    keys = [f.key for f in field_defs]
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            issues.append(f"Duplicate key: {k}")
        seen.add(k)

    # external_save / external_load の関数参照チェック
    for f in field_defs:
        if f.save_interface == SaveInterface.EXTERNAL_SAVE and f.save_func is None:
            issues.append(f"Field {f.key}: EXTERNAL_SAVE but no save_func")
        if f.load_interface == LoadInterface.EXTERNAL_LOAD and f.load_func is None:
            issues.append(f"Field {f.key}: EXTERNAL_LOAD but no load_func")

    # load_type チェック
    for f in field_defs:
        if f.load_interface in (
            LoadInterface.STATE_ATTR,
            LoadInterface.PRIVATE_STATE_ATTR,
            LoadInterface.DIRECT_ASSIGN,
        ) and f.load_type is None:
            issues.append(f"Field {f.key}: {f.load_interface.value} but no load_type")

    # マイグレーションチェーンとの整合性チェック
    migration_keys = get_all_known_field_keys()
    field_def_keys = set(keys)
    # 特殊フィールドは save ヘルパーでは直接扱わない:
    # - tick_count: orchestrator.save/load で直接処理
    # - return_pathway_history: 外部ツールの発火履歴（orchestrator.save/load で直接処理）
    special_keys = {"tick_count", "return_pathway_history"}
    migration_without_special = migration_keys - special_keys
    for mk in migration_without_special:
        if mk not in field_def_keys:
            issues.append(f"Migration key {mk} not in field_defs")
    for fk in field_def_keys:
        if fk not in migration_keys:
            issues.append(f"Field def key {fk} not in migration chain")

    return issues
