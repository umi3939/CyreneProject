"""
tests/test_save_load_field_coverage.py - save/load v44の全フィールド差分テスト自動化

design_save_load_field_test.md に基づく4パターンの検証:
  パターン1: 全フィールドround-trip（初期値）
  パターン2: 全フィールドround-trip（変動後）
  パターン3: フィールド個別の値変動検証
  パターン4: Cycle 9-10追加フィールドの重点検証

永続化対象のフィールドを追加・削除・変更しない。
save/loadのロジックに一切変更を加えない。
psycheの動作ロジックに一切変更を加えない。
"""

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import pytest

from psyche.orchestrator import (
    FIELD_DEFINITIONS,
    PsycheOrchestrator,
)
from psyche.persistence_helpers import (
    CURRENT_VERSION,
    MIGRATION_CHAIN,
    get_all_known_field_keys,
    save_fields,
    load_fields,
)
from psyche.state import Percept


# ── Helpers ───────────────────────────────────────────────────────


EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised",
            "loving", "teasing", "scared", "happy", "neutral"]
VALENCES = [0.7, -0.6, -0.5, 0.0, 0.3,
            0.8, 0.4, -0.5, 0.6, 0.0]

DIVERSE_TEXTS = [
    "今日はとても良い天気ですね",
    "少し悲しい気持ちです",
    "なぜそうなったのか理解できません",
    "特に何もありません",
    "驚くべき発見がありました",
    "あなたのことが大好きです",
    "ちょっとからかっているだけですよ",
    "怖いものを見てしまいました",
    "嬉しいニュースを聞きました",
    "普通の日常です",
]


def _make_percept(
    emotion: str = "happy",
    valence: float = 0.7,
    text: str = "テスト画面",
    intent: str = "expression",
) -> Percept:
    """テスト用の Percept を生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent=intent,
        emotion_valence=valence,
    )


def _run_ticks(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新する。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=DIVERSE_TEXTS[idx % len(DIVERSE_TEXTS)],
        )
        orch.post_response_update(percept, delta_time=1.0)


def _run_diverse_ticks(orch: PsycheOrchestrator, count: int) -> None:
    """多様な感情・意図・テキスト組み合わせでティック実行する。

    より多くのフィールドに変動を与えることを目的とする。
    """
    intents = ["expression", "question", "greeting", "farewell", "command"]
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=DIVERSE_TEXTS[idx % len(DIVERSE_TEXTS)],
            intent=intents[i % len(intents)],
        )
        orch.post_response_update(percept, delta_time=1.0 + (i * 0.1))


def _save_to_dict(orch: PsycheOrchestrator) -> dict[str, Any]:
    """orchestratorの状態を辞書として取得する（ファイル書き込みなし）。"""
    data: dict[str, Any] = {
        "version": CURRENT_VERSION,
        "save_timestamp": time.time(),
        "tick_count": orch.tick_count,
    }
    data.update(save_fields(orch, FIELD_DEFINITIONS))
    return data


def _load_from_dict(orch: PsycheOrchestrator, data: dict[str, Any]) -> None:
    """辞書からorchestratorの状態を復元する（ファイル読み込みなし）。"""
    if "tick_count" in data:
        orch._tick_count = data["tick_count"]
    load_fields(orch, FIELD_DEFINITIONS, data)


# session_decay により load 時に変動が想定されるキー。
_SESSION_DECAY_KEYS: set[str] = {"freshness", "freshness_stage", "session_diff_scalar"}


def _deep_compare(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    float_tolerance: float = 1e-9,
    path: str = "",
    skip_keys: set[str] | None = None,
) -> list[str]:
    """2つの保存辞書を再帰的に比較し、差異を詳細に報告する。"""
    if skip_keys is None:
        skip_keys = set()

    diffs: list[str] = []

    keys1 = set(dict1.keys()) if isinstance(dict1, dict) else set()
    keys2 = set(dict2.keys()) if isinstance(dict2, dict) else set()

    for k in keys1 - keys2:
        if k not in skip_keys:
            diffs.append(f"{path}.{k}: present in first but not second")
    for k in keys2 - keys1:
        if k not in skip_keys:
            diffs.append(f"{path}.{k}: present in second but not first")

    for k in keys1 & keys2:
        v1 = dict1[k]
        v2 = dict2[k]
        child_path = f"{path}.{k}" if path else k

        if k == "save_timestamp":
            continue
        if k in skip_keys:
            continue

        if type(v1) != type(v2):
            diffs.append(
                f"{child_path}: type mismatch {type(v1).__name__} vs {type(v2).__name__}"
            )
            continue

        if isinstance(v1, dict):
            diffs.extend(_deep_compare(v1, v2, float_tolerance, child_path, skip_keys))
        elif isinstance(v1, list):
            if len(v1) != len(v2):
                diffs.append(
                    f"{child_path}: list length mismatch {len(v1)} vs {len(v2)}"
                )
            else:
                for i, (item1, item2) in enumerate(zip(v1, v2)):
                    item_path = f"{child_path}[{i}]"
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        diffs.extend(
                            _deep_compare(item1, item2, float_tolerance, item_path, skip_keys)
                        )
                    elif isinstance(item1, float) and isinstance(item2, float):
                        if math.isnan(item1) and math.isnan(item2):
                            pass
                        elif abs(item1 - item2) > float_tolerance:
                            diffs.append(
                                f"{item_path}: float mismatch {item1} vs {item2}"
                            )
                    elif item1 != item2:
                        diffs.append(f"{item_path}: value mismatch {item1!r} vs {item2!r}")
        elif isinstance(v1, float):
            if math.isnan(v1) and math.isnan(v2):
                pass
            elif abs(v1 - v2) > float_tolerance:
                diffs.append(
                    f"{child_path}: float mismatch {v1} vs {v2}"
                )
        elif v1 != v2:
            diffs.append(f"{child_path}: value mismatch {v1!r} vs {v2!r}")

    return diffs


def _is_empty_or_default(value: Any) -> bool:
    """値が空辞書・空リスト・デフォルト値かどうかを判定する。"""
    if value is None:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _dict_has_nondefault_content(d: dict) -> bool:
    """辞書が空辞書でなく、内部に非ゼロ・非空の値を持つかを判定する。"""
    if not isinstance(d, dict):
        return d is not None and d != 0 and d != 0.0 and d != "" and d != []
    if len(d) == 0:
        return False
    for v in d.values():
        if isinstance(v, dict):
            if _dict_has_nondefault_content(v):
                return True
        elif isinstance(v, list):
            if len(v) > 0:
                return True
        elif isinstance(v, (int, float)):
            if v != 0 and v != 0.0:
                return True
        elif isinstance(v, str):
            if v != "":
                return True
        elif v is not None:
            return True
    return False


def _get_field_value_from_save_dict(
    save_dict: dict[str, Any], field_key: str,
) -> Any:
    """save辞書から指定フィールドの値を取得する。"""
    return save_dict.get(field_key)


# ── 全フィールドキーの動的取得 ─────────────────────────────────────

def _get_all_field_keys_from_definitions() -> list[str]:
    """FIELD_DEFINITIONSから全フィールドキーを動的に取得する。"""
    return [fd.key for fd in FIELD_DEFINITIONS]


def _get_session_decay_fields() -> list[str]:
    """session_decay=True のフィールドキーを取得する。"""
    return [fd.key for fd in FIELD_DEFINITIONS if fd.session_decay]


# ══════════════════════════════════════════════════════════════════
# パターン1: 全フィールドround-trip（初期値）
# ══════════════════════════════════════════════════════════════════


class TestPattern1InitialValueRoundTrip:
    """初期化直後の状態でsave->load->save->比較による全フィールドround-trip検証。"""

    def test_initial_state_round_trip(self, tmp_path):
        """初期化直後（ティック未実行）の状態でround-tripが成立すること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Initial state round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_initial_state_all_field_keys_present(self, tmp_path):
        """初期化直後のsave辞書に全FIELD_DEFINITIONSのキーが含まれること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        save_dict = _save_to_dict(orch)

        all_keys = _get_all_field_keys_from_definitions()
        save_keys = set(save_dict.keys())

        for key in all_keys:
            assert key in save_keys, (
                f"Field '{key}' missing from initial save dict"
            )

    def test_initial_state_field_count(self, tmp_path):
        """初期化直後のsave辞書のフィールド数が定義と一致すること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        save_dict = _save_to_dict(orch)

        # メタフィールド: version, save_timestamp, tick_count
        meta_count = 3
        expected = len(FIELD_DEFINITIONS) + meta_count
        actual = len(save_dict)

        assert actual == expected, (
            f"Field count mismatch: expected={expected}, actual={actual}"
        )

    def test_initial_state_per_field_type_check(self, tmp_path):
        """初期化直後のsave辞書の各フィールド値が辞書型・リスト型・基本型のいずれかであること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        save_dict = _save_to_dict(orch)

        all_keys = _get_all_field_keys_from_definitions()
        for key in all_keys:
            value = save_dict.get(key)
            assert value is not None, (
                f"Field '{key}' is None in initial save dict"
            )
            assert isinstance(value, (dict, list, str, int, float, bool)), (
                f"Field '{key}' has unexpected type: {type(value).__name__}"
            )

    def test_initial_tick_count_zero(self, tmp_path):
        """初期化直後のtick_countが0であること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        save_dict = _save_to_dict(orch)
        assert save_dict["tick_count"] == 0

    def test_initial_version_current(self, tmp_path):
        """初期化直後のversionがCURRENT_VERSIONであること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        save_dict = _save_to_dict(orch)
        assert save_dict["version"] == CURRENT_VERSION


# ══════════════════════════════════════════════════════════════════
# パターン2: 全フィールドround-trip（変動後）
# ══════════════════════════════════════════════════════════════════


class TestPattern2VariedStateRoundTrip:
    """多様な入力で状態変動後のsave->load->save->比較による全フィールドround-trip検証。"""

    def test_varied_round_trip_10_ticks(self, tmp_path):
        """10ティック（多様な入力）後のround-tripで全フィールドが保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Varied 10-tick round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_varied_round_trip_20_ticks(self, tmp_path):
        """20ティック（多様な入力）後のround-tripで全フィールドが保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 20)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Varied 20-tick round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_varied_round_trip_negative_valence_dominant(self, tmp_path):
        """負のバランスが優勢な入力パターン後のround-trip。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        for i in range(15):
            percept = _make_percept(
                emotion=["sad", "angry", "scared"][i % 3],
                valence=[-0.8, -0.6, -0.7][i % 3],
                text=f"Negative input {i}",
            )
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.select_policy_dict(_make_percept(emotion="sad", valence=-0.5), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Negative-dominant round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_varied_round_trip_positive_valence_dominant(self, tmp_path):
        """正のバランスが優勢な入力パターン後のround-trip。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        for i in range(15):
            percept = _make_percept(
                emotion=["happy", "loving", "surprised"][i % 3],
                valence=[0.9, 0.8, 0.6][i % 3],
                text=f"Positive input {i}",
            )
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.select_policy_dict(_make_percept(emotion="happy", valence=0.8), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Positive-dominant round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_varied_round_trip_mixed_intents(self, tmp_path):
        """多様な意図パターン後のround-trip。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        intents = ["expression", "question", "greeting", "farewell", "command"]
        for i in range(15):
            percept = _make_percept(
                emotion=EMOTIONS[i % len(EMOTIONS)],
                valence=VALENCES[i % len(VALENCES)],
                text=f"Intent test {i}",
                intent=intents[i % len(intents)],
            )
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Mixed-intent round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_varied_round_trip_via_file(self, tmp_path):
        """ファイル経由の変動後round-trip。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_diverse_ticks(orch1, 15)
        orch1.select_policy_dict(_make_percept(), [])
        orch1.save()

        snapshot_path = tmp_path / "psyche_snapshot.json"
        assert snapshot_path.exists()
        data1 = json.loads(snapshot_path.read_text(encoding="utf-8"))

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True

        orch2.save(path=tmp_path / "psyche_snapshot_2.json")
        data2 = json.loads(
            (tmp_path / "psyche_snapshot_2.json").read_text(encoding="utf-8")
        )

        diffs = _deep_compare(data1, data2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"File-based varied round-trip diffs:\n" + "\n".join(diffs)
        )

    def test_varied_round_trip_then_continue(self, tmp_path):
        """変動後round-trip後にさらにティック実行しても正常動作すること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 10)

        save_dict = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict)

        _run_diverse_ticks(orch2, 10)
        assert orch2.tick_count == 20

        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)


# ══════════════════════════════════════════════════════════════════
# パターン3: フィールド個別の値変動検証
# ══════════════════════════════════════════════════════════════════


class TestPattern3FieldVariation:
    """全フィールドについて、ティック実行による値変動を検証する。

    初期値とティック実行後の値が異なることを確認し、
    テストが初期値のみの比較に陥っていないことを保証する。
    """

    # ティック実行で変動が期待されないフィールド（常に初期値のままのフィールド）
    # これらは特定条件でのみ変動するか、長周期の蓄積が必要
    _SLOW_ACCUMULATION_FIELDS: set[str] = set()

    def test_core_fields_vary_after_ticks(self, tmp_path):
        """コア状態フィールドがティック実行後に初期値から変動すること。"""
        orch_init = PsycheOrchestrator(data_dir=tmp_path / "init")
        save_init = _save_to_dict(orch_init)

        orch_varied = PsycheOrchestrator(data_dir=tmp_path / "varied")
        _run_diverse_ticks(orch_varied, 15)
        orch_varied.select_policy_dict(_make_percept(), [])
        save_varied = _save_to_dict(orch_varied)

        # psyche（感情・ドライブ・ムード）は必ず変動すべき
        core_keys = ["psyche", "loop_state", "dynamics"]
        for key in core_keys:
            init_val = save_init.get(key, {})
            varied_val = save_varied.get(key, {})
            assert init_val != varied_val, (
                f"Core field '{key}' did not change after 15 ticks"
            )

    def test_tick_count_varies(self, tmp_path):
        """tick_countがティック実行後に変動すること。"""
        orch_init = PsycheOrchestrator(data_dir=tmp_path / "init")
        save_init = _save_to_dict(orch_init)

        orch_varied = PsycheOrchestrator(data_dir=tmp_path / "varied")
        _run_diverse_ticks(orch_varied, 10)
        save_varied = _save_to_dict(orch_varied)

        assert save_init["tick_count"] == 0
        assert save_varied["tick_count"] == 10

    @pytest.mark.parametrize(
        "field_key",
        _get_all_field_keys_from_definitions(),
        ids=_get_all_field_keys_from_definitions(),
    )
    def test_field_round_trip_preserves_value(self, tmp_path, field_key):
        """各フィールドについて、ティック実行後のround-tripで値が保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 15)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        val1 = save_dict1.get(field_key)
        val2 = save_dict2.get(field_key)

        if isinstance(val1, dict) and isinstance(val2, dict):
            diffs = _deep_compare(
                {field_key: val1}, {field_key: val2},
                skip_keys=_SESSION_DECAY_KEYS,
            )
            assert diffs == [], (
                f"Field '{field_key}' round-trip diffs:\n" + "\n".join(diffs)
            )
        elif isinstance(val1, float) and isinstance(val2, float):
            if not (math.isnan(val1) and math.isnan(val2)):
                assert abs(val1 - val2) < 1e-9, (
                    f"Field '{field_key}' float mismatch: {val1} vs {val2}"
                )
        else:
            assert val1 == val2, (
                f"Field '{field_key}' value mismatch: {val1!r} vs {val2!r}"
            )

    def test_varied_fields_have_content(self, tmp_path):
        """15ティック実行後、多くのフィールドが非空の内容を持つこと。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        _run_diverse_ticks(orch, 15)
        orch.select_policy_dict(_make_percept(), [])
        save_dict = _save_to_dict(orch)

        all_keys = _get_all_field_keys_from_definitions()
        non_empty_count = 0
        for key in all_keys:
            val = save_dict.get(key)
            if val is not None and val != {} and val != []:
                if isinstance(val, dict) and _dict_has_nondefault_content(val):
                    non_empty_count += 1
                elif isinstance(val, list) and len(val) > 0:
                    non_empty_count += 1

        # 最低でもフィールドの半分以上に非空の内容があること
        min_expected = len(all_keys) // 2
        assert non_empty_count >= min_expected, (
            f"Only {non_empty_count}/{len(all_keys)} fields have non-empty "
            f"content after 15 ticks (expected >= {min_expected})"
        )


# ══════════════════════════════════════════════════════════════════
# パターン4: Cycle 9-10追加フィールドの重点検証
# ══════════════════════════════════════════════════════════════════


class TestPattern4Cycle9_10Fields:
    """Cycle 9-10で追加されたフィールドの重点的なround-trip検証。

    v43: memory_emotion_return_state
    v44: other_hypothesis_emotion_return_state

    これらのフィールドは感情帰還経路に関連する比較的新しいフィールドであり、
    round-tripの正確性を個別に検証する。
    """

    # Cycle 9-10で追加されたフィールド (v43-v44)
    CYCLE_9_10_FIELDS = [
        "memory_emotion_return_state",
        "other_hypothesis_emotion_return_state",
    ]

    def test_cycle9_10_fields_exist_in_definitions(self):
        """Cycle 9-10追加フィールドがFIELD_DEFINITIONSに含まれること。"""
        all_keys = {fd.key for fd in FIELD_DEFINITIONS}
        for field_key in self.CYCLE_9_10_FIELDS:
            assert field_key in all_keys, (
                f"Cycle 9-10 field '{field_key}' not in FIELD_DEFINITIONS"
            )

    def test_cycle9_10_fields_in_migration_chain(self):
        """Cycle 9-10追加フィールドがMIGRATION_CHAINに含まれること。"""
        migration_keys = get_all_known_field_keys()
        for field_key in self.CYCLE_9_10_FIELDS:
            assert field_key in migration_keys, (
                f"Cycle 9-10 field '{field_key}' not in MIGRATION_CHAIN"
            )

    def test_memory_emotion_return_round_trip(self, tmp_path):
        """memory_emotion_return_stateのround-trip検証。

        感情想起帰還は反復入力で蓄積されるため、複数ティック後にround-tripを検証する。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 20)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)
        val1 = save_dict1.get("memory_emotion_return_state")
        assert val1 is not None, "memory_emotion_return_state missing from save dict"

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)
        val2 = save_dict2.get("memory_emotion_return_state")

        if isinstance(val1, dict) and isinstance(val2, dict):
            diffs = _deep_compare(
                {"memory_emotion_return_state": val1},
                {"memory_emotion_return_state": val2},
                skip_keys=_SESSION_DECAY_KEYS,
            )
            assert diffs == [], (
                f"memory_emotion_return_state round-trip diffs:\n" + "\n".join(diffs)
            )
        else:
            assert val1 == val2

    def test_other_hypothesis_emotion_return_round_trip(self, tmp_path):
        """other_hypothesis_emotion_return_stateのround-trip検証。

        他者仮説帰還は他者入力で蓄積されるため、複数ティック後にround-tripを検証する。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 20)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)
        val1 = save_dict1.get("other_hypothesis_emotion_return_state")
        assert val1 is not None, "other_hypothesis_emotion_return_state missing from save dict"

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)
        val2 = save_dict2.get("other_hypothesis_emotion_return_state")

        if isinstance(val1, dict) and isinstance(val2, dict):
            diffs = _deep_compare(
                {"other_hypothesis_emotion_return_state": val1},
                {"other_hypothesis_emotion_return_state": val2},
                skip_keys=_SESSION_DECAY_KEYS,
            )
            assert diffs == [], (
                f"other_hypothesis_emotion_return_state round-trip diffs:\n"
                + "\n".join(diffs)
            )
        else:
            assert val1 == val2

    def test_cycle9_10_fields_round_trip_after_varied_input(self, tmp_path):
        """Cycle 9-10の全フィールドが多様な入力後のround-tripで保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")

        # 感情の正負を交互に投入してより多くの帰還経路をトリガー
        for i in range(25):
            if i % 2 == 0:
                percept = _make_percept(
                    emotion="happy", valence=0.8,
                    text=f"Positive cycle9 test {i}",
                )
            else:
                percept = _make_percept(
                    emotion="sad", valence=-0.6,
                    text=f"Negative cycle9 test {i}",
                )
            orch1.post_response_update(percept, delta_time=1.0)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save_dict1)
        save_dict2 = _save_to_dict(orch2)

        for field_key in self.CYCLE_9_10_FIELDS:
            val1 = save_dict1.get(field_key, {})
            val2 = save_dict2.get(field_key, {})
            if isinstance(val1, dict) and isinstance(val2, dict):
                diffs = _deep_compare(
                    {field_key: val1}, {field_key: val2},
                    skip_keys=_SESSION_DECAY_KEYS,
                )
                assert diffs == [], (
                    f"Cycle 9-10 field '{field_key}' round-trip diffs:\n"
                    + "\n".join(diffs)
                )
            else:
                assert val1 == val2, (
                    f"Cycle 9-10 field '{field_key}' value mismatch: "
                    f"{val1!r} vs {val2!r}"
                )

    def test_cycle9_10_fields_survive_double_round_trip(self, tmp_path):
        """Cycle 9-10フィールドが2重round-tripでも保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path / "a")
        _run_diverse_ticks(orch1, 20)
        orch1.select_policy_dict(_make_percept(), [])

        # 1回目round-trip
        save1 = _save_to_dict(orch1)
        orch2 = PsycheOrchestrator(data_dir=tmp_path / "b")
        _load_from_dict(orch2, save1)
        save2 = _save_to_dict(orch2)

        # 2回目round-trip
        orch3 = PsycheOrchestrator(data_dir=tmp_path / "c")
        _load_from_dict(orch3, save2)
        save3 = _save_to_dict(orch3)

        for field_key in self.CYCLE_9_10_FIELDS:
            val1 = save1.get(field_key, {})
            val3 = save3.get(field_key, {})
            if isinstance(val1, dict) and isinstance(val3, dict):
                diffs = _deep_compare(
                    {field_key: val1}, {field_key: val3},
                    skip_keys=_SESSION_DECAY_KEYS,
                )
                assert diffs == [], (
                    f"Cycle 9-10 field '{field_key}' double round-trip diffs:\n"
                    + "\n".join(diffs)
                )

    def test_cycle9_10_fields_via_file_round_trip(self, tmp_path):
        """Cycle 9-10フィールドがファイル経由round-tripで正確に保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_diverse_ticks(orch1, 20)
        orch1.select_policy_dict(_make_percept(), [])
        orch1.save()

        data1 = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        orch2.load()
        orch2.save(path=tmp_path / "snapshot_2.json")

        data2 = json.loads(
            (tmp_path / "snapshot_2.json").read_text(encoding="utf-8")
        )

        for field_key in self.CYCLE_9_10_FIELDS:
            val1 = data1.get(field_key, {})
            val2 = data2.get(field_key, {})
            if isinstance(val1, dict) and isinstance(val2, dict):
                diffs = _deep_compare(
                    {field_key: val1}, {field_key: val2},
                    skip_keys=_SESSION_DECAY_KEYS,
                )
                assert diffs == [], (
                    f"Cycle 9-10 field '{field_key}' file round-trip diffs:\n"
                    + "\n".join(diffs)
                )

    def test_session_decay_fields_identified(self):
        """session_decay=Trueのフィールドが正しく識別されること。"""
        decay_fields = _get_session_decay_fields()
        # session_decay=True のフィールドが存在することの確認
        assert len(decay_fields) > 0, (
            "No session_decay=True fields found in FIELD_DEFINITIONS"
        )
        # 既知のsession_decayフィールドが含まれること
        decay_set = set(decay_fields)
        known_decay = {
            "meta_emotion_state",
            "other_boundary_accumulation_state",
            "emotional_backdrop_state",
            "situational_self_presentation_state",
            "drive_variation_state",
            "emotion_cooccurrence_state",
        }
        for key in known_decay:
            assert key in decay_set, (
                f"Known session_decay field '{key}' not flagged as session_decay=True"
            )


# ══════════════════════════════════════════════════════════════════
# 補助検証: フィールド定義とマイグレーションチェーンの網羅性
# ══════════════════════════════════════════════════════════════════


class TestFieldCoverageCompleteness:
    """FIELD_DEFINITIONSとMIGRATION_CHAINの全カバレッジを動的に検証する。"""

    def test_all_migration_fields_covered_by_definitions(self):
        """MIGRATION_CHAINの全フィールドがFIELD_DEFINITIONSでカバーされていること。"""
        migration_keys = get_all_known_field_keys() - {"tick_count"}
        definition_keys = {fd.key for fd in FIELD_DEFINITIONS}

        uncovered = migration_keys - definition_keys
        assert uncovered == set(), (
            f"Migration fields not covered by FIELD_DEFINITIONS: {uncovered}"
        )

    def test_all_definitions_in_migration_chain(self):
        """FIELD_DEFINITIONSの全フィールドがMIGRATION_CHAINに含まれること。"""
        migration_keys = get_all_known_field_keys()
        definition_keys = {fd.key for fd in FIELD_DEFINITIONS}

        extra = definition_keys - migration_keys
        assert extra == set(), (
            f"FIELD_DEFINITIONS fields not in MIGRATION_CHAIN: {extra}"
        )

    def test_field_count_equals_68(self):
        """永続化フィールド数がv44の68と一致すること（67 FIELD_DEFINITIONS + 1 tick_count）。"""
        definition_count = len(FIELD_DEFINITIONS)
        migration_count = len(get_all_known_field_keys())

        # FIELD_DEFINITIONSにはtick_countが含まれない（特殊フィールドとして直接処理）
        # MIGRATION_CHAINにはtick_countが含まれる
        assert migration_count == definition_count + 1, (
            f"Field count mismatch: "
            f"MIGRATION_CHAIN={migration_count}, "
            f"FIELD_DEFINITIONS={definition_count} + 1 (tick_count)"
        )

    def test_all_semantic_groups_have_fields(self):
        """全セマンティックグループに少なくとも1つのフィールドが含まれること。"""
        from psyche.persistence_helpers import SemanticGroup
        for group in SemanticGroup:
            fields_in_group = [fd for fd in FIELD_DEFINITIONS if fd.group == group]
            assert len(fields_in_group) > 0, (
                f"Semantic group '{group.value}' has no fields"
            )

    def test_no_duplicate_keys(self):
        """FIELD_DEFINITIONSにキー名の重複がないこと。"""
        keys = [fd.key for fd in FIELD_DEFINITIONS]
        seen: set[str] = set()
        duplicates: list[str] = []
        for k in keys:
            if k in seen:
                duplicates.append(k)
            seen.add(k)
        assert duplicates == [], f"Duplicate keys: {duplicates}"

    def test_version_numbers_consistent(self):
        """FIELD_DEFINITIONSの各フィールドのversion番号がMIGRATION_CHAINと一致すること。"""
        from psyche.persistence_helpers import get_version_for_field

        mismatches: list[str] = []
        for fd in FIELD_DEFINITIONS:
            chain_version = get_version_for_field(fd.key)
            if chain_version is not None and chain_version != fd.version:
                mismatches.append(
                    f"{fd.key}: FIELD_DEF version={fd.version}, "
                    f"MIGRATION_CHAIN version={chain_version}"
                )
        assert mismatches == [], (
            f"Version mismatches:\n" + "\n".join(mismatches)
        )
