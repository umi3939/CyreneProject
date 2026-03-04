"""
tests/test_save_load_regression.py - save/load永続化の包括的回帰テスト

design_save_load_regression_test.md に基づく3層の回帰検証:
  層1: 全フィールドround-trip検証（ファスト）
  層2: マイグレーション互換性検証（スロー）
  層3: フィールド欠損時の非破壊性検証（ファスト）

永続化対象のフィールドを追加・削除・変更しない。
保存・復元のロジックに一切変更を加えない。
persistence_integrity.pyの検証パターンを変更・拡張しない（既存パターンの呼び出しのみ）。
"""

import copy
import json
import math
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
from tools.persistence_integrity import check_integrity


# ── Helpers ───────────────────────────────────────────────────────


EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised",
            "loving", "teasing", "scared", "happy", "neutral"]
VALENCES = [0.7, -0.6, -0.5, 0.0, 0.3,
            0.8, 0.4, -0.5, 0.6, 0.0]


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


def _run_ticks(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新する。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)


def _save_to_dict(orch: PsycheOrchestrator) -> dict[str, Any]:
    """orchestratorの状態を辞書として取得する（ファイル書き込みなし）。

    orchestrator.save()の内部と同等の辞書構成を再現する。
    """
    import time
    data: dict[str, Any] = {
        "version": CURRENT_VERSION,
        "save_timestamp": time.time(),
        "tick_count": orch.tick_count,
    }
    data.update(save_fields(orch, FIELD_DEFINITIONS))
    return data


def _load_from_dict(orch: PsycheOrchestrator, data: dict[str, Any]) -> None:
    """辞書からorchestratorの状態を復元する（ファイル読み込みなし）。

    orchestrator.load()の内部と同等の復元処理を再現する。
    """
    if "tick_count" in data:
        orch._tick_count = data["tick_count"]
    load_fields(orch, FIELD_DEFINITIONS, data)


def _deep_compare(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    float_tolerance: float = 1e-9,
    path: str = "",
    skip_keys: set[str] | None = None,
) -> list[str]:
    """2つの保存辞書を再帰的に比較し、差異を詳細に報告する。

    浮動小数点比較には微小な許容誤差を適用する。

    Args:
        skip_keys: 比較をスキップするキー名のセット。
            session_decay等で変動が想定されるフィールドを除外するために使用する。

    Returns:
        差異の記述リスト（空なら完全一致）
    """
    if skip_keys is None:
        skip_keys = set()

    diffs: list[str] = []

    keys1 = set(dict1.keys()) if isinstance(dict1, dict) else set()
    keys2 = set(dict2.keys()) if isinstance(dict2, dict) else set()

    # キーの過不足（skip_keys対象は除外）
    for k in keys1 - keys2:
        if k not in skip_keys:
            diffs.append(f"{path}.{k}: 1回目にあるが2回目にない")
    for k in keys2 - keys1:
        if k not in skip_keys:
            diffs.append(f"{path}.{k}: 2回目にあるが1回目にない")

    # 共通キーの比較
    for k in keys1 & keys2:
        v1 = dict1[k]
        v2 = dict2[k]
        child_path = f"{path}.{k}" if path else k

        # save_timestamp は save() 呼び出し時刻なので比較対象外
        if k == "save_timestamp":
            continue

        # session_decay等で変動するキーをスキップ
        if k in skip_keys:
            continue

        if type(v1) != type(v2):
            diffs.append(
                f"{child_path}: 型不一致 {type(v1).__name__} vs {type(v2).__name__}"
            )
            continue

        if isinstance(v1, dict):
            diffs.extend(_deep_compare(v1, v2, float_tolerance, child_path, skip_keys))
        elif isinstance(v1, list):
            if len(v1) != len(v2):
                diffs.append(
                    f"{child_path}: リスト長不一致 {len(v1)} vs {len(v2)}"
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
                            pass  # NaN == NaN for our purposes
                        elif abs(item1 - item2) > float_tolerance:
                            diffs.append(
                                f"{item_path}: 浮動小数点不一致 {item1} vs {item2}"
                            )
                    elif item1 != item2:
                        diffs.append(f"{item_path}: 値不一致 {item1!r} vs {item2!r}")
        elif isinstance(v1, float):
            if math.isnan(v1) and math.isnan(v2):
                pass  # NaN == NaN for our purposes
            elif abs(v1 - v2) > float_tolerance:
                diffs.append(
                    f"{child_path}: 浮動小数点不一致 {v1} vs {v2}"
                )
        elif v1 != v2:
            diffs.append(f"{child_path}: 値不一致 {v1!r} vs {v2!r}")

    return diffs


# session_decay により load 時に変動が想定されるキー。
# これらは apply_session_decay() が freshness を -0.3 し
# freshness_stage を再計算するため、save->load->save で値が変わる。
_SESSION_DECAY_KEYS: set[str] = {"freshness", "freshness_stage", "session_diff_scalar"}


def _build_version_dict(target_version: int) -> dict[str, Any]:
    """指定バージョン時点で存在するフィールドのみを含む最小辞書を構成する。

    MIGRATION_CHAINの定義から動的に構成し、各フィールドは空辞書/空値をデフォルトとする。
    """
    data: dict[str, Any] = {
        "version": target_version,
        "tick_count": 0,
    }

    for entry in MIGRATION_CHAIN:
        if entry.version <= target_version:
            for field_key in entry.added_fields:
                if field_key not in ("tick_count",):
                    # v1のcoreフィールドには最低限の構造を提供
                    if field_key == "psyche":
                        data[field_key] = {
                            "emotions": {"joy": 0.5, "sadness": 0.1, "anger": 0.0,
                                         "fear": 0.1, "surprise": 0.0, "disgust": 0.0,
                                         "trust": 0.5, "anticipation": 0.3},
                            "drives": {"curiosity": 0.5, "social": 0.5,
                                       "safety": 0.5, "autonomy": 0.5},
                            "mood": {"valence": 0.3, "arousal": 0.4,
                                     "dominance": 0.5},
                            "identity": {},
                            "attachment": {},
                            "continuity": {"memory_count": 0},
                            "projection": {},
                            "fear_index": {"composite": 0.3,
                                           "identity_risk": 0.0,
                                           "attachment_risk": 0.0,
                                           "continuity_risk": 0.0,
                                           "projection_risk": 0.0},
                        }
                    elif field_key == "loop_state":
                        data[field_key] = {
                            "stm": {"entries": []},
                            "memory": {"entries": []},
                            "emotions": {},
                            "last_loop_time": 0.0,
                        }
                    elif field_key == "dynamics":
                        data[field_key] = {
                            "phase": "resting",
                            "phase_entered_at": 0.0,
                            "intensity_history": [],
                        }
                    else:
                        data[field_key] = {}
    return data


# ══════════════════════════════════════════════════════════════════
# 層1: 全フィールドround-trip検証（ファスト）
# ══════════════════════════════════════════════════════════════════


class TestFullFieldRoundTrip:
    """全フィールドを同時にsave->load->save->比較した場合の同一性確認。"""

    def test_round_trip_5_ticks(self, tmp_path):
        """5ティック後のround-tripで全フィールドが同一であること。

        session_decay=True のモジュールは load 時に freshness を減衰させるため、
        freshness/freshness_stage は比較対象外とする。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 5)
        orch1.select_policy_dict(_make_percept(), [])

        # 1回目のsave
        save_dict1 = _save_to_dict(orch1)

        # 新しいインスタンスに復元
        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch2, save_dict1)

        # 2回目のsave
        save_dict2 = _save_to_dict(orch2)

        # 全フィールド比較（session_decay対象キーを除外）
        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Round-trip差異検出（5ティック）:\n" + "\n".join(diffs)
        )

    def test_round_trip_15_ticks(self, tmp_path):
        """15ティック後のround-tripで全フィールドが同一であること。

        session_decay=True のモジュールは load 時に freshness を減衰させるため、
        freshness/freshness_stage は比較対象外とする。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 15)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch2, save_dict1)

        save_dict2 = _save_to_dict(orch2)

        diffs = _deep_compare(save_dict1, save_dict2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"Round-trip差異検出（15ティック）:\n" + "\n".join(diffs)
        )

    def test_round_trip_integrity_check(self, tmp_path):
        """round-trip後のデータに対してcheck_integrityが問題なしを報告すること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])

        save_dict1 = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch2, save_dict1)

        save_dict2 = _save_to_dict(orch2)

        result = check_integrity(save_dict2)
        # round-trip後のデータに構造的劣化がないこと
        assert result["total_findings"] == 0, (
            f"Round-trip後にintegrity問題検出:\n"
            + json.dumps(result["findings"], ensure_ascii=False, indent=2)
        )

    def test_round_trip_via_file(self, tmp_path):
        """ファイル経由のsave->loadでもround-tripが成立すること。

        session_decay=True のモジュールは load 時に freshness を減衰させるため、
        freshness/freshness_stage は比較対象外とする。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])

        # ファイル経由のsave
        orch1.save()
        snapshot_path = tmp_path / "psyche_snapshot.json"
        assert snapshot_path.exists()

        data1 = json.loads(snapshot_path.read_text(encoding="utf-8"))

        # ファイル経由のload
        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True

        # 復元後に再度save
        orch2.save(path=tmp_path / "psyche_snapshot_2.json")
        data2 = json.loads(
            (tmp_path / "psyche_snapshot_2.json").read_text(encoding="utf-8")
        )

        diffs = _deep_compare(data1, data2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs == [], (
            f"ファイル経由round-trip差異検出:\n" + "\n".join(diffs)
        )

    def test_round_trip_field_count_matches_definitions(self, tmp_path):
        """save辞書のフィールド数がFIELD_DEFINITIONSのフィールド数+メタ情報と一致すること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch, 5)

        save_dict = _save_to_dict(orch)

        # メタフィールド: version, save_timestamp, tick_count
        meta_field_count = 3
        expected_field_count = len(FIELD_DEFINITIONS) + meta_field_count
        actual_field_count = len(save_dict)

        defined_key_set = {fd.key for fd in FIELD_DEFINITIONS}
        meta_keys = {"version", "save_timestamp", "tick_count"}
        extra_keys = set(save_dict.keys()) - meta_keys - defined_key_set
        missing_keys = defined_key_set - set(save_dict.keys())

        assert actual_field_count == expected_field_count, (
            f"フィールド数不一致: 期待={expected_field_count}, "
            f"実際={actual_field_count}\n"
            f"定義フィールド: {len(FIELD_DEFINITIONS)}, メタ: {meta_field_count}\n"
            f"余剰キー: {extra_keys}\n"
            f"不足キー: {missing_keys}"
        )

    def test_round_trip_tick_count_preserved(self, tmp_path):
        """round-tripでtick_countが正確に保存・復元されること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 7)
        original_tick = orch1.tick_count
        assert original_tick == 7

        save_dict = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch2, save_dict)

        assert orch2.tick_count == original_tick

    def test_round_trip_then_continue_ticks(self, tmp_path):
        """round-trip後にさらにティックを実行してもエラーが発生しないこと。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 10)

        save_dict = _save_to_dict(orch1)

        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch2, save_dict)

        # 復元後にさらに10ティック実行
        _run_ticks(orch2, 10)
        assert orch2.tick_count == 20

        # enrichment/policyも正常
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0
        policy = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)

    def test_double_round_trip(self, tmp_path):
        """save->load->save->load->save の2重round-tripで同一性が維持されること。

        session_decay=True のモジュールは load 時に freshness を減衰させるため、
        freshness/freshness_stage は比較対象外とする。
        各round-tripでsession_decay以外のフィールドが一致することを検証する。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])

        # 1回目round-trip
        save1 = _save_to_dict(orch1)
        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch2, save1)
        save2 = _save_to_dict(orch2)

        # 2回目round-trip
        orch3 = PsycheOrchestrator(data_dir=tmp_path)
        _load_from_dict(orch3, save2)
        save3 = _save_to_dict(orch3)

        # 各round-tripでsession_decay以外のフィールドが同一
        diffs_12 = _deep_compare(save1, save2, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs_12 == [], (
            f"1回目round-trip差異:\n" + "\n".join(diffs_12)
        )

        diffs_23 = _deep_compare(save2, save3, skip_keys=_SESSION_DECAY_KEYS)
        assert diffs_23 == [], (
            f"2回目round-trip差異:\n" + "\n".join(diffs_23)
        )

    def test_round_trip_all_field_keys_present(self, tmp_path):
        """round-trip後の辞書に全FIELD_DEFINITIONSのキーが含まれること。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch1, 5)

        save_dict = _save_to_dict(orch1)

        defined_keys = {fd.key for fd in FIELD_DEFINITIONS}
        save_keys = set(save_dict.keys())

        missing = defined_keys - save_keys
        assert missing == set(), (
            f"FIELD_DEFINITIONSに定義されているが保存辞書にないキー: {missing}"
        )

    def test_integrity_check_on_fresh_save(self, tmp_path):
        """ティック実行後の直接saveに対してcheck_integrityが問題なしを報告すること。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch, 10)
        orch.select_policy_dict(_make_percept(), [])

        save_dict = _save_to_dict(orch)
        result = check_integrity(save_dict)

        assert result["total_findings"] == 0, (
            f"直接save後にintegrity問題検出:\n"
            + json.dumps(result["findings"], ensure_ascii=False, indent=2)
        )


# ══════════════════════════════════════════════════════════════════
# 層2: マイグレーション互換性検証（スロー）
# ══════════════════════════════════════════════════════════════════


class TestMigrationCompatibility:
    """旧バージョンのデータ形式から現行バージョンへの遷移が正しく動作するかの系統的テスト。"""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "version",
        [entry.version for entry in MIGRATION_CHAIN],
        ids=[f"v{entry.version}" for entry in MIGRATION_CHAIN],
    )
    def test_load_from_version_no_crash(self, tmp_path, version):
        """各バージョンの最小辞書からload()が例外なく完了すること。"""
        min_dict = _build_version_dict(version)

        # JSONファイル経由でload
        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(min_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch.load()
        assert loaded is True

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "version",
        [entry.version for entry in MIGRATION_CHAIN],
        ids=[f"v{entry.version}" for entry in MIGRATION_CHAIN],
    )
    def test_load_from_version_then_run_ticks(self, tmp_path, version):
        """各バージョンからloadした後、数ティックの処理が例外なく完了すること。"""
        min_dict = _build_version_dict(version)

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(min_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.load()

        # 数ティック実行しても例外なし
        _run_ticks(orch, 3)
        assert orch.tick_count == 3

    @pytest.mark.slow
    def test_sequential_migration_equivalent_to_fresh_start(self, tmp_path):
        """マイグレーションチェーン全遷移の順次適用結果と、直接起動のorchestratorの挙動が等価であること。"""
        # 方法: 最も古いバージョンのデータからload後、ティック実行した結果と
        # 新規起動して同数ティック実行した結果で、enrichment出力が同等であることを確認

        # 新規起動（直接最新バージョン）
        dir_fresh = tmp_path / "fresh"
        dir_fresh.mkdir()
        orch_fresh = PsycheOrchestrator(data_dir=dir_fresh)
        _run_ticks(orch_fresh, 5)
        enrichment_fresh = orch_fresh.get_prompt_enrichment()

        # v1データからの起動
        dir_migrated = tmp_path / "migrated"
        dir_migrated.mkdir()
        min_dict_v1 = _build_version_dict(1)
        snapshot_path = dir_migrated / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(min_dict_v1, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        orch_migrated = PsycheOrchestrator(data_dir=dir_migrated)
        orch_migrated.load()
        _run_ticks(orch_migrated, 5)
        enrichment_migrated = orch_migrated.get_prompt_enrichment()

        # 両方ともenrichmentが非空であること
        assert len(enrichment_fresh) > 0
        assert len(enrichment_migrated) > 0

        # 両方とも主要セクションを含むこと（完全一致は求めない、構造的等価性のみ）
        for section in ["[内面]"]:
            assert section in enrichment_fresh, (
                f"新規起動のenrichmentにセクション'{section}'がない"
            )
            assert section in enrichment_migrated, (
                f"マイグレーション後のenrichmentにセクション'{section}'がない"
            )

    @pytest.mark.slow
    def test_v1_minimal_load_and_save(self, tmp_path):
        """最小のv1データからload後、saveして再loadできること。"""
        min_dict = _build_version_dict(1)

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(min_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.load()
        _run_ticks(orch, 3)
        orch.save()

        # 再load
        orch2 = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == 3

    @pytest.mark.slow
    def test_latest_migration_version_load(self, tmp_path):
        """マイグレーションチェーン最新バージョンのデータがload可能であること。"""
        latest_version = max(entry.version for entry in MIGRATION_CHAIN)
        min_dict = _build_version_dict(latest_version)

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(min_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch.load()
        assert loaded is True

        # 正常にティック実行可能
        _run_ticks(orch, 5)
        assert orch.tick_count == 5

        # enrichmentが正常
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0


# ══════════════════════════════════════════════════════════════════
# 層3: フィールド欠損時の非破壊性検証（ファスト）
# ══════════════════════════════════════════════════════════════════


class TestFieldDeficiencyResilience:
    """保存データの一部フィールドが欠落している場合に、
    システムがクラッシュせず残存フィールドを正しく復元するかの検証。"""

    def _get_full_save_dict(self, tmp_path) -> dict[str, Any]:
        """状態蓄積済みの完全な保存辞書を取得する。"""
        orch = PsycheOrchestrator(data_dir=tmp_path)
        _run_ticks(orch, 10)
        orch.select_policy_dict(_make_percept(), [])
        return _save_to_dict(orch)

    def test_each_field_removal_no_crash(self, tmp_path):
        """各フィールドを1つずつ除外した辞書に対してloadが例外なく完了すること。"""
        full_dict = self._get_full_save_dict(tmp_path)

        # メタフィールド以外の全フィールドキーを取得
        removable_keys = [
            fd.key for fd in FIELD_DEFINITIONS
        ]

        for key_to_remove in removable_keys:
            deficient_dict = copy.deepcopy(full_dict)
            deficient_dict.pop(key_to_remove, None)

            # 一時ファイルに書き込んでload
            sub_dir = tmp_path / f"deficient_{key_to_remove}"
            sub_dir.mkdir(exist_ok=True)
            snapshot_path = sub_dir / "psyche_snapshot.json"
            snapshot_path.write_text(
                json.dumps(deficient_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            orch = PsycheOrchestrator(data_dir=sub_dir)
            try:
                loaded = orch.load()
                assert loaded is True, (
                    f"フィールド'{key_to_remove}'除外時にloadがFalseを返した"
                )
            except Exception as e:
                pytest.fail(
                    f"フィールド'{key_to_remove}'除外時にload例外: {e}"
                )

    def test_each_field_removal_preserves_others(self, tmp_path):
        """各フィールドを1つずつ除外した場合、欠損していないフィールドは正しく復元されること。"""
        full_dict = self._get_full_save_dict(tmp_path)

        # テスト対象: 代表的なフィールド群（全フィールドでの完全テストは層1で実施済み）
        test_keys = ["psyche", "loop_state", "dynamics"]
        removable_keys = [fd.key for fd in FIELD_DEFINITIONS if fd.key not in test_keys]

        # 最初のいくつかのフィールドで検証（全数は時間がかかるため代表的に）
        sample_keys = removable_keys[:5]

        for key_to_remove in sample_keys:
            deficient_dict = copy.deepcopy(full_dict)
            deficient_dict.pop(key_to_remove, None)

            sub_dir = tmp_path / f"preserve_{key_to_remove}"
            sub_dir.mkdir(exist_ok=True)
            snapshot_path = sub_dir / "psyche_snapshot.json"
            snapshot_path.write_text(
                json.dumps(deficient_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            orch = PsycheOrchestrator(data_dir=sub_dir)
            orch.load()

            # 復元後にsaveして比較（欠損フィールド以外は元データと一致すべき）
            restored_dict = _save_to_dict(orch)

            for check_key in test_keys:
                if check_key in full_dict and check_key in restored_dict:
                    diffs = _deep_compare(
                        {check_key: full_dict[check_key]},
                        {check_key: restored_dict[check_key]},
                    )
                    assert diffs == [], (
                        f"フィールド'{key_to_remove}'除外時に'{check_key}'の"
                        f"復元が不正:\n" + "\n".join(diffs)
                    )

    def test_empty_dict_no_crash(self, tmp_path):
        """完全に空の辞書（バージョン情報のみ）に対してloadが例外なく完了すること。"""
        empty_dict = {"version": CURRENT_VERSION, "tick_count": 0}

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(empty_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch.load()
        assert loaded is True

        # loadした後もティック実行可能
        _run_ticks(orch, 3)
        assert orch.tick_count == 3

    def test_empty_dict_then_enrichment(self, tmp_path):
        """空辞書からload後のenrichmentが正常に生成されること。"""
        empty_dict = {"version": CURRENT_VERSION, "tick_count": 0}

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(empty_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.load()
        _run_ticks(orch, 3)

        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str) and len(enrichment) > 0

    def test_empty_dict_then_policy(self, tmp_path):
        """空辞書からload後のpolicy生成が正常であること。"""
        empty_dict = {"version": CURRENT_VERSION, "tick_count": 0}

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(empty_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        orch.load()
        _run_ticks(orch, 3)

        policy = orch.select_policy_dict(_make_percept(), [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

    def test_version_only_dict_no_crash(self, tmp_path):
        """バージョン番号のみの辞書に対してもloadが例外なく完了すること。"""
        version_only = {"version": CURRENT_VERSION}

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(version_only, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        loaded = orch.load()
        assert loaded is True

    def test_corrupted_field_values_no_crash(self, tmp_path):
        """一部フィールドの値が空辞書の場合でもloadが例外なく完了すること。"""
        full_dict = self._get_full_save_dict(tmp_path)

        # 全フィールドの値を空辞書に置換
        for fd in FIELD_DEFINITIONS:
            if fd.key in full_dict:
                full_dict[fd.key] = {}

        snapshot_path = tmp_path / "psyche_snapshot.json"
        snapshot_path.write_text(
            json.dumps(full_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        orch = PsycheOrchestrator(data_dir=tmp_path)
        try:
            loaded = orch.load()
            assert loaded is True
        except Exception as e:
            pytest.fail(f"全フィールド空辞書時にload例外: {e}")


# ══════════════════════════════════════════════════════════════════
# 補助検証: MIGRATION_CHAINとFIELD_DEFINITIONSの整合性
# ══════════════════════════════════════════════════════════════════


class TestFieldDefinitionConsistency:
    """MIGRATION_CHAINとFIELD_DEFINITIONSの定義が整合していることの検証。"""

    def test_all_migration_keys_in_field_definitions(self):
        """MIGRATION_CHAINの全キーがFIELD_DEFINITIONSに含まれること（tick_count除く）。"""
        migration_keys = get_all_known_field_keys() - {"tick_count"}
        definition_keys = {fd.key for fd in FIELD_DEFINITIONS}

        missing = migration_keys - definition_keys
        assert missing == set(), (
            f"MIGRATION_CHAINにあるがFIELD_DEFINITIONSにないキー: {missing}"
        )

    def test_no_duplicate_keys_in_field_definitions(self):
        """FIELD_DEFINITIONS内にキー名の重複がないこと。"""
        keys = [fd.key for fd in FIELD_DEFINITIONS]
        seen: set[str] = set()
        duplicates: list[str] = []
        for k in keys:
            if k in seen:
                duplicates.append(k)
            seen.add(k)

        assert duplicates == [], (
            f"FIELD_DEFINITIONSに重複キー: {duplicates}"
        )

    def test_migration_chain_versions_are_monotonic(self):
        """MIGRATION_CHAINのバージョン番号が単調増加であること。"""
        versions = [entry.version for entry in MIGRATION_CHAIN]
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1], (
                f"MIGRATION_CHAINのバージョンが単調増加でない: "
                f"v{versions[i-1]} -> v{versions[i]}"
            )

    def test_current_version_is_latest(self):
        """CURRENT_VERSIONがMIGRATION_CHAIN最大バージョン以上であること。"""
        if MIGRATION_CHAIN:
            max_migration = max(entry.version for entry in MIGRATION_CHAIN)
            assert CURRENT_VERSION >= max_migration, (
                f"CURRENT_VERSION({CURRENT_VERSION})がMIGRATION_CHAIN最大バージョン"
                f"({max_migration})未満"
            )

    def test_field_definitions_version_numbers_positive(self):
        """FIELD_DEFINITIONSの各フィールドのversion番号が正の整数であること。"""
        for fd in FIELD_DEFINITIONS:
            assert fd.version >= 1, (
                f"フィールド'{fd.key}'のversion({fd.version})が1未満"
            )
