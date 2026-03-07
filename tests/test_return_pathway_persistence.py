"""
tests/test_return_pathway_persistence.py - 帰還経路発火履歴永続化のテスト

design_return_pathway_persistence.md に基づく検証:
  - 永続化データの正しい保存・復元
  - inject_cumulative_counts による初期値注入
  - セッション回数カウンタ
  - ダッシュボードの拡張表示
  - 安全弁の動作確認
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tools.return_pathway_monitor import (
    ReturnPathwayMonitor,
    PATHWAY_A,
    PATHWAY_B,
    PATHWAY_C,
    PATHWAY_D,
    PATHWAY_E,
    _ALL_PATHWAYS,
)
from tools.dashboard import Dashboard, SECTION_PATHWAY
from psyche.persistence_helpers import (
    CURRENT_VERSION,
    MIGRATION_CHAIN,
    get_all_known_field_keys,
)


# ── ヘルパー ──────────────────────────────────────────────────────


def _make_monitor_with_firings(
    pathways: list[str] | None = None,
    tick: int = 1,
    fire_count_per_pathway: int = 1,
) -> ReturnPathwayMonitor:
    """テスト用に発火記録済みのモニターを生成する。"""
    monitor = ReturnPathwayMonitor(enabled=True)
    if pathways is None:
        pathways = [PATHWAY_A]
    for i in range(fire_count_per_pathway):
        for pathway_id in pathways:
            t = tick + i
            deltas: dict[str, Any] = {}
            if pathway_id in (PATHWAY_A, PATHWAY_B, PATHWAY_C):
                deltas = {"emotion_deltas": {"joy": 0.01}}
            elif pathway_id == PATHWAY_D:
                deltas = {"drive_deltas": {"curiosity": 0.005}}
            elif pathway_id == PATHWAY_E:
                deltas = {"mood_speed_deltas": {"valence_modulation": 0.002}}
            monitor.record_firing(pathway_id, t, **deltas)
        monitor.finalize_tick(tick + i)
    return monitor


# ── get_persistence_data のテスト ─────────────────────────────────


class TestGetPersistenceData:
    """get_persistence_data の読み取り専用アクセサのテスト。"""

    def test_empty_monitor_returns_zeros(self) -> None:
        """発火なしのモニターは全カウンタゼロ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        data = monitor.get_persistence_data()
        assert isinstance(data, dict)
        for pathway_id in _ALL_PATHWAYS:
            assert data["pathway_fire_counts"][pathway_id] == 0
        assert data["concurrent_2plus_count"] == 0
        assert data["concurrent_3plus_count"] == 0
        assert data["concurrent_4plus_count"] == 0
        assert data["concurrent_5_count"] == 0
        for kind in ("emotion", "drive", "mood_speed"):
            assert data["aggregate_cap_hit_counts"][kind] == 0

    def test_single_pathway_firing(self) -> None:
        """単一経路の発火が正しくカウントされる。"""
        monitor = _make_monitor_with_firings([PATHWAY_A], fire_count_per_pathway=3)
        data = monitor.get_persistence_data()
        assert data["pathway_fire_counts"][PATHWAY_A] == 3
        assert data["pathway_fire_counts"][PATHWAY_B] == 0

    def test_multiple_pathway_concurrent(self) -> None:
        """複数経路の同時発火で同時発火カウンタが増加する。"""
        monitor = _make_monitor_with_firings(
            [PATHWAY_A, PATHWAY_B, PATHWAY_C], fire_count_per_pathway=1
        )
        data = monitor.get_persistence_data()
        assert data["concurrent_2plus_count"] == 1
        assert data["concurrent_3plus_count"] == 1

    def test_cap_hit_counts_included(self) -> None:
        """合算帯域上限到達カウンタが含まれる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_aggregate_cap_hit("emotion")
        monitor.record_aggregate_cap_hit("emotion")
        monitor.record_aggregate_cap_hit("drive")
        data = monitor.get_persistence_data()
        assert data["aggregate_cap_hit_counts"]["emotion"] == 2
        assert data["aggregate_cap_hit_counts"]["drive"] == 1
        assert data["aggregate_cap_hit_counts"]["mood_speed"] == 0

    def test_data_is_copy(self) -> None:
        """返された辞書はコピーであり、変更が元に影響しない。"""
        monitor = _make_monitor_with_firings([PATHWAY_A])
        data = monitor.get_persistence_data()
        data["pathway_fire_counts"][PATHWAY_A] = 9999
        assert monitor.pathway_fire_counts[PATHWAY_A] == 1


# ── inject_cumulative_counts のテスト ─────────────────────────────


class TestInjectCumulativeCounts:
    """inject_cumulative_counts による初期値注入のテスト。"""

    def test_inject_adds_to_zero_initial(self) -> None:
        """ゼロ初期値に注入が正しく加算される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {
                PATHWAY_A: 10,
                PATHWAY_B: 5,
                PATHWAY_C: 3,
                PATHWAY_D: 2,
                PATHWAY_E: 1,
            },
            "concurrent_2plus_count": 7,
            "concurrent_3plus_count": 4,
            "concurrent_4plus_count": 2,
            "concurrent_5_count": 1,
            "aggregate_cap_hit_counts": {
                "emotion": 3,
                "drive": 1,
                "mood_speed": 0,
            },
        }
        monitor.inject_cumulative_counts(saved)
        assert monitor.pathway_fire_counts[PATHWAY_A] == 10
        assert monitor.pathway_fire_counts[PATHWAY_B] == 5
        assert monitor.concurrent_2plus_count == 7
        assert monitor.concurrent_3plus_count == 4
        assert monitor.concurrent_4plus_count == 2
        assert monitor.concurrent_5_count == 1
        assert monitor.aggregate_cap_hit_counts["emotion"] == 3

    def test_inject_then_fire_accumulates(self) -> None:
        """注入後の発火でカウンタが正しく累積する。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {PATHWAY_A: 10},
        }
        monitor.inject_cumulative_counts(saved)
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.01})
        monitor.finalize_tick(1)
        assert monitor.pathway_fire_counts[PATHWAY_A] == 11

    def test_inject_empty_dict_is_safe(self) -> None:
        """空辞書の注入は安全に処理される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.inject_cumulative_counts({})
        # 全カウンタがゼロのまま
        for pathway_id in _ALL_PATHWAYS:
            assert monitor.pathway_fire_counts[pathway_id] == 0

    def test_inject_none_is_safe(self) -> None:
        """Noneの注入は安全に処理される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.inject_cumulative_counts(None)  # type: ignore
        for pathway_id in _ALL_PATHWAYS:
            assert monitor.pathway_fire_counts[pathway_id] == 0

    def test_inject_invalid_type_is_safe(self) -> None:
        """無効な型の注入は安全に処理される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.inject_cumulative_counts("invalid")  # type: ignore
        for pathway_id in _ALL_PATHWAYS:
            assert monitor.pathway_fire_counts[pathway_id] == 0

    def test_inject_partial_data(self) -> None:
        """部分的なデータの注入は、存在するフィールドのみ適用される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {PATHWAY_A: 5},
            # concurrent counts and cap hits missing
        }
        monitor.inject_cumulative_counts(saved)
        assert monitor.pathway_fire_counts[PATHWAY_A] == 5
        assert monitor.pathway_fire_counts[PATHWAY_B] == 0
        assert monitor.concurrent_2plus_count == 0

    def test_inject_unknown_pathway_ignored(self) -> None:
        """未知の経路識別子は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {
                PATHWAY_A: 5,
                "unknown_pathway": 99,
            },
        }
        monitor.inject_cumulative_counts(saved)
        assert monitor.pathway_fire_counts[PATHWAY_A] == 5
        # unknown_pathway は _pathway_fire_counts に存在しないため加算されない

    def test_inject_negative_values_treated_as_int(self) -> None:
        """負の値も整数変換されて加算される（事実記録）。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {PATHWAY_A: -3},
        }
        monitor.inject_cumulative_counts(saved)
        assert monitor.pathway_fire_counts[PATHWAY_A] == -3

    def test_inject_float_values_truncated(self) -> None:
        """浮動小数点値は整数に切り捨てられる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {PATHWAY_A: 5.9},
            "concurrent_2plus_count": 3.7,
        }
        monitor.inject_cumulative_counts(saved)
        assert monitor.pathway_fire_counts[PATHWAY_A] == 5
        assert monitor.concurrent_2plus_count == 3

    def test_inject_string_values_ignored(self) -> None:
        """文字列の値は無視される（isinstance check for int/float）。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        saved = {
            "pathway_fire_counts": {PATHWAY_A: "not_a_number"},
            "concurrent_2plus_count": "bad",
        }
        monitor.inject_cumulative_counts(saved)
        assert monitor.pathway_fire_counts[PATHWAY_A] == 0
        assert monitor.concurrent_2plus_count == 0


# ── Round-trip テスト ─────────────────────────────────────────────


class TestRoundTrip:
    """get_persistence_data → inject_cumulative_counts のラウンドトリップ。"""

    def test_basic_round_trip(self) -> None:
        """保存→復元の一巡で値が保持される。"""
        # セッション1: 発火を蓄積
        m1 = _make_monitor_with_firings(
            [PATHWAY_A, PATHWAY_B], fire_count_per_pathway=5
        )
        m1.record_aggregate_cap_hit("emotion")
        saved = m1.get_persistence_data()

        # セッション2: 注入して継続
        m2 = ReturnPathwayMonitor(enabled=True)
        m2.inject_cumulative_counts(saved)

        # セッション2でさらに発火
        m2.record_firing(PATHWAY_A, 100, emotion_deltas={"joy": 0.01})
        m2.finalize_tick(100)

        assert m2.pathway_fire_counts[PATHWAY_A] == 6  # 5 + 1
        assert m2.pathway_fire_counts[PATHWAY_B] == 5  # 5 + 0
        assert m2.aggregate_cap_hit_counts["emotion"] == 1

    def test_multi_session_accumulation(self) -> None:
        """3セッション間で累計が正しく蓄積する。"""
        # セッション1
        m1 = _make_monitor_with_firings([PATHWAY_C], fire_count_per_pathway=3)
        d1 = m1.get_persistence_data()

        # セッション2
        m2 = ReturnPathwayMonitor(enabled=True)
        m2.inject_cumulative_counts(d1)
        m2.record_firing(PATHWAY_C, 10, emotion_deltas={"joy": 0.01})
        m2.finalize_tick(10)
        d2 = m2.get_persistence_data()

        # セッション3
        m3 = ReturnPathwayMonitor(enabled=True)
        m3.inject_cumulative_counts(d2)
        assert m3.pathway_fire_counts[PATHWAY_C] == 4  # 3 + 1

    def test_round_trip_preserves_all_counters(self) -> None:
        """全カウンタ種類がラウンドトリップで保持される。"""
        m1 = ReturnPathwayMonitor(enabled=True)
        # 5経路全て同時に発火
        for pw in [PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E]:
            if pw in (PATHWAY_A, PATHWAY_B, PATHWAY_C):
                m1.record_firing(pw, 1, emotion_deltas={"joy": 0.01})
            elif pw == PATHWAY_D:
                m1.record_firing(pw, 1, drive_deltas={"curiosity": 0.005})
            elif pw == PATHWAY_E:
                m1.record_firing(pw, 1, mood_speed_deltas={"valence_modulation": 0.002})
        m1.finalize_tick(1)
        m1.record_aggregate_cap_hit("emotion")
        m1.record_aggregate_cap_hit("drive")
        m1.record_aggregate_cap_hit("mood_speed")

        saved = m1.get_persistence_data()

        m2 = ReturnPathwayMonitor(enabled=True)
        m2.inject_cumulative_counts(saved)

        for pw in _ALL_PATHWAYS:
            assert m2.pathway_fire_counts[pw] == 1
        assert m2.concurrent_2plus_count == 1
        assert m2.concurrent_3plus_count == 1
        assert m2.concurrent_4plus_count == 1
        assert m2.concurrent_5_count == 1
        assert m2.aggregate_cap_hit_counts["emotion"] == 1
        assert m2.aggregate_cap_hit_counts["drive"] == 1
        assert m2.aggregate_cap_hit_counts["mood_speed"] == 1


# ── Orchestrator save/load統合テスト ──────────────────────────────


class TestOrchestratorIntegration:
    """orchestrator save/load における帰還経路永続化のテスト。"""

    def test_save_includes_return_pathway_history(self, tmp_path: Path) -> None:
        """save() が return_pathway_history フィールドを含む。"""
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        orch = PsycheOrchestrator()
        # 1ティック実行して帰還経路発火の可能性を作る
        percept = Percept(
            text="test", meaning="test",
            emotion="happy", intent="expression", emotion_valence=0.5,
        )
        orch.post_response_update(percept, delta_time=1.0)

        save_path = tmp_path / "test_snapshot.json"
        orch.save(save_path)

        data = json.loads(save_path.read_text(encoding="utf-8"))
        assert "return_pathway_history" in data
        rph = data["return_pathway_history"]
        assert "pathway_fire_counts" in rph
        assert "session_count" in rph
        assert rph["session_count"] == 1

    def test_save_session_count_increments(self, tmp_path: Path) -> None:
        """複数回のsave()でsession_countが増加する。"""
        from psyche.orchestrator import PsycheOrchestrator

        orch = PsycheOrchestrator()
        save_path = tmp_path / "test_snapshot.json"

        orch.save(save_path)
        d1 = json.loads(save_path.read_text(encoding="utf-8"))
        assert d1["return_pathway_history"]["session_count"] == 1

        orch.save(save_path)
        d2 = json.loads(save_path.read_text(encoding="utf-8"))
        assert d2["return_pathway_history"]["session_count"] == 2

    def test_load_restores_return_pathway_history(self, tmp_path: Path) -> None:
        """load() が return_pathway_history を帰還経路モニターに注入する。"""
        from psyche.orchestrator import PsycheOrchestrator

        save_path = tmp_path / "test_snapshot.json"

        # セッション1: 保存（モニターを直接有効化して発火記録）
        orch1 = PsycheOrchestrator()
        orch1._return_pathway_monitor._enabled = True
        rpm1 = orch1._return_pathway_monitor
        rpm1.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.01})
        rpm1.finalize_tick(1)
        orch1.save(save_path)

        # セッション2: 復元
        orch2 = PsycheOrchestrator()
        assert orch2._return_pathway_monitor.pathway_fire_counts[PATHWAY_A] == 0
        orch2.load(save_path)
        assert orch2._return_pathway_monitor.pathway_fire_counts[PATHWAY_A] == 1
        assert orch2._return_pathway_session_count == 1

    def test_load_old_version_without_field(self, tmp_path: Path) -> None:
        """return_pathway_history フィールドがない旧バージョンからの読込。"""
        from psyche.orchestrator import PsycheOrchestrator

        save_path = tmp_path / "test_snapshot.json"

        # 旧フォーマットのデータ（return_pathway_history なし）
        orch_temp = PsycheOrchestrator()
        orch_temp.save(save_path)

        # return_pathway_history を削除して旧バージョンを模倣
        data = json.loads(save_path.read_text(encoding="utf-8"))
        data.pop("return_pathway_history", None)
        data["version"] = 44
        save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 読込: エラーなし、カウンタはゼロのまま
        orch = PsycheOrchestrator()
        result = orch.load(save_path)
        assert result is True
        assert orch._return_pathway_monitor.pathway_fire_counts[PATHWAY_A] == 0
        assert orch._return_pathway_session_count == 0

    def test_accumulation_across_save_load(self, tmp_path: Path) -> None:
        """save→load→fire→save→load で累計が正しく蓄積する。"""
        from psyche.orchestrator import PsycheOrchestrator

        save_path = tmp_path / "test_snapshot.json"

        # セッション1: 3回発火して保存
        orch1 = PsycheOrchestrator()
        orch1._return_pathway_monitor._enabled = True
        for i in range(3):
            orch1._return_pathway_monitor.record_firing(
                PATHWAY_B, i, emotion_deltas={"joy": 0.01}
            )
            orch1._return_pathway_monitor.finalize_tick(i)
        orch1.save(save_path)

        # セッション2: 復元→2回発火→保存
        orch2 = PsycheOrchestrator()
        orch2.load(save_path)
        orch2._return_pathway_monitor._enabled = True
        assert orch2._return_pathway_monitor.pathway_fire_counts[PATHWAY_B] == 3
        for i in range(2):
            orch2._return_pathway_monitor.record_firing(
                PATHWAY_B, 100 + i, emotion_deltas={"joy": 0.01}
            )
            orch2._return_pathway_monitor.finalize_tick(100 + i)
        orch2.save(save_path)

        # セッション3: 復元して確認
        orch3 = PsycheOrchestrator()
        orch3.load(save_path)
        assert orch3._return_pathway_monitor.pathway_fire_counts[PATHWAY_B] == 5
        assert orch3._return_pathway_session_count == 2


# ── バージョン管理テスト ──────────────────────────────────────────


class TestVersionManagement:
    """永続化バージョンとマイグレーションチェーンのテスト。"""

    def test_current_version_is_45(self) -> None:
        """CURRENT_VERSION が 45 であること。"""
        assert CURRENT_VERSION == 45

    def test_migration_chain_includes_v45(self) -> None:
        """マイグレーションチェーンにバージョン45が含まれる。"""
        versions = [entry.version for entry in MIGRATION_CHAIN]
        assert 45 in versions

    def test_v45_field_is_return_pathway_history(self) -> None:
        """バージョン45で追加されるフィールドは return_pathway_history。"""
        for entry in MIGRATION_CHAIN:
            if entry.version == 45:
                assert "return_pathway_history" in entry.added_fields
                break
        else:
            pytest.fail("Version 45 not found in migration chain")

    def test_return_pathway_history_in_all_known_keys(self) -> None:
        """return_pathway_history が全既知キーセットに含まれる。"""
        all_keys = get_all_known_field_keys()
        assert "return_pathway_history" in all_keys


# ── ダッシュボードテスト ──────────────────────────────────────────


class TestDashboardPathwaySection:
    """dashboard.py の帰還経路セクション拡張のテスト。"""

    def test_pathway_section_with_monitor(self) -> None:
        """帰還経路セクションが正しく表示される。"""
        monitor = _make_monitor_with_firings(
            [PATHWAY_A, PATHWAY_B], fire_count_per_pathway=3
        )
        dashboard = Dashboard(return_pathway_monitor=monitor)
        data = dashboard.collect([SECTION_PATHWAY])

        assert SECTION_PATHWAY in data
        pathway_data = data[SECTION_PATHWAY]
        assert "pathway_fire_counts" in pathway_data
        assert pathway_data["pathway_fire_counts"][PATHWAY_A] == 3

    def test_pathway_section_text_format(self) -> None:
        """テキスト形式に全カウンタ段階が含まれる。"""
        monitor = _make_monitor_with_firings(
            [PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E],
            fire_count_per_pathway=1,
        )
        monitor.record_aggregate_cap_hit("emotion")
        dashboard = Dashboard(return_pathway_monitor=monitor)
        text = dashboard.format_text([SECTION_PATHWAY])

        assert "Return Pathways" in text
        assert "Concurrent (2+):" in text
        assert "Concurrent (3+):" in text
        assert "Concurrent (4+):" in text
        assert "Concurrent (5):" in text
        assert "Cap hits:" in text
        assert "emotion:" in text

    def test_pathway_section_includes_persistence_data(self) -> None:
        """collect() が persistence_data を含む。"""
        monitor = _make_monitor_with_firings([PATHWAY_A])
        dashboard = Dashboard(return_pathway_monitor=monitor)
        data = dashboard.collect([SECTION_PATHWAY])
        pathway_data = data[SECTION_PATHWAY]
        assert "persistence_data" in pathway_data

    def test_pathway_section_not_connected(self) -> None:
        """モニター未接続時は not_connected。"""
        dashboard = Dashboard()
        data = dashboard.collect([SECTION_PATHWAY])
        assert data[SECTION_PATHWAY] == {"status": "not_connected"}

    def test_pathway_section_json_format(self) -> None:
        """JSON形式でも正しく出力される。"""
        monitor = _make_monitor_with_firings([PATHWAY_A])
        dashboard = Dashboard(return_pathway_monitor=monitor)
        json_text = dashboard.format_json([SECTION_PATHWAY])
        parsed = json.loads(json_text)
        assert SECTION_PATHWAY in parsed
        assert "pathway_fire_counts" in parsed[SECTION_PATHWAY]


# ── 安全弁テスト ──────────────────────────────────────────────────


class TestSafetyValves:
    """設計書に記載された安全弁の検証。"""

    def test_data_size_is_fixed(self) -> None:
        """安全弁1: データサイズが帰還経路数に比例する固定量。"""
        monitor = _make_monitor_with_firings(
            list(_ALL_PATHWAYS), fire_count_per_pathway=100
        )
        data = monitor.get_persistence_data()
        # フィールド数は固定
        assert len(data["pathway_fire_counts"]) == 5
        assert len(data["aggregate_cap_hit_counts"]) == 3
        # 整数値のみ
        for v in data["pathway_fire_counts"].values():
            assert isinstance(v, int)
        assert isinstance(data["concurrent_2plus_count"], int)
        assert isinstance(data["concurrent_3plus_count"], int)
        assert isinstance(data["concurrent_4plus_count"], int)
        assert isinstance(data["concurrent_5_count"], int)
        for v in data["aggregate_cap_hit_counts"].values():
            assert isinstance(v, int)

    def test_default_value_on_missing_field(self) -> None:
        """安全弁2: フィールド欠損時のデフォルト値保証。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.inject_cumulative_counts({
            # pathway_fire_counts だけ存在、他は欠損
            "pathway_fire_counts": {PATHWAY_A: 5},
        })
        assert monitor.pathway_fire_counts[PATHWAY_A] == 5
        assert monitor.concurrent_2plus_count == 0
        assert monitor.aggregate_cap_hit_counts["emotion"] == 0

    def test_existing_safety_valves_preserved(self) -> None:
        """安全弁3: 既存の安全弁が維持されている。"""
        # disabled monitor does not record
        monitor = ReturnPathwayMonitor(enabled=False)
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.01})
        assert monitor.pathway_fire_counts[PATHWAY_A] == 0

    def test_no_evaluative_vocabulary_in_text(self) -> None:
        """安全弁4: 可視化部分に評価的語彙がない。"""
        monitor = _make_monitor_with_firings(
            list(_ALL_PATHWAYS), fire_count_per_pathway=5
        )
        dashboard = Dashboard(return_pathway_monitor=monitor)
        text = dashboard.format_text([SECTION_PATHWAY])
        # 評価的形容詞が含まれないことを確認
        evaluative_words = ["excessive", "insufficient", "abnormal", "normal",
                           "過剰", "不足", "異常", "正常"]
        for word in evaluative_words:
            assert word not in text.lower()

    def test_no_reset_method_exists(self) -> None:
        """安全弁5: カウンタのリセット・修正・削除メソッドが存在しない。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # リセット系メソッドが存在しないことを確認
        assert not hasattr(monitor, "reset_counts")
        assert not hasattr(monitor, "clear_counts")
        assert not hasattr(monitor, "delete_history")
        assert not hasattr(monitor, "reset_persistence")


# ── 構造的分離テスト ──────────────────────────────────────────────


class TestStructuralSeparation:
    """永続化データがpsycheの処理パイプラインに流れないことの検証。"""

    def test_persistence_data_not_in_enrichment(self) -> None:
        """永続化データがenrichmentに含まれない。"""
        from psyche.orchestrator import PsycheOrchestrator

        orch = PsycheOrchestrator()
        # 帰還経路に発火を記録
        orch._return_pathway_monitor.record_firing(
            PATHWAY_A, 1, emotion_deltas={"joy": 0.01}
        )
        orch._return_pathway_monitor.finalize_tick(1)

        enrichment = orch.get_prompt_enrichment()
        enrichment_str = str(enrichment)
        assert "return_pathway_history" not in enrichment_str
        assert "return_pathway_persistence" not in enrichment_str

    def test_persistence_data_does_not_affect_monitor_behavior(self) -> None:
        """永続化されたカウンタ値が帰還経路の動作に影響しない。"""
        # 大量の累積値を注入
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.inject_cumulative_counts({
            "pathway_fire_counts": {PATHWAY_A: 100000},
            "concurrent_2plus_count": 50000,
        })

        # 発火記録の動作は通常と同じ
        monitor.record_firing(PATHWAY_A, 1, emotion_deltas={"joy": 0.01})
        monitor.finalize_tick(1)

        # 累積値が加算されるだけで、動作に変化なし
        assert monitor.pathway_fire_counts[PATHWAY_A] == 100001
        last = monitor.last_tick_record
        assert last is not None
        assert last["fire_count"] == 1  # ティック内発火数は1


# ── JSON直列化テスト ──────────────────────────────────────────────


class TestJsonSerialization:
    """永続化データのJSON直列化テスト。"""

    def test_persistence_data_is_json_serializable(self) -> None:
        """get_persistence_data の返り値がJSON直列化可能。"""
        monitor = _make_monitor_with_firings(
            list(_ALL_PATHWAYS), fire_count_per_pathway=5
        )
        monitor.record_aggregate_cap_hit("emotion")
        data = monitor.get_persistence_data()
        # JSON直列化が例外を起こさない
        text = json.dumps(data, ensure_ascii=False)
        # デシリアライズして元に戻る
        restored = json.loads(text)
        assert restored["pathway_fire_counts"][PATHWAY_A] == 5
        # 5経路が毎ティック同時発火するので concurrent_2plus_count == 5
        assert restored["concurrent_2plus_count"] == 5
        assert restored["aggregate_cap_hit_counts"]["emotion"] == 1
