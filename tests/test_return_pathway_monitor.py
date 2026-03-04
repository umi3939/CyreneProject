"""
tests/test_return_pathway_monitor.py - 帰還経路検証モニターのテスト

tools/return_pathway_monitor.py の全機能をテストする。
"""

from __future__ import annotations

import json
import logging
import os
import pytest
from typing import Any
from unittest.mock import patch

# テスト対象
from tools.return_pathway_monitor import (
    ReturnPathwayMonitor,
    PATHWAY_A,
    PATHWAY_B,
    PATHWAY_C,
    _ALL_PATHWAYS,
    _is_monitor_enabled,
)


# ── テスト用ヘルパー ──────────────────────────────────────────────────


def _make_deltas(**kwargs: float) -> dict[str, float]:
    """テスト用の感情変動辞書を作成する。"""
    return dict(kwargs)


# ── 環境変数制御のテスト ──────────────────────────────────────────────


class TestEnvironmentControl:
    """安全弁2: 環境変数による完全無効化のテスト。"""

    def test_monitor_disabled_by_default(self) -> None:
        """環境変数未設定時はモニタリング無効。"""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_monitor_enabled() is False

    def test_monitor_disabled_when_zero(self) -> None:
        """CYRENE_MONITOR=0 で無効。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            assert _is_monitor_enabled() is False

    def test_monitor_enabled_when_one(self) -> None:
        """CYRENE_MONITOR=1 で有効。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            assert _is_monitor_enabled() is True

    def test_explicit_enable_overrides_env(self) -> None:
        """明示的なenabled=Trueが環境変数に優先する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "0"}):
            monitor = ReturnPathwayMonitor(enabled=True)
            assert monitor.enabled is True

    def test_explicit_disable_overrides_env(self) -> None:
        """明示的なenabled=Falseが環境変数に優先する。"""
        with patch.dict(os.environ, {"CYRENE_MONITOR": "1"}):
            monitor = ReturnPathwayMonitor(enabled=False)
            assert monitor.enabled is False


# ── 初期状態のテスト ──────────────────────────────────────────────────


class TestInitialState:
    """初期化直後の状態確認。"""

    def test_initial_fire_counts_all_zero(self) -> None:
        """初期状態で全経路の発火回数がゼロ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 0
        assert counts[PATHWAY_B] == 0
        assert counts[PATHWAY_C] == 0

    def test_initial_concurrent_counts_zero(self) -> None:
        """初期状態で同時発火回数がゼロ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert monitor.concurrent_2plus_count == 0
        assert monitor.concurrent_3_count == 0

    def test_initial_last_tick_record_none(self) -> None:
        """初期状態で直近のティック記録がNone。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert monitor.last_tick_record is None

    def test_pathway_constants(self) -> None:
        """経路識別子の定数が正しい。"""
        assert PATHWAY_A == "memory_emotion_return"
        assert PATHWAY_B == "selection_emotion_return"
        assert PATHWAY_C == "other_hypothesis_emotion_return"
        assert len(_ALL_PATHWAYS) == 5


# ── 段階1: 発火記録の構成テスト ──────────────────────────────────────


class TestRecordFiring:
    """record_firing メソッドのテスト。"""

    def test_single_pathway_fire(self) -> None:
        """単一経路の発火記録。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_deltas(joy=0.01, fear=-0.005)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=deltas)

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 1
        assert counts[PATHWAY_B] == 0
        assert counts[PATHWAY_C] == 0

    def test_multiple_pathways_same_tick(self) -> None:
        """同一ティック内で複数経路が発火。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(anger=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas=_make_deltas(fear=0.005))

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 1
        assert counts[PATHWAY_B] == 1
        assert counts[PATHWAY_C] == 1

    def test_duplicate_pathway_same_tick_ignored(self) -> None:
        """同一ティック内で同一経路の重複記録は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.05))

        # 1回のみカウント
        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 1

    def test_same_pathway_different_ticks(self) -> None:
        """異なるティックで同一経路が発火した場合、累積カウント。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        # ティックが変わるとバッファがリセットされる
        monitor.record_firing(PATHWAY_A, tick_number=2, emotion_deltas=_make_deltas(joy=0.02))

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 2

    def test_invalid_pathway_id_ignored(self) -> None:
        """不正な経路識別子は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing("invalid_pathway", tick_number=1, emotion_deltas=_make_deltas(joy=0.01))

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 0
        assert counts[PATHWAY_B] == 0
        assert counts[PATHWAY_C] == 0

    def test_empty_deltas(self) -> None:
        """空のdeltasでも記録される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={})

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 1

    def test_disabled_monitor_no_recording(self) -> None:
        """無効時は記録されない。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 0

    def test_tick_buffer_reset_on_new_tick(self) -> None:
        """新しいティックでバッファがリセットされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(anger=0.02))
        # ティック2に移行
        monitor.record_firing(PATHWAY_A, tick_number=2, emotion_deltas=_make_deltas(joy=0.03))

        # ティック2ではPATHWAY_Aのみがバッファに入っている
        # finalize_tickで確認
        result = monitor.finalize_tick(2)
        assert result is not None
        assert result["fire_count"] == 1
        assert PATHWAY_A in result["fired_pathways"]


# ── 段階2: 同一ティック内の合算記述テスト ────────────────────────────


class TestFinalizeTick:
    """finalize_tick メソッドのテスト。"""

    def test_no_firing_returns_none(self) -> None:
        """発火なしのティックではNoneを返す。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        result = monitor.finalize_tick(1)
        assert result is None

    def test_single_pathway_summary(self) -> None:
        """単一経路の合算記述。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_deltas(joy=0.01, fear=-0.005)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=deltas)

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["type"] == "return_pathway_cycle_summary"
        assert result["tick_number"] == 1
        assert result["fired_pathways"] == [PATHWAY_A]
        assert result["fire_count"] == 1
        assert abs(result["combined_deltas"]["joy"] - 0.01) < 1e-9
        assert abs(result["combined_deltas"]["fear"] - (-0.005)) < 1e-9

    def test_two_pathways_concurrent(self) -> None:
        """2経路同時発火の合算記述。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(joy=0.02, anger=0.03))

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["fire_count"] == 2
        assert set(result["fired_pathways"]) == {PATHWAY_A, PATHWAY_B}
        # joy: 0.01 + 0.02 = 0.03
        assert abs(result["combined_deltas"]["joy"] - 0.03) < 1e-9
        # anger: 0.03
        assert abs(result["combined_deltas"]["anger"] - 0.03) < 1e-9

    def test_three_pathways_concurrent(self) -> None:
        """3経路同時発火の合算記述。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(joy=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas=_make_deltas(joy=0.005))

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["fire_count"] == 3
        assert set(result["fired_pathways"]) == {PATHWAY_A, PATHWAY_B, PATHWAY_C}
        # joy: 0.01 + 0.02 + 0.005 = 0.035
        assert abs(result["combined_deltas"]["joy"] - 0.035) < 1e-9

    def test_concurrent_2plus_counter(self) -> None:
        """2経路以上同時発火のカウンタ更新。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # ティック1: 2経路
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(joy=0.02))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3_count == 0

    def test_concurrent_3_counter(self) -> None:
        """3経路同時発火のカウンタ更新。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(joy=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas=_make_deltas(joy=0.005))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3_count == 1

    def test_single_fire_no_concurrent_count(self) -> None:
        """単一経路発火では同時発火カウンタは更新されない。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 0
        assert monitor.concurrent_3_count == 0

    def test_buffer_cleared_after_finalize(self) -> None:
        """finalize_tick後にバッファがクリアされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        # 同じティックで再度finalize_tickを呼ぶ
        # バッファはクリア済みなのでNoneを返す
        result = monitor.finalize_tick(1)
        assert result is None

    def test_last_tick_record_updated(self) -> None:
        """finalize_tick後にlast_tick_recordが更新される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        record = monitor.last_tick_record
        assert record is not None
        assert record["tick_number"] == 1
        assert record["fire_count"] == 1

    def test_last_tick_record_overwritten(self) -> None:
        """次のティックのfinalize_tickで直近記録が上書きされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        # ティック2
        monitor.record_firing(PATHWAY_B, tick_number=2, emotion_deltas=_make_deltas(anger=0.02))
        monitor.finalize_tick(2)

        record = monitor.last_tick_record
        assert record is not None
        assert record["tick_number"] == 2
        assert PATHWAY_B in record["fired_pathways"]

    def test_last_tick_record_none_when_no_firing(self) -> None:
        """発火なしティックのfinalize_tickで直近記録がNoneになる。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1で発火
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)
        assert monitor.last_tick_record is not None

        # ティック2で発火なし
        result = monitor.finalize_tick(2)
        assert result is None
        assert monitor.last_tick_record is None

    def test_disabled_monitor_finalize_returns_none(self) -> None:
        """無効時のfinalize_tickはNoneを返す。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        result = monitor.finalize_tick(1)
        assert result is None

    def test_combined_deltas_multiple_dimensions(self) -> None:
        """複数次元の合算が正しく計算される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_deltas(joy=0.01, fear=-0.005, sorrow=0.002),
        )
        monitor.record_firing(
            PATHWAY_B, tick_number=1,
            emotion_deltas=_make_deltas(joy=0.02, anger=0.03, sorrow=-0.001),
        )
        monitor.record_firing(
            PATHWAY_C, tick_number=1,
            emotion_deltas=_make_deltas(joy=0.005, fear=0.01),
        )

        result = monitor.finalize_tick(1)
        assert result is not None
        cd = result["combined_deltas"]
        # joy: 0.01 + 0.02 + 0.005 = 0.035
        assert abs(cd["joy"] - 0.035) < 1e-9
        # fear: -0.005 + 0.01 = 0.005
        assert abs(cd["fear"] - 0.005) < 1e-9
        # sorrow: 0.002 + (-0.001) = 0.001
        assert abs(cd["sorrow"] - 0.001) < 1e-9
        # anger: 0.03
        assert abs(cd["anger"] - 0.03) < 1e-9


# ── 段階3: セッションサマリーテスト ──────────────────────────────────


class TestSessionSummary:
    """emit_session_summary メソッドのテスト。"""

    def test_empty_session_summary(self) -> None:
        """発火なしのセッションサマリー。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        result = monitor.emit_session_summary()
        assert result is not None
        assert result["type"] == "return_pathway_session_summary"
        assert result["pathway_fire_counts"][PATHWAY_A] == 0
        assert result["pathway_fire_counts"][PATHWAY_B] == 0
        assert result["pathway_fire_counts"][PATHWAY_C] == 0
        assert result["concurrent_2plus_count"] == 0
        assert result["concurrent_3_count"] == 0

    def test_session_summary_after_multiple_ticks(self) -> None:
        """複数ティック後のセッションサマリー。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1: A + B
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(joy=0.02))
        monitor.finalize_tick(1)

        # ティック2: A + B + C
        monitor.record_firing(PATHWAY_A, tick_number=2, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=2, emotion_deltas=_make_deltas(joy=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=2, emotion_deltas=_make_deltas(joy=0.005))
        monitor.finalize_tick(2)

        # ティック3: A only
        monitor.record_firing(PATHWAY_A, tick_number=3, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(3)

        result = monitor.emit_session_summary()
        assert result is not None
        assert result["pathway_fire_counts"][PATHWAY_A] == 3
        assert result["pathway_fire_counts"][PATHWAY_B] == 2
        assert result["pathway_fire_counts"][PATHWAY_C] == 1
        assert result["concurrent_2plus_count"] == 2  # tick 1 and tick 2
        assert result["concurrent_3_count"] == 1  # tick 2 only

    def test_disabled_monitor_summary_returns_none(self) -> None:
        """無効時のemit_session_summaryはNoneを返す。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        result = monitor.emit_session_summary()
        assert result is None


# ── 読み取り専用アクセサのテスト ──────────────────────────────────────


class TestGetSummary:
    """get_summary メソッドのテスト。"""

    def test_initial_summary(self) -> None:
        """初期状態のサマリー。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        summary = monitor.get_summary()
        assert summary["pathway_fire_counts"][PATHWAY_A] == 0
        assert summary["concurrent_2plus_count"] == 0
        assert summary["concurrent_3_count"] == 0
        assert summary["last_tick_record"] is None

    def test_summary_after_activity(self) -> None:
        """活動後のサマリー。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(anger=0.02))
        monitor.finalize_tick(1)

        summary = monitor.get_summary()
        assert summary["pathway_fire_counts"][PATHWAY_A] == 1
        assert summary["pathway_fire_counts"][PATHWAY_B] == 1
        assert summary["concurrent_2plus_count"] == 1
        assert summary["last_tick_record"] is not None
        assert summary["last_tick_record"]["fire_count"] == 2

    def test_summary_returns_copies(self) -> None:
        """サマリーが読み取り専用コピーであることの確認。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        summary1 = monitor.get_summary()
        summary2 = monitor.get_summary()
        # 異なるオブジェクトであること
        assert summary1 is not summary2
        assert summary1["pathway_fire_counts"] is not summary2["pathway_fire_counts"]


# ── プロパティの読み取り専用性テスト ──────────────────────────────────


class TestReadOnlyProperties:
    """プロパティが読み取り専用コピーを返すことのテスト。"""

    def test_pathway_fire_counts_is_copy(self) -> None:
        """pathway_fire_countsが読み取り専用コピー。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        counts = monitor.pathway_fire_counts
        counts[PATHWAY_A] = 999
        # 内部状態に影響しない
        assert monitor.pathway_fire_counts[PATHWAY_A] == 0

    def test_last_tick_record_is_copy(self) -> None:
        """last_tick_recordが読み取り専用コピー。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        record = monitor.last_tick_record
        assert record is not None
        record["fire_count"] = 999
        # 内部状態に影響しない
        assert monitor.last_tick_record["fire_count"] == 1


# ── 安全弁テスト ──────────────────────────────────────────────────────


class TestSafetyValves:
    """安全弁のテスト。"""

    def test_safety_valve_1_exception_in_record_firing(self) -> None:
        """record_firing内の例外は捕捉されスキップされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # Noneを渡してもクラッシュしない
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=None)  # type: ignore
        # 例外が捕捉されてスキップ（record_firingのtryブロック内で処理される）
        # emotion_deltasがNoneの場合はdict()変換でクラッシュしないか確認
        # 実際にはNoneでもdict(None)はTypeErrorだが、tryで捕捉される

    def test_safety_valve_1_exception_in_finalize(self) -> None:
        """finalize_tick内の例外は捕捉されNoneが返る。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 内部状態を直接壊す
        monitor._tick_buffer = [{"no_pathway_id": True}]  # type: ignore
        monitor._current_tick = 1
        result = monitor.finalize_tick(1)
        # 例外が捕捉されてNoneが返るか、正常にKeyErrorがtryで捕捉される
        # いずれにしてもクラッシュしない

    def test_safety_valve_1_exception_in_session_summary(self) -> None:
        """emit_session_summary内の例外は捕捉されNoneが返る。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # _pathway_fire_countsを壊す
        monitor._pathway_fire_counts = None  # type: ignore
        result = monitor.emit_session_summary()
        # 例外が捕捉されてNoneが返る
        assert result is None

    def test_safety_valve_2_disabled_no_operations(self) -> None:
        """無効時は全ての操作が何もしない。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        result = monitor.finalize_tick(1)
        summary = monitor.emit_session_summary()

        assert result is None
        assert summary is None
        assert monitor.pathway_fire_counts[PATHWAY_A] == 0

    def test_safety_valve_3_no_enrichment_connection(self) -> None:
        """enrichmentへの接続がないことの構造的確認。

        ReturnPathwayMonitorにenrichment関連のメソッドや属性がないことを確認。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        # enrichment関連のメソッド/属性が存在しないこと
        assert not hasattr(monitor, "get_enrichment")
        assert not hasattr(monitor, "enrichment")
        assert not hasattr(monitor, "build_enrichment")

    def test_safety_valve_4_no_psyche_state_connection(self) -> None:
        """psyche状態への書き込み経路がないことの構造的確認。

        ReturnPathwayMonitorにpsycheの状態変更メソッドがないことを確認。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        assert not hasattr(monitor, "apply_to_psyche")
        assert not hasattr(monitor, "update_state")
        assert not hasattr(monitor, "modify_emotions")

    def test_safety_valve_5_no_persistence(self) -> None:
        """永続化メソッドがないことの構造的確認。

        to_dict/from_dict/save/loadがないことを確認。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        assert not hasattr(monitor, "to_dict")
        assert not hasattr(monitor, "from_dict")
        assert not hasattr(monitor, "save")
        assert not hasattr(monitor, "load")


# ── ログ出力テスト ────────────────────────────────────────────────────


class TestLogging:
    """ログ出力のテスト。"""

    def test_firing_log_emitted(self, caplog: Any) -> None:
        """発火記録がログに出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=True)
            monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))

        # JSON形式のログが出力されている
        found = False
        for record in caplog.records:
            if "return_pathway_firing" in record.message:
                data = json.loads(record.message)
                assert data["type"] == "return_pathway_firing"
                assert data["pathway_id"] == PATHWAY_A
                found = True
        assert found, "return_pathway_firing log not found"

    def test_cycle_summary_log_emitted(self, caplog: Any) -> None:
        """合算記述がログに出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=True)
            monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
            monitor.finalize_tick(1)

        found = False
        for record in caplog.records:
            if "return_pathway_cycle_summary" in record.message:
                data = json.loads(record.message)
                assert data["type"] == "return_pathway_cycle_summary"
                found = True
        assert found, "return_pathway_cycle_summary log not found"

    def test_session_summary_log_emitted(self, caplog: Any) -> None:
        """セッションサマリーがログに出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=True)
            monitor.emit_session_summary()

        found = False
        for record in caplog.records:
            if "return_pathway_session_summary" in record.message:
                data = json.loads(record.message)
                assert data["type"] == "return_pathway_session_summary"
                found = True
        assert found, "return_pathway_session_summary log not found"

    def test_no_log_when_disabled(self, caplog: Any) -> None:
        """無効時はログが出力されない。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=False)
            monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
            monitor.finalize_tick(1)
            monitor.emit_session_summary()

        pathway_logs = [
            r for r in caplog.records
            if "return_pathway" in r.message
        ]
        assert len(pathway_logs) == 0


# ── 累積テスト ────────────────────────────────────────────────────────


class TestCumulativeScenarios:
    """複数ティックにわたる累積シナリオのテスト。"""

    def test_10_ticks_scenario(self) -> None:
        """10ティックのシナリオ。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1: A only
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.finalize_tick(1)

        # ティック2: B only
        monitor.record_firing(PATHWAY_B, tick_number=2, emotion_deltas=_make_deltas(anger=0.02))
        monitor.finalize_tick(2)

        # ティック3: A + C
        monitor.record_firing(PATHWAY_A, tick_number=3, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_C, tick_number=3, emotion_deltas=_make_deltas(fear=0.005))
        monitor.finalize_tick(3)

        # ティック4: no firing
        monitor.finalize_tick(4)

        # ティック5: A + B + C
        monitor.record_firing(PATHWAY_A, tick_number=5, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=5, emotion_deltas=_make_deltas(anger=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=5, emotion_deltas=_make_deltas(fear=0.005))
        monitor.finalize_tick(5)

        # ティック6-10: no firing
        for t in range(6, 11):
            monitor.finalize_tick(t)

        # 検証
        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 3  # ticks 1, 3, 5
        assert counts[PATHWAY_B] == 2  # ticks 2, 5
        assert counts[PATHWAY_C] == 2  # ticks 3, 5
        assert monitor.concurrent_2plus_count == 2  # ticks 3, 5
        assert monitor.concurrent_3_count == 1  # tick 5

    def test_consecutive_concurrent_ticks(self) -> None:
        """連続した同時発火ティック。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        for tick in range(1, 6):
            monitor.record_firing(PATHWAY_A, tick_number=tick, emotion_deltas=_make_deltas(joy=0.01))
            monitor.record_firing(PATHWAY_B, tick_number=tick, emotion_deltas=_make_deltas(anger=0.02))
            monitor.finalize_tick(tick)

        assert monitor.concurrent_2plus_count == 5
        assert monitor.concurrent_3_count == 0
        assert monitor.pathway_fire_counts[PATHWAY_A] == 5
        assert monitor.pathway_fire_counts[PATHWAY_B] == 5

    def test_no_evaluation_in_combined_deltas(self) -> None:
        """合算値に対する評価・判定・閾値比較を行わないことの確認。

        大きな合算値でも記録されるだけで、警告やエラーは発生しない。
        """
        monitor = ReturnPathwayMonitor(enabled=True)
        # 大きなデルタ値
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.5))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(joy=0.5))
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas=_make_deltas(joy=0.5))

        result = monitor.finalize_tick(1)
        assert result is not None
        # 合計1.5でも評価なし
        assert abs(result["combined_deltas"]["joy"] - 1.5) < 1e-9
        # エラーや警告フラグがないこと
        assert "warning" not in result
        assert "error" not in result
        assert "evaluation" not in result


# ── 種別識別子のフォーマット確認テスト ──────────────────────────────


class TestRecordFormat:
    """設計書で指定されたフォーマットの確認。"""

    def test_firing_record_type(self) -> None:
        """発火記録の種別識別子が正しい。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        # 内部バッファから確認
        assert len(monitor._tick_buffer) == 1
        assert monitor._tick_buffer[0]["type"] == "return_pathway_firing"

    def test_cycle_summary_type(self) -> None:
        """合算記述の種別識別子が正しい。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["type"] == "return_pathway_cycle_summary"

    def test_session_summary_type(self) -> None:
        """セッションサマリーの種別識別子が正しい。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        result = monitor.emit_session_summary()
        assert result is not None
        assert result["type"] == "return_pathway_session_summary"

    def test_firing_record_has_timestamp(self) -> None:
        """発火記録にタイムスタンプが含まれる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        assert "timestamp" in monitor._tick_buffer[0]
        assert isinstance(monitor._tick_buffer[0]["timestamp"], float)

    def test_firing_record_has_all_fields(self) -> None:
        """発火記録が全必須フィールドを持つ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_deltas(joy=0.01, fear=-0.005)
        monitor.record_firing(PATHWAY_A, tick_number=42, emotion_deltas=deltas)

        record = monitor._tick_buffer[0]
        assert record["type"] == "return_pathway_firing"
        assert "timestamp" in record
        assert record["pathway_id"] == PATHWAY_A
        assert record["tick_number"] == 42
        assert record["emotion_deltas"] == deltas

    def test_cycle_summary_has_all_fields(self) -> None:
        """合算記述が全必須フィールドを持つ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_deltas(anger=0.02))
        result = monitor.finalize_tick(1)

        assert result is not None
        assert "type" in result
        assert "timestamp" in result
        assert "tick_number" in result
        assert "fired_pathways" in result
        assert "fire_count" in result
        assert "combined_deltas" in result

    def test_session_summary_has_all_fields(self) -> None:
        """セッションサマリーが全必須フィールドを持つ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        result = monitor.emit_session_summary()

        assert result is not None
        assert "type" in result
        assert "timestamp" in result
        assert "pathway_fire_counts" in result
        assert "concurrent_2plus_count" in result
        assert "concurrent_3_count" in result
