"""
tests/test_return_pathway_5channel.py - 帰還経路5本体制の統合検証テスト

設計書: design_return_pathway_5channel.md

テスト設計11項目:
1. 経路D（ドライブ帯域帰還）の発火記録が正しく構成されること
2. 経路E（ムード追従速度帰還）の発火記録が正しく構成されること
3. 5本全てが同一ティック内で発火した場合の合算記述が正しく構成されること
4. 帰還先種類別の合算が独立して算出されること（感情帯域合算にドライブ帯域が混入しないこと）
5. 同時発火カウンタ（2+/3+/4+/5）が正しく更新されること
6. セッションサマリーに5本分の累積発火回数が含まれること
7. 既存3本（経路A/B/C）の動作が拡張前と完全に同一であること（回帰テスト）
8. 無効化時（環境変数OFF）に全記録処理がスキップされること
9. 通知点で例外が発生した場合に帰還処理が継続すること
10. 経路識別子が不正な場合の安全な拒否
11. 同一経路の重複記録防止（1ティック内で同一経路が2回通知された場合）
"""

from __future__ import annotations

import json
import logging
import os
import pytest
from typing import Any
from unittest.mock import patch

from tools.return_pathway_monitor import (
    ReturnPathwayMonitor,
    PATHWAY_A,
    PATHWAY_B,
    PATHWAY_C,
    PATHWAY_D,
    PATHWAY_E,
    _ALL_PATHWAYS,
    _EMOTION_PATHWAYS,
    _DRIVE_PATHWAYS,
    _MOOD_SPEED_PATHWAYS,
    _is_monitor_enabled,
)


# ── テスト用ヘルパー ──────────────────────────────────────────────────


def _make_emotion_deltas(**kwargs: float) -> dict[str, float]:
    """テスト用の感情変動辞書を作成する。"""
    return dict(kwargs)


def _make_drive_deltas(**kwargs: float) -> dict[str, float]:
    """テスト用のドライブ変動辞書を作成する。"""
    return dict(kwargs)


def _make_mood_speed_deltas(**kwargs: float) -> dict[str, float]:
    """テスト用のムード追従速度変調辞書を作成する。"""
    return dict(kwargs)


# ── テスト項目1: 経路D（ドライブ帯域帰還）の発火記録 ──────────────────


class TestPathwayDFiring:
    """経路D（ドライブ帯域帰還）の発火記録が正しく構成されること。"""

    def test_pathway_d_single_fire(self) -> None:
        """経路Dの単一発火記録。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_drive_deltas(social=0.005, curiosity=0.005, expression=0.005)
        monitor.record_firing(
            PATHWAY_D, tick_number=1, drive_deltas=deltas,
        )

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_D] == 1

    def test_pathway_d_record_format(self) -> None:
        """経路Dの発火記録フォーマットにdrive_deltasが含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_drive_deltas(social=0.005, curiosity=0.003)
        monitor.record_firing(
            PATHWAY_D, tick_number=1, drive_deltas=deltas,
        )

        assert len(monitor._tick_buffer) == 1
        record = monitor._tick_buffer[0]
        assert record["type"] == "return_pathway_firing"
        assert record["pathway_id"] == PATHWAY_D
        assert record["tick_number"] == 1
        assert "drive_deltas" in record
        assert record["drive_deltas"]["social"] == 0.005
        assert record["drive_deltas"]["curiosity"] == 0.003
        # emotion_deltasは含まれない
        assert "emotion_deltas" not in record

    def test_pathway_d_empty_deltas(self) -> None:
        """経路Dで空のdrive_deltasでも記録される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={})
        assert monitor.pathway_fire_counts[PATHWAY_D] == 1

    def test_pathway_d_none_deltas(self) -> None:
        """経路DでNoneのdrive_deltasでも記録される（空辞書に変換）。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas=None)
        assert monitor.pathway_fire_counts[PATHWAY_D] == 1
        record = monitor._tick_buffer[0]
        assert record["drive_deltas"] == {}

    def test_pathway_d_cumulative(self) -> None:
        """経路Dが複数ティックで累積カウントされること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        for tick in range(1, 4):
            monitor.record_firing(
                PATHWAY_D, tick_number=tick,
                drive_deltas=_make_drive_deltas(social=0.001),
            )
        assert monitor.pathway_fire_counts[PATHWAY_D] == 3


# ── テスト項目2: 経路E（ムード追従速度帰還）の発火記録 ────────────────


class TestPathwayEFiring:
    """経路E（ムード追従速度帰還）の発火記録が正しく構成されること。"""

    def test_pathway_e_single_fire(self) -> None:
        """経路Eの単一発火記録。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_mood_speed_deltas(
            valence_modulation=0.02, arousal_modulation=0.01,
        )
        monitor.record_firing(
            PATHWAY_E, tick_number=1, mood_speed_deltas=deltas,
        )

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_E] == 1

    def test_pathway_e_record_format(self) -> None:
        """経路Eの発火記録フォーマットにmood_speed_deltasが含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        deltas = _make_mood_speed_deltas(
            valence_modulation=0.02, arousal_modulation=0.015,
        )
        monitor.record_firing(
            PATHWAY_E, tick_number=1, mood_speed_deltas=deltas,
        )

        assert len(monitor._tick_buffer) == 1
        record = monitor._tick_buffer[0]
        assert record["type"] == "return_pathway_firing"
        assert record["pathway_id"] == PATHWAY_E
        assert "mood_speed_deltas" in record
        assert abs(record["mood_speed_deltas"]["valence_modulation"] - 0.02) < 1e-9
        assert abs(record["mood_speed_deltas"]["arousal_modulation"] - 0.015) < 1e-9
        # emotion_deltas, drive_deltasは含まれない
        assert "emotion_deltas" not in record
        assert "drive_deltas" not in record

    def test_pathway_e_empty_deltas(self) -> None:
        """経路Eで空のmood_speed_deltasでも記録される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={})
        assert monitor.pathway_fire_counts[PATHWAY_E] == 1

    def test_pathway_e_cumulative(self) -> None:
        """経路Eが複数ティックで累積カウントされること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        for tick in range(1, 6):
            monitor.record_firing(
                PATHWAY_E, tick_number=tick,
                mood_speed_deltas=_make_mood_speed_deltas(valence_modulation=0.01),
            )
        assert monitor.pathway_fire_counts[PATHWAY_E] == 5


# ── テスト項目3: 5本全てが同一ティック内で発火した場合の合算記述 ──────


class TestAllFivePathwaysConcurrent:
    """5本全てが同一ティック内で発火した場合の合算記述が正しく構成されること。"""

    def test_five_pathways_all_fire(self) -> None:
        """5経路全てが同一ティックで発火。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 経路A/B/C: 感情帯域
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.01),
        )
        monitor.record_firing(
            PATHWAY_B, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.02, anger=0.03),
        )
        monitor.record_firing(
            PATHWAY_C, tick_number=1,
            emotion_deltas=_make_emotion_deltas(fear=0.005),
        )
        # 経路D: ドライブ帯域
        monitor.record_firing(
            PATHWAY_D, tick_number=1,
            drive_deltas=_make_drive_deltas(social=0.005, curiosity=0.003),
        )
        # 経路E: ムード追従速度
        monitor.record_firing(
            PATHWAY_E, tick_number=1,
            mood_speed_deltas=_make_mood_speed_deltas(
                valence_modulation=0.02, arousal_modulation=0.01,
            ),
        )

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["fire_count"] == 5
        assert set(result["fired_pathways"]) == {
            PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E,
        }

    def test_five_pathways_combined_summary_structure(self) -> None:
        """5経路発火時のサマリーに種類別合算フィールドが含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.01),
        )
        monitor.record_firing(
            PATHWAY_B, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.02),
        )
        monitor.record_firing(
            PATHWAY_C, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.005),
        )
        monitor.record_firing(
            PATHWAY_D, tick_number=1,
            drive_deltas=_make_drive_deltas(social=0.005),
        )
        monitor.record_firing(
            PATHWAY_E, tick_number=1,
            mood_speed_deltas=_make_mood_speed_deltas(valence_modulation=0.02),
        )

        result = monitor.finalize_tick(1)
        assert result is not None
        assert "combined_emotion_deltas" in result
        assert "combined_drive_deltas" in result
        assert "combined_mood_speed_deltas" in result
        # 後方互換性のcombined_deltasも存在
        assert "combined_deltas" in result


# ── テスト項目4: 帰還先種類別の合算が独立して算出されること ──────────


class TestTypeSeparatedCombination:
    """帰還先種類別の合算が独立して算出されること（感情帯域合算にドライブ帯域が混入しないこと）。"""

    def test_emotion_deltas_independent(self) -> None:
        """感情帯域合算にドライブ帯域やムード追従速度が混入しないこと。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.01),
        )
        monitor.record_firing(
            PATHWAY_D, tick_number=1,
            drive_deltas=_make_drive_deltas(social=0.005),
        )
        monitor.record_firing(
            PATHWAY_E, tick_number=1,
            mood_speed_deltas=_make_mood_speed_deltas(valence_modulation=0.02),
        )

        result = monitor.finalize_tick(1)
        assert result is not None

        # 感情帯域合算にはjoyのみ
        assert abs(result["combined_emotion_deltas"]["joy"] - 0.01) < 1e-9
        assert "social" not in result["combined_emotion_deltas"]
        assert "valence_modulation" not in result["combined_emotion_deltas"]

    def test_drive_deltas_independent(self) -> None:
        """ドライブ帯域合算に感情帯域やムード追従速度が混入しないこと。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.01),
        )
        monitor.record_firing(
            PATHWAY_D, tick_number=1,
            drive_deltas=_make_drive_deltas(social=0.005, curiosity=0.003),
        )

        result = monitor.finalize_tick(1)
        assert result is not None

        # ドライブ帯域合算にはsocial/curiosityのみ
        assert abs(result["combined_drive_deltas"]["social"] - 0.005) < 1e-9
        assert abs(result["combined_drive_deltas"]["curiosity"] - 0.003) < 1e-9
        assert "joy" not in result["combined_drive_deltas"]

    def test_mood_speed_deltas_independent(self) -> None:
        """ムード追従速度合算に感情帯域やドライブ帯域が混入しないこと。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_B, tick_number=1,
            emotion_deltas=_make_emotion_deltas(anger=0.02),
        )
        monitor.record_firing(
            PATHWAY_E, tick_number=1,
            mood_speed_deltas=_make_mood_speed_deltas(
                valence_modulation=0.02, arousal_modulation=0.01,
            ),
        )

        result = monitor.finalize_tick(1)
        assert result is not None

        # ムード追従速度合算にはvalence_modulation/arousal_modulationのみ
        assert abs(result["combined_mood_speed_deltas"]["valence_modulation"] - 0.02) < 1e-9
        assert abs(result["combined_mood_speed_deltas"]["arousal_modulation"] - 0.01) < 1e-9
        assert "anger" not in result["combined_mood_speed_deltas"]
        assert "social" not in result["combined_mood_speed_deltas"]

    def test_combined_deltas_backward_compat(self) -> None:
        """後方互換性: combined_deltasはcombined_emotion_deltasと同一であること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.01),
        )
        monitor.record_firing(
            PATHWAY_B, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.02),
        )
        monitor.record_firing(
            PATHWAY_D, tick_number=1,
            drive_deltas=_make_drive_deltas(social=0.005),
        )

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["combined_deltas"] == result["combined_emotion_deltas"]

    def test_emotion_combination_across_abc(self) -> None:
        """3経路A/B/Cの感情帯域が正しく合算されること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.01, fear=-0.005),
        )
        monitor.record_firing(
            PATHWAY_B, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.02, anger=0.03),
        )
        monitor.record_firing(
            PATHWAY_C, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.005, fear=0.01),
        )

        result = monitor.finalize_tick(1)
        assert result is not None
        cd = result["combined_emotion_deltas"]
        assert abs(cd["joy"] - 0.035) < 1e-9
        assert abs(cd["fear"] - 0.005) < 1e-9
        assert abs(cd["anger"] - 0.03) < 1e-9

    def test_no_cross_type_scalar_sum(self) -> None:
        """3種類の帯域を横断的に1つのスカラー値に合算していないこと（安全弁6）。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(
            PATHWAY_A, tick_number=1,
            emotion_deltas=_make_emotion_deltas(joy=0.5),
        )
        monitor.record_firing(
            PATHWAY_D, tick_number=1,
            drive_deltas=_make_drive_deltas(social=0.5),
        )
        monitor.record_firing(
            PATHWAY_E, tick_number=1,
            mood_speed_deltas=_make_mood_speed_deltas(valence_modulation=0.5),
        )

        result = monitor.finalize_tick(1)
        assert result is not None
        # 横断的なtotal_deltaのようなフィールドがないこと
        assert "total_delta" not in result
        assert "cross_type_sum" not in result


# ── テスト項目5: 同時発火カウンタ（2+/3+/4+/5） ──────────────────────


class TestConcurrentCounters:
    """同時発火カウンタ（2+/3+/4+/5）が正しく更新されること。"""

    def test_2_pathways_concurrent(self) -> None:
        """2経路同時発火: 2+のみカウント。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_emotion_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas=_make_drive_deltas(social=0.01))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3plus_count == 0
        assert monitor.concurrent_4plus_count == 0
        assert monitor.concurrent_5_count == 0

    def test_3_pathways_concurrent(self) -> None:
        """3経路同時発火: 2+と3+がカウント。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_emotion_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_emotion_deltas(anger=0.02))
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas=_make_drive_deltas(social=0.01))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3plus_count == 1
        assert monitor.concurrent_4plus_count == 0
        assert monitor.concurrent_5_count == 0

    def test_4_pathways_concurrent(self) -> None:
        """4経路同時発火: 2+/3+/4+がカウント。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_emotion_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_emotion_deltas(anger=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas=_make_emotion_deltas(fear=0.005))
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas=_make_drive_deltas(social=0.01))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3plus_count == 1
        assert monitor.concurrent_4plus_count == 1
        assert monitor.concurrent_5_count == 0

    def test_5_pathways_concurrent(self) -> None:
        """5経路同時発火: 全てのカウンタがカウント。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=_make_emotion_deltas(joy=0.01))
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas=_make_emotion_deltas(anger=0.02))
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas=_make_emotion_deltas(fear=0.005))
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas=_make_drive_deltas(social=0.01))
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas=_make_mood_speed_deltas(valence_modulation=0.02))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3plus_count == 1
        assert monitor.concurrent_4plus_count == 1
        assert monitor.concurrent_5_count == 1

    def test_single_pathway_no_concurrent(self) -> None:
        """単一経路発火では全てのカウンタがゼロ。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas=_make_drive_deltas(social=0.01))
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 0
        assert monitor.concurrent_3plus_count == 0
        assert monitor.concurrent_4plus_count == 0
        assert monitor.concurrent_5_count == 0

    def test_cumulative_concurrent_counters(self) -> None:
        """複数ティックにわたる累積カウンタ。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1: 5経路同時
        for pathway, deltas_key in [
            (PATHWAY_A, "emotion_deltas"), (PATHWAY_B, "emotion_deltas"),
            (PATHWAY_C, "emotion_deltas"), (PATHWAY_D, "drive_deltas"),
            (PATHWAY_E, "mood_speed_deltas"),
        ]:
            kwargs = {deltas_key: {"x": 0.01}}
            monitor.record_firing(pathway, tick_number=1, **kwargs)
        monitor.finalize_tick(1)

        # ティック2: 3経路同時
        monitor.record_firing(PATHWAY_A, tick_number=2, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_D, tick_number=2, drive_deltas={"social": 0.01})
        monitor.record_firing(PATHWAY_E, tick_number=2, mood_speed_deltas={"v": 0.01})
        monitor.finalize_tick(2)

        # ティック3: 2経路同時
        monitor.record_firing(PATHWAY_A, tick_number=3, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=3, emotion_deltas={"anger": 0.02})
        monitor.finalize_tick(3)

        assert monitor.concurrent_2plus_count == 3
        assert monitor.concurrent_3plus_count == 2
        assert monitor.concurrent_4plus_count == 1
        assert monitor.concurrent_5_count == 1

    def test_concurrent_3_count_backward_compat(self) -> None:
        """後方互換性: concurrent_3_countがconcurrent_3plus_countと同値。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas={"fear": 0.005})
        monitor.finalize_tick(1)

        assert monitor.concurrent_3_count == monitor.concurrent_3plus_count


# ── テスト項目6: セッションサマリーに5本分の累積発火回数 ──────────────


class TestSessionSummary5Channel:
    """セッションサマリーに5本分の累積発火回数が含まれること。"""

    def test_session_summary_includes_all_5_pathways(self) -> None:
        """セッションサマリーに5経路分の発火回数が含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # 各経路を異なるティックで発火
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.finalize_tick(1)
        monitor.record_firing(PATHWAY_B, tick_number=2, emotion_deltas={"anger": 0.02})
        monitor.finalize_tick(2)
        monitor.record_firing(PATHWAY_C, tick_number=3, emotion_deltas={"fear": 0.005})
        monitor.finalize_tick(3)
        monitor.record_firing(PATHWAY_D, tick_number=4, drive_deltas={"social": 0.005})
        monitor.finalize_tick(4)
        monitor.record_firing(PATHWAY_E, tick_number=5, mood_speed_deltas={"valence_modulation": 0.02})
        monitor.finalize_tick(5)

        summary = monitor.emit_session_summary()
        assert summary is not None
        counts = summary["pathway_fire_counts"]
        assert counts[PATHWAY_A] == 1
        assert counts[PATHWAY_B] == 1
        assert counts[PATHWAY_C] == 1
        assert counts[PATHWAY_D] == 1
        assert counts[PATHWAY_E] == 1

    def test_session_summary_concurrent_counters(self) -> None:
        """セッションサマリーに4段階のカウンタが含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # 5経路同時発火
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas={"fear": 0.005})
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.005})
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.02})
        monitor.finalize_tick(1)

        summary = monitor.emit_session_summary()
        assert summary is not None
        assert summary["concurrent_2plus_count"] == 1
        assert summary["concurrent_3_count"] == 1
        assert summary["concurrent_3plus_count"] == 1
        assert summary["concurrent_4plus_count"] == 1
        assert summary["concurrent_5_count"] == 1

    def test_empty_session_summary_5_pathways(self) -> None:
        """発火なしのセッションサマリーに5経路分のゼロカウントが含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        summary = monitor.emit_session_summary()
        assert summary is not None
        counts = summary["pathway_fire_counts"]
        assert len(counts) == 5
        assert counts[PATHWAY_D] == 0
        assert counts[PATHWAY_E] == 0

    def test_get_summary_includes_new_counters(self) -> None:
        """get_summaryアクセサに4段階のカウンタが含まれること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        summary = monitor.get_summary()
        assert "concurrent_4plus_count" in summary
        assert "concurrent_5_count" in summary
        assert summary["concurrent_4plus_count"] == 0
        assert summary["concurrent_5_count"] == 0


# ── テスト項目7: 既存3本（経路A/B/C）の回帰テスト ──────────────────────


class TestBackwardCompatibility:
    """既存3本（経路A/B/C）の動作が拡張前と完全に同一であること（回帰テスト）。"""

    def test_abc_fire_counts_unchanged(self) -> None:
        """経路A/B/Cの発火カウントが拡張前と同一動作。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas={"fear": 0.005})

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 1
        assert counts[PATHWAY_B] == 1
        assert counts[PATHWAY_C] == 1

    def test_abc_combined_deltas_unchanged(self) -> None:
        """経路A/B/Cの合算記述が拡張前と同一。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"joy": 0.02, "anger": 0.03})
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas={"joy": 0.005})

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["fire_count"] == 3
        # combined_deltas は感情帯域合算のみ（後方互換）
        cd = result["combined_deltas"]
        assert abs(cd["joy"] - 0.035) < 1e-9
        assert abs(cd["anger"] - 0.03) < 1e-9

    def test_abc_concurrent_counters_unchanged(self) -> None:
        """経路A/B/Cのみの場合の同時発火カウンタが拡張前と同一動作。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # ティック1: 2経路
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"anger": 0.02})
        monitor.finalize_tick(1)

        assert monitor.concurrent_2plus_count == 1
        assert monitor.concurrent_3_count == 0

    def test_abc_session_summary_unchanged(self) -> None:
        """経路A/B/Cのセッションサマリーが拡張前と同一フィールドを含む。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.finalize_tick(1)

        summary = monitor.emit_session_summary()
        assert summary is not None
        assert "pathway_fire_counts" in summary
        assert "concurrent_2plus_count" in summary
        assert "concurrent_3_count" in summary  # 後方互換キー

    def test_abc_record_format_unchanged(self) -> None:
        """経路A/B/Cの発火記録がemotion_deltasを含むこと。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})

        record = monitor._tick_buffer[0]
        assert record["type"] == "return_pathway_firing"
        assert "emotion_deltas" in record
        assert record["emotion_deltas"]["joy"] == 0.01

    def test_abc_only_no_drive_or_mood_in_combined(self) -> None:
        """経路A/B/Cのみ発火時にdrive/moodの合算が空であること。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas={"fear": 0.005})

        result = monitor.finalize_tick(1)
        assert result is not None
        assert result["combined_drive_deltas"] == {}
        assert result["combined_mood_speed_deltas"] == {}

    def test_pathway_constants_backward_compat(self) -> None:
        """経路識別子定数が既存値を維持していること。"""
        assert PATHWAY_A == "memory_emotion_return"
        assert PATHWAY_B == "selection_emotion_return"
        assert PATHWAY_C == "other_hypothesis_emotion_return"

    def test_all_pathways_includes_de(self) -> None:
        """_ALL_PATHWAYSが5本を含むこと。"""
        assert len(_ALL_PATHWAYS) == 5
        assert PATHWAY_D in _ALL_PATHWAYS
        assert PATHWAY_E in _ALL_PATHWAYS

    def test_pathway_classification_sets(self) -> None:
        """経路分類が正しいこと。"""
        assert _EMOTION_PATHWAYS == frozenset({PATHWAY_A, PATHWAY_B, PATHWAY_C})
        assert _DRIVE_PATHWAYS == frozenset({PATHWAY_D})
        assert _MOOD_SPEED_PATHWAYS == frozenset({PATHWAY_E})


# ── テスト項目8: 無効化時に全記録処理がスキップされること ──────────────


class TestDisabledMonitor5Channel:
    """無効化時（環境変数OFF）に全記録処理がスキップされること。"""

    def test_disabled_no_recording_de(self) -> None:
        """無効時は経路D/Eの記録がされない。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.01})
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.02})

        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_D] == 0
        assert counts[PATHWAY_E] == 0

    def test_disabled_finalize_returns_none(self) -> None:
        """無効時のfinalize_tickはNoneを返す。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        result = monitor.finalize_tick(1)
        assert result is None

    def test_disabled_session_summary_returns_none(self) -> None:
        """無効時のemit_session_summaryはNoneを返す。"""
        monitor = ReturnPathwayMonitor(enabled=False)
        result = monitor.emit_session_summary()
        assert result is None

    def test_disabled_no_logs(self, caplog: Any) -> None:
        """無効時はログが出力されない。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=False)
            monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
            monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.01})
            monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.02})
            monitor.finalize_tick(1)
            monitor.emit_session_summary()

        pathway_logs = [r for r in caplog.records if "return_pathway" in r.message]
        assert len(pathway_logs) == 0


# ── テスト項目9: 例外発生時に帰還処理が継続すること ──────────────────


class TestExceptionSafety:
    """通知点で例外が発生した場合に帰還処理が継続すること。"""

    def test_record_firing_exception_safe(self) -> None:
        """record_firing内の例外は捕捉されスキップされる。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # emotion_deltasにNoneを渡してもクラッシュしない
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas=None)

    def test_record_firing_drive_exception_safe(self) -> None:
        """drive_deltasに不正値を渡してもクラッシュしない。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 文字列値を含む辞書でもクラッシュしない
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": "invalid"})  # type: ignore

    def test_finalize_tick_exception_safe(self) -> None:
        """finalize_tick内の例外は捕捉されNoneが返る。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 内部状態を直接壊す
        monitor._tick_buffer = [{"no_pathway_id": True}]
        monitor._current_tick = 1
        result = monitor.finalize_tick(1)
        # 例外が捕捉されてNoneが返るか、正常処理される

    def test_session_summary_exception_safe(self) -> None:
        """emit_session_summary内の例外は捕捉されNoneが返る。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor._pathway_fire_counts = None  # type: ignore
        result = monitor.emit_session_summary()
        assert result is None

    def test_safety_valve_3_no_enrichment(self) -> None:
        """enrichmentへの接続がないことの構造的確認。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert not hasattr(monitor, "get_enrichment")
        assert not hasattr(monitor, "enrichment")
        assert not hasattr(monitor, "build_enrichment")

    def test_safety_valve_4_no_psyche_state(self) -> None:
        """psyche状態への書き込み経路がないことの構造的確認。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert not hasattr(monitor, "apply_to_psyche")
        assert not hasattr(monitor, "update_state")
        assert not hasattr(monitor, "modify_emotions")
        assert not hasattr(monitor, "modify_drives")
        assert not hasattr(monitor, "modify_mood")

    def test_safety_valve_5_no_persistence(self) -> None:
        """永続化メソッドがないことの構造的確認。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        assert not hasattr(monitor, "to_dict")
        assert not hasattr(monitor, "from_dict")
        assert not hasattr(monitor, "save")
        assert not hasattr(monitor, "load")


# ── テスト項目10: 不正な経路識別子の安全な拒否 ────────────────────────


class TestInvalidPathwayId:
    """経路識別子が不正な場合の安全な拒否。"""

    def test_invalid_pathway_ignored(self) -> None:
        """不正な経路識別子は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing("invalid_pathway", tick_number=1, emotion_deltas={"joy": 0.01})

        counts = monitor.pathway_fire_counts
        for pathway in _ALL_PATHWAYS:
            assert counts[pathway] == 0

    def test_empty_string_pathway_ignored(self) -> None:
        """空文字列の経路識別子は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing("", tick_number=1, emotion_deltas={"joy": 0.01})
        assert all(v == 0 for v in monitor.pathway_fire_counts.values())

    def test_none_like_pathway_ignored(self) -> None:
        """数値の経路識別子は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(42, tick_number=1, emotion_deltas={"joy": 0.01})  # type: ignore
        assert all(v == 0 for v in monitor.pathway_fire_counts.values())

    def test_partial_match_pathway_ignored(self) -> None:
        """部分一致の経路識別子は無視される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing("memory_emotion", tick_number=1, emotion_deltas={"joy": 0.01})
        assert all(v == 0 for v in monitor.pathway_fire_counts.values())


# ── テスト項目11: 同一経路の重複記録防止 ──────────────────────────────


class TestDuplicatePrevention:
    """同一経路の重複記録防止（1ティック内で同一経路が2回通知された場合）。"""

    def test_duplicate_emotion_pathway_prevented(self) -> None:
        """同一ティック内で同一感情経路の重複が防止される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.05})
        assert monitor.pathway_fire_counts[PATHWAY_A] == 1

    def test_duplicate_drive_pathway_prevented(self) -> None:
        """同一ティック内で同一ドライブ経路の重複が防止される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.01})
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.05})
        assert monitor.pathway_fire_counts[PATHWAY_D] == 1

    def test_duplicate_mood_pathway_prevented(self) -> None:
        """同一ティック内で同一ムード経路の重複が防止される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.01})
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.05})
        assert monitor.pathway_fire_counts[PATHWAY_E] == 1

    def test_different_pathways_same_tick_allowed(self) -> None:
        """異なる経路は同一ティック内で全て記録される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=1, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=1, emotion_deltas={"fear": 0.005})
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.01})
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.02})

        assert len(monitor._tick_buffer) == 5
        for pathway in _ALL_PATHWAYS:
            assert monitor.pathway_fire_counts[pathway] == 1

    def test_duplicate_resolved_across_ticks(self) -> None:
        """異なるティックでは同一経路が再度記録される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.01})
        monitor.record_firing(PATHWAY_D, tick_number=2, drive_deltas={"social": 0.02})
        assert monitor.pathway_fire_counts[PATHWAY_D] == 2

    def test_first_record_wins_on_duplicate(self) -> None:
        """重複時は最初の記録が保持される。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.01})
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.05})

        result = monitor.finalize_tick(1)
        assert result is not None
        # 最初のdrive_deltasが保持されている
        assert abs(result["combined_drive_deltas"]["social"] - 0.01) < 1e-9


# ── ログ出力テスト ────────────────────────────────────────────────────


class TestLogging5Channel:
    """新設経路のログ出力テスト。"""

    def test_pathway_d_log_emitted(self, caplog: Any) -> None:
        """経路Dの発火記録がログに出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=True)
            monitor.record_firing(
                PATHWAY_D, tick_number=1,
                drive_deltas={"social": 0.005},
            )

        found = False
        for record in caplog.records:
            if "return_pathway_firing" in record.message:
                data = json.loads(record.message)
                if data.get("pathway_id") == PATHWAY_D:
                    assert "drive_deltas" in data
                    found = True
        assert found, "Pathway D firing log not found"

    def test_pathway_e_log_emitted(self, caplog: Any) -> None:
        """経路Eの発火記録がログに出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=True)
            monitor.record_firing(
                PATHWAY_E, tick_number=1,
                mood_speed_deltas={"valence_modulation": 0.02},
            )

        found = False
        for record in caplog.records:
            if "return_pathway_firing" in record.message:
                data = json.loads(record.message)
                if data.get("pathway_id") == PATHWAY_E:
                    assert "mood_speed_deltas" in data
                    found = True
        assert found, "Pathway E firing log not found"

    def test_5channel_cycle_summary_log(self, caplog: Any) -> None:
        """5経路サイクルサマリーがログに出力される。"""
        with caplog.at_level(logging.DEBUG, logger="cyrene.monitor.return_pathway"):
            monitor = ReturnPathwayMonitor(enabled=True)
            monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
            monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.005})
            monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.02})
            monitor.finalize_tick(1)

        found = False
        for record in caplog.records:
            if "return_pathway_cycle_summary" in record.message:
                data = json.loads(record.message)
                assert data["fire_count"] == 3
                assert "combined_drive_deltas" in data
                assert "combined_mood_speed_deltas" in data
                found = True
        assert found, "5-channel cycle summary log not found"


# ── 複合シナリオテスト ──────────────────────────────────────────────


class TestComplexScenarios:
    """複数ティックにわたる複合シナリオのテスト。"""

    def test_10_tick_mixed_scenario(self) -> None:
        """10ティックの混合シナリオ。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1: A + D (2経路)
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.005})
        monitor.finalize_tick(1)

        # ティック2: B + E (2経路)
        monitor.record_firing(PATHWAY_B, tick_number=2, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_E, tick_number=2, mood_speed_deltas={"valence_modulation": 0.02})
        monitor.finalize_tick(2)

        # ティック3: A + B + C (3経路)
        monitor.record_firing(PATHWAY_A, tick_number=3, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=3, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=3, emotion_deltas={"fear": 0.005})
        monitor.finalize_tick(3)

        # ティック4: no firing
        monitor.finalize_tick(4)

        # ティック5: 全5経路
        monitor.record_firing(PATHWAY_A, tick_number=5, emotion_deltas={"joy": 0.01})
        monitor.record_firing(PATHWAY_B, tick_number=5, emotion_deltas={"anger": 0.02})
        monitor.record_firing(PATHWAY_C, tick_number=5, emotion_deltas={"fear": 0.005})
        monitor.record_firing(PATHWAY_D, tick_number=5, drive_deltas={"social": 0.005})
        monitor.record_firing(PATHWAY_E, tick_number=5, mood_speed_deltas={"valence_modulation": 0.02})
        monitor.finalize_tick(5)

        # ティック6-10: D only
        for t in range(6, 11):
            monitor.record_firing(PATHWAY_D, tick_number=t, drive_deltas={"curiosity": 0.003})
            monitor.finalize_tick(t)

        # 検証
        counts = monitor.pathway_fire_counts
        assert counts[PATHWAY_A] == 3   # ticks 1, 3, 5
        assert counts[PATHWAY_B] == 3   # ticks 2, 3, 5
        assert counts[PATHWAY_C] == 2   # ticks 3, 5
        assert counts[PATHWAY_D] == 7   # ticks 1, 5, 6-10
        assert counts[PATHWAY_E] == 2   # ticks 2, 5
        assert monitor.concurrent_2plus_count == 4  # ticks 1, 2, 3, 5
        assert monitor.concurrent_3plus_count == 2  # ticks 3, 5
        assert monitor.concurrent_4plus_count == 1  # tick 5
        assert monitor.concurrent_5_count == 1      # tick 5

    def test_last_tick_record_updates(self) -> None:
        """last_tick_recordが正しく更新されること。"""
        monitor = ReturnPathwayMonitor(enabled=True)

        # ティック1: D only
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.005})
        monitor.finalize_tick(1)
        record = monitor.last_tick_record
        assert record is not None
        assert record["fire_count"] == 1
        assert PATHWAY_D in record["fired_pathways"]

        # ティック2: E only
        monitor.record_firing(PATHWAY_E, tick_number=2, mood_speed_deltas={"v": 0.02})
        monitor.finalize_tick(2)
        record = monitor.last_tick_record
        assert record is not None
        assert record["fire_count"] == 1
        assert PATHWAY_E in record["fired_pathways"]

        # ティック3: no firing
        monitor.finalize_tick(3)
        assert monitor.last_tick_record is None

    def test_no_evaluation_in_any_combined_deltas(self) -> None:
        """合算値に対する評価・判定・閾値比較がないことの確認。"""
        monitor = ReturnPathwayMonitor(enabled=True)
        # 大きな値を投入
        monitor.record_firing(PATHWAY_A, tick_number=1, emotion_deltas={"joy": 0.5})
        monitor.record_firing(PATHWAY_D, tick_number=1, drive_deltas={"social": 0.5})
        monitor.record_firing(PATHWAY_E, tick_number=1, mood_speed_deltas={"v": 0.5})

        result = monitor.finalize_tick(1)
        assert result is not None
        # 評価関連フィールドが無いこと
        assert "warning" not in result
        assert "error" not in result
        assert "evaluation" not in result
        assert "threshold" not in result
