"""
tools/return_pathway_monitor.py - 帰還経路の動作検証と相互干渉検出

5本の帰還経路が実運用中に発火した事実と、複数経路が同一ティック内で
同時に適用された際の合算帯域を、事後的に記録する仕組みである。

帰還経路一覧:
- 経路A: 記憶想起→感情帯域変動
- 経路B: ポリシー選択→感情帯域変動
- 経路C: 他者仮説→感情帯域変動
- 経路D: 行動結果の蓄積多様性→ドライブ帯域の微弱反映
- 経路E: 感情帰還方向の連続性→ムード追従速度への変調量

設計書: design_return_pathway_verification.md / design_return_pathway_5channel.md

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)のみ
- 内部システムの状態変数を一切変更しない(READ-ONLY観測のみ)
- 判断・行動・選択に一切介入しない
- 計測値に基づく条件分岐を一切持たない
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない
- enrichmentに反映されない
- psycheの状態更新経路に接続されない

安全弁:
1. 計測失敗時の安全な無視(例外を捕捉しログスキップ、本体処理続行)
2. 環境変数による完全無効化(既存のモニタリング基盤と同一: CYRENE_MONITOR)
3. enrichment非接続: 記録内容はenrichmentの構成・内容・項目数に一切影響しない
4. psyche状態非接続: 記録内容はpsycheの状態変数に一切書き込まない
5. 永続化非対象: 全内部状態はセッション境界で消失する
6. 帰還先種類混合の禁止: 感情帯域・ドライブ帯域・ムード追従速度の3種を
   横断的に1つのスカラー値に合算しない。種類ごとに独立した合算を記述する
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

# 独自ログ名前空間(既存ログとの混在防止)
_logger = logging.getLogger("cyrene.monitor.return_pathway")


# ── 帰還経路識別子 ──────────────────────────────────────────────────

PATHWAY_A = "memory_emotion_return"
PATHWAY_B = "selection_emotion_return"
PATHWAY_C = "other_hypothesis_emotion_return"
PATHWAY_D = "result_diversity_drive_return"
PATHWAY_E = "emotion_return_tracking_speed"

_ALL_PATHWAYS = frozenset({PATHWAY_A, PATHWAY_B, PATHWAY_C, PATHWAY_D, PATHWAY_E})

# 帰還先による経路の分類
_EMOTION_PATHWAYS = frozenset({PATHWAY_A, PATHWAY_B, PATHWAY_C})
_DRIVE_PATHWAYS = frozenset({PATHWAY_D})
_MOOD_SPEED_PATHWAYS = frozenset({PATHWAY_E})


# ── 環境変数制御 ──────────────────────────────────────────────────

def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存の実行時観測基盤と同じ環境変数(CYRENE_MONITOR)に従う。
    インポート時ではなく呼び出し時に環境変数を確認する。
    テストフィクスチャでの動的設定変更に対応するため。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── ReturnPathwayMonitor ────────────────────────────────────────────


class ReturnPathwayMonitor:
    """帰還経路の発火記録とティック内合算記述。

    既存のモニタリング基盤(ExecutionMonitor)と並立する独立セクションとして配置。
    ExecutionMonitorの既存メソッドを変更せず、新規クラスとして追加。
    既存のPhase計測・圧縮比記録・API記録・enrichment分布記述とは独立し、
    相互参照しない。

    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない
    - 計測値に基づく条件分岐を一切持たない(記録のみ)
    - 出力はログストリームへのJSON書き込みのみ
    - 合算値に対する評価・判定・閾値比較を行わない
    """

    def __init__(self, enabled: Optional[bool] = None) -> None:
        """初期化。

        Args:
            enabled: 明示的な有効/無効指定。Noneの場合は環境変数で判定。
        """
        # 有効/無効判定(明示指定 > 環境変数)
        # 安全弁2: 環境変数による完全無効化
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # ── ティック内一時バッファ ──
        # 現在処理中のティック番号
        self._current_tick: int = -1
        # そのティック内で受け取った帰還経路発火記録のリスト
        # 最大5件: 経路A/B/C/D/Eの各1件まで
        self._tick_buffer: list[dict[str, Any]] = []

        # ── セッション累積カウンタ ──
        # 各帰還経路の累積発火回数(経路識別子をキーとする辞書)
        self._pathway_fire_counts: dict[str, int] = {
            PATHWAY_A: 0,
            PATHWAY_B: 0,
            PATHWAY_C: 0,
            PATHWAY_D: 0,
            PATHWAY_E: 0,
        }
        # 同時発火カウンタ(4段階: 2+/3+/4+/5)
        self._concurrent_2plus_count: int = 0
        self._concurrent_3plus_count: int = 0
        self._concurrent_4plus_count: int = 0
        self._concurrent_5_count: int = 0

        # ── 合算帯域上限到達カウンタ(セッション消失) ──
        # 種類ごとの合算上限到達回数(事後分析用。永続化対象外)
        self._aggregate_cap_hit_counts: dict[str, int] = {
            "emotion": 0,
            "drive": 0,
            "mood_speed": 0,
        }

        # ── 直近の発火情報(1ティック分のみ保持) ──
        self._last_tick_record: Optional[dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        """モニタリングが有効かどうか。"""
        return self._enabled

    @property
    def pathway_fire_counts(self) -> dict[str, int]:
        """各帰還経路の累積発火回数(読み取り専用コピー)。"""
        return dict(self._pathway_fire_counts)

    @property
    def concurrent_2plus_count(self) -> int:
        """同時発火(2経路以上)の累積回数。"""
        return self._concurrent_2plus_count

    @property
    def concurrent_3_count(self) -> int:
        """3経路以上同時発火の累積回数。

        後方互換性のため concurrent_3plus_count と同じ値を返す。
        """
        return self._concurrent_3plus_count

    @property
    def concurrent_3plus_count(self) -> int:
        """3経路以上同時発火の累積回数。"""
        return self._concurrent_3plus_count

    @property
    def concurrent_4plus_count(self) -> int:
        """4経路以上同時発火の累積回数。"""
        return self._concurrent_4plus_count

    @property
    def concurrent_5_count(self) -> int:
        """5経路同時発火の累積回数。"""
        return self._concurrent_5_count

    @property
    def last_tick_record(self) -> Optional[dict[str, Any]]:
        """直近のティックにおける発火記録(読み取り専用コピー)。"""
        if self._last_tick_record is None:
            return None
        return dict(self._last_tick_record)

    # ── 段階1: 発火記録の構成 ────────────────────────────────────────

    def record_firing(
        self,
        pathway_id: str,
        tick_number: int,
        emotion_deltas: Optional[dict[str, float]] = None,
        drive_deltas: Optional[dict[str, float]] = None,
        mood_speed_deltas: Optional[dict[str, float]] = None,
    ) -> None:
        """帰還経路の発火事実を1件の記録として構成する。

        各帰還経路から発火事実を受け取るたびに呼ばれる。
        帰還経路の処理が完了した後に通知される形式であり、
        帰還経路の処理自体には一切関与しない。

        Args:
            pathway_id: 発火した帰還経路の識別子
                (PATHWAY_A〜PATHWAY_Eのいずれか)
            tick_number: 発火が発生したティック番号
            emotion_deltas: 適用された感情帯域変動の各次元の値
                (経路A/B/C用)
            drive_deltas: 適用されたドライブ帯域変動の各軸の値
                (経路D用。キーはドライブ軸名: social/curiosity/expression)
            mood_speed_deltas: 適用されたムード追従速度変調量
                (経路E用。キーはvalence_modulation/arousal_modulation)
        """
        if not self._enabled:
            return
        try:
            # 識別子の検証
            if pathway_id not in _ALL_PATHWAYS:
                return

            # 新しいティックが開始された場合、バッファをリセット
            if tick_number != self._current_tick:
                self._current_tick = tick_number
                self._tick_buffer = []

            # ティック内バッファに追加(最大5件: 経路A/B/C/D/Eの各1件まで)
            # 同一経路の重複記録を防止
            existing_ids = {r["pathway_id"] for r in self._tick_buffer}
            if pathway_id in existing_ids:
                return

            record: dict[str, Any] = {
                "type": "return_pathway_firing",
                "timestamp": time.time(),
                "pathway_id": pathway_id,
                "tick_number": tick_number,
            }

            # 帰還先種類に応じた変動量の記録
            if pathway_id in _EMOTION_PATHWAYS:
                record["emotion_deltas"] = dict(emotion_deltas) if emotion_deltas else {}
            elif pathway_id in _DRIVE_PATHWAYS:
                record["drive_deltas"] = dict(drive_deltas) if drive_deltas else {}
            elif pathway_id in _MOOD_SPEED_PATHWAYS:
                record["mood_speed_deltas"] = dict(mood_speed_deltas) if mood_speed_deltas else {}

            self._tick_buffer.append(record)

            # 累積カウンタ更新
            self._pathway_fire_counts[pathway_id] = (
                self._pathway_fire_counts.get(pathway_id, 0) + 1
            )

            # ログ出力
            self._emit_json(record)

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    # ── 合算帯域上限の記録 ─────────────────────────────────────────

    @property
    def aggregate_cap_hit_counts(self) -> dict[str, int]:
        """種類ごとの合算上限到達回数(読み取り専用コピー)。

        キー: "emotion", "drive", "mood_speed"
        値: セッション内の到達回数
        永続化対象外(セッション境界で消失)。
        """
        return dict(self._aggregate_cap_hit_counts)

    def record_aggregate_cap_hit(self, kind: str) -> None:
        """合算帯域上限に到達した事実を記録する。

        帰還先種類(emotion/drive/mood_speed)ごとに独立してカウントする。
        通知失敗時は例外を捕捉してスキップする(安全弁パターン踏襲)。

        Args:
            kind: 帰還先種類("emotion", "drive", "mood_speed"のいずれか)
        """
        try:
            if kind in self._aggregate_cap_hit_counts:
                self._aggregate_cap_hit_counts[kind] += 1

                # ログ出力
                record: dict[str, Any] = {
                    "type": "return_pathway_aggregate_cap_hit",
                    "timestamp": time.time(),
                    "kind": kind,
                    "cumulative_count": self._aggregate_cap_hit_counts[kind],
                }
                self._emit_json(record)
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    def get_tick_buffer(self) -> list[dict[str, Any]]:
        """現在のティック内バッファを読み取り専用で返す。

        合算帯域上限の算出に使用する。
        バッファの内容を変更しない。

        Returns:
            ティック内発火記録のリスト(コピー)。
        """
        return [dict(r) for r in self._tick_buffer]

    def get_current_tick(self) -> int:
        """現在処理中のティック番号を返す。"""
        return self._current_tick

    # ── 段階2: 同一ティック内の合算記述 ──────────────────────────────

    def finalize_tick(self, tick_number: int) -> Optional[dict[str, Any]]:
        """1サイクル(1ティック)の処理完了時に、ティック内の発火記録を集約する。

        そのティック内で発火した全帰還経路の記録を集約し、
        以下の情報を記述する:
        - そのティック内で発火した帰還経路の一覧(識別子の列挙)
        - 発火した経路の数(同時発火数)
        - 種類別の合算変動値:
          - 感情帯域の合算(経路A/B/C分)
          - ドライブ帯域の合算(経路D分)
          - ムード追従速度の合算(経路E分)

        この合算記述は事実の列挙のみであり、合算値に対する
        評価・判定・閾値比較は行わない。

        安全弁6: 感情帯域・ドライブ帯域・ムード追従速度の3種を
        横断的に1つのスカラー値に合算しない。種類ごとに独立した合算を記述する。

        Args:
            tick_number: 完了したティック番号

        Returns:
            合算記述の辞書。発火がなかった場合はNone。
        """
        if not self._enabled:
            return None
        try:
            # このティックの記録がなければスキップ
            if tick_number != self._current_tick or not self._tick_buffer:
                # 直近記録を更新(発火なし)
                self._last_tick_record = None
                return None

            # 発火した経路の一覧
            fired_pathways = [r["pathway_id"] for r in self._tick_buffer]
            fire_count = len(fired_pathways)

            # 種類別の合算変動値(安全弁6: 種類ごとに独立した合算)
            # 感情帯域の合算(経路A/B/C)
            combined_emotion_deltas: dict[str, float] = {}
            for record in self._tick_buffer:
                for dim, delta in record.get("emotion_deltas", {}).items():
                    if isinstance(delta, (int, float)):
                        combined_emotion_deltas[dim] = (
                            combined_emotion_deltas.get(dim, 0.0) + delta
                        )

            # ドライブ帯域の合算(経路D)
            combined_drive_deltas: dict[str, float] = {}
            for record in self._tick_buffer:
                for dim, delta in record.get("drive_deltas", {}).items():
                    if isinstance(delta, (int, float)):
                        combined_drive_deltas[dim] = (
                            combined_drive_deltas.get(dim, 0.0) + delta
                        )

            # ムード追従速度の合算(経路E)
            combined_mood_speed_deltas: dict[str, float] = {}
            for record in self._tick_buffer:
                for dim, delta in record.get("mood_speed_deltas", {}).items():
                    if isinstance(delta, (int, float)):
                        combined_mood_speed_deltas[dim] = (
                            combined_mood_speed_deltas.get(dim, 0.0) + delta
                        )

            # 同時発火カウンタの更新(4段階)
            if fire_count >= 2:
                self._concurrent_2plus_count += 1
            if fire_count >= 3:
                self._concurrent_3plus_count += 1
            if fire_count >= 4:
                self._concurrent_4plus_count += 1
            if fire_count >= 5:
                self._concurrent_5_count += 1

            # 合算記述の構成
            cycle_summary: dict[str, Any] = {
                "type": "return_pathway_cycle_summary",
                "timestamp": time.time(),
                "tick_number": tick_number,
                "fired_pathways": fired_pathways,
                "fire_count": fire_count,
                # 後方互換性のため combined_deltas を感情帯域合算として維持
                "combined_deltas": combined_emotion_deltas,
                # 種類別合算(独立)
                "combined_emotion_deltas": combined_emotion_deltas,
                "combined_drive_deltas": combined_drive_deltas,
                "combined_mood_speed_deltas": combined_mood_speed_deltas,
            }

            # 直近記録を更新(次のティック処理完了時に上書き)
            self._last_tick_record = cycle_summary

            # ログ出力
            self._emit_json(cycle_summary)

            # ティック内バッファをクリア
            self._tick_buffer = []

            return cycle_summary

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            return None

    # ── 段階3: セッションサマリーの出力 ──────────────────────────────

    def emit_session_summary(self) -> Optional[dict[str, Any]]:
        """セッション終了時のセッション累積情報を出力する。

        セッション全体にわたる累積情報を出力する:
        - 各帰還経路の累積発火回数(5本分)
        - 同時発火カウンタ(4段階: 2+/3+/4+/5)

        Returns:
            セッションサマリーの辞書。
        """
        if not self._enabled:
            return None
        try:
            summary: dict[str, Any] = {
                "type": "return_pathway_session_summary",
                "timestamp": time.time(),
                "pathway_fire_counts": dict(self._pathway_fire_counts),
                "concurrent_2plus_count": self._concurrent_2plus_count,
                "concurrent_3_count": self._concurrent_3plus_count,
                "concurrent_3plus_count": self._concurrent_3plus_count,
                "concurrent_4plus_count": self._concurrent_4plus_count,
                "concurrent_5_count": self._concurrent_5_count,
                "aggregate_cap_hit_counts": dict(self._aggregate_cap_hit_counts),
            }

            # ログ出力
            self._emit_json(summary)

            return summary

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            return None

    # ── 読み取り専用アクセサ ─────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """現在の累積情報を読み取り専用で返す。

        外部の分析ツール(シミュレータ等)が呼び出す読み取り専用アクセサ。

        Returns:
            累積情報の辞書。
        """
        return {
            "pathway_fire_counts": dict(self._pathway_fire_counts),
            "concurrent_2plus_count": self._concurrent_2plus_count,
            "concurrent_3_count": self._concurrent_3plus_count,
            "concurrent_3plus_count": self._concurrent_3plus_count,
            "concurrent_4plus_count": self._concurrent_4plus_count,
            "concurrent_5_count": self._concurrent_5_count,
            "last_tick_record": dict(self._last_tick_record) if self._last_tick_record else None,
            "aggregate_cap_hit_counts": dict(self._aggregate_cap_hit_counts),
        }

    # ── 内部: JSON構造化ログ出力 ─────────────────────────────────────

    def _emit_json(self, record: dict[str, Any]) -> None:
        """JSON構造化ログをログストリームに出力する。"""
        try:
            text = json.dumps(record, ensure_ascii=False, default=str)
            _logger.debug(text)
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass
