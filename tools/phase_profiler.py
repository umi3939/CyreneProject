"""
tools/phase_profiler.py - Phase単位実行時間プロファイリング

設計書: design_phase_profiling.md

帯域内部のPhase単位の実行時間を事実として記録する。
既存の帯域計測基盤(BandTimer)と並行動作し、帯域記録を置換しない。
Phase単位の計測は帯域計測の内訳情報として追加される。

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)と読み取り専用アクセサのみ
- psycheの内部状態・enrichment・判断・行動に一切参照されない
- 計測結果に基づいてPhaseの実行順序・スキップ・並列化を行わない
- 「遅い」「速い」「ボトルネック」等の評価的判定を行わない
- Phaseの実行可否を計測結果から決定しない
- 計測結果をpsycheの内部状態・enrichment・判断に参照させない
- 計測結果をPhase宣言の依存関係や実行エンジンに帰還させない
- enrichment生成の所要時間に基づいてenrichment項目を削減しない
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない

安全弁:
1. 計測失敗時の安全な無視(例外を捕捉しスキップ、元のPhase実行を中断しない)
2. FIFOバッファの上限(記録バッファは固定長FIFOとし、メモリ消費を制限)
3. 環境変数による完全無効化(既存のCYRENE_MONITOR制御機構に従う)
4. 永続化の非対象(save/loadフィールド追加なし、セッション境界で消失)
5. psyche非参照(計測結果を参照するpsycheモジュール・enrichment項目・帰還経路が存在しない)
6. 計測オーバーヘッドの最小化(時刻取得と辞書への記録のみ。無効化時は時刻取得すら行わない)
7. ログ出力量の制限(既存の実行時観測基盤のログ出力量制限機構に従う)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Optional

# 独自ログ名前空間(既存のmonitorログと統合)
monitor_logger = logging.getLogger("cyrene.monitor")


# ── 環境変数制御 ──────────────────────────────────────────────────

def _is_profiler_enabled() -> bool:
    """プロファイリングが有効かどうかを実行時に判定する。

    既存のCYRENE_MONITOR環境変数制御機構に従う(安全弁3)。
    インポート時ではなく呼び出し時に環境変数を確認する。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── FIFOバッファのデフォルト上限(安全弁2) ────────────────────────

_DEFAULT_TICK_BUFFER_MAX = 100


# ── PhaseTimingRecord ─────────────────────────────────────────────


class PhaseTimingRecord:
    """1つのPhase実行の計測記録。

    ティック番号・帯域名・Phase名・経過時間の組を保持する。
    """

    __slots__ = ("tick", "band_name", "phase_name", "elapsed")

    def __init__(
        self,
        tick: int,
        band_name: str,
        phase_name: str,
        elapsed: float,
    ) -> None:
        self.tick: int = tick
        self.band_name: str = band_name
        self.phase_name: str = phase_name
        self.elapsed: float = elapsed

    def to_dict(self) -> dict[str, Any]:
        """辞書表現を返す。"""
        return {
            "tick": self.tick,
            "band_name": self.band_name,
            "phase_name": self.phase_name,
            "elapsed": round(self.elapsed, 6),
        }


# ── TickProfile ───────────────────────────────────────────────────


class TickProfile:
    """1ティック分のPhase実行時間内訳。

    FIFOバッファの1エントリとして使用される。
    """

    __slots__ = ("tick", "records", "enrichment_elapsed", "timestamp")

    def __init__(self, tick: int) -> None:
        self.tick: int = tick
        self.records: list[PhaseTimingRecord] = []
        self.enrichment_elapsed: Optional[float] = None
        self.timestamp: float = time.time()

    def to_dict(self) -> dict[str, Any]:
        """辞書表現を返す。"""
        result: dict[str, Any] = {
            "tick": self.tick,
            "records": [r.to_dict() for r in self.records],
            "timestamp": self.timestamp,
        }
        if self.enrichment_elapsed is not None:
            result["enrichment_elapsed"] = round(self.enrichment_elapsed, 6)
        return result


# ── PhaseProfiler ─────────────────────────────────────────────────


class PhaseProfiler:
    """Phase単位実行時間プロファイリング。

    Phase実行エンジンが各Phaseのハンドラを実行する際、実行前後の
    時刻差を記録する。計測はPhase実行エンジンの呼び出し構造内で行い、
    個別のPhaseハンドラ内部には計測コードを挿入しない。

    セッション開始時にインスタンスを生成し、セッション終了時に
    emit_session_summary()を呼んで累積を出力する。
    全内部状態はインスタンス破棄時に消失する(永続化対象外: 安全弁4)。

    本クラスの全メソッドは:
    - psycheの内部状態を一切変更しない
    - 計測値に基づく条件分岐を一切持たない(蓄積と参照のみ)
    - 出力はログストリームへのJSON書き込みと読み取り専用アクセサのみ
    - Phase実行の可否・順序・帯域設定を変更しない
    - enrichment項目の生成・削除・圧縮設定を変更しない
    """

    def __init__(
        self,
        tick_buffer_max: int = _DEFAULT_TICK_BUFFER_MAX,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            tick_buffer_max: ティック別FIFOバッファの上限サイズ(安全弁2)。
            enabled: 明示的な有効/無効指定。Noneの場合は
                     既存のCYRENE_MONITOR環境変数で判定(安全弁3)。
        """
        # 有効/無効判定(明示指定 > 環境変数)
        self._enabled = (
            enabled if enabled is not None
            else _is_profiler_enabled()
        )

        # (1) ティック別記録のFIFOバッファ(安全弁2)
        self._tick_buffer: deque[TickProfile] = deque(
            maxlen=max(1, tick_buffer_max)
        )

        # (2) セッション累積カウンタ群
        # Phase名→累積実行時間
        self._phase_cumulative_time: dict[str, float] = {}
        # Phase名→呼び出し回数
        self._phase_call_count: dict[str, int] = {}
        # 帯域名→累積実行時間(帯域内全Phase合算)
        self._band_cumulative_time: dict[str, float] = {}
        # enrichment生成の累積実行時間と呼び出し回数
        self._enrichment_cumulative_time: float = 0.0
        self._enrichment_call_count: int = 0

        # (3) 直近1ティックの内訳記録(ダッシュボード等の即時参照用)
        self._current_tick_profile: Optional[TickProfile] = None

        # セッション累積ティック数
        self._total_ticks_profiled: int = 0

    @property
    def enabled(self) -> bool:
        """プロファイリングが有効かどうか。"""
        return self._enabled

    @property
    def total_ticks_profiled(self) -> int:
        """プロファイリングされたティック総数。"""
        return self._total_ticks_profiled

    @property
    def phase_cumulative_time(self) -> dict[str, float]:
        """Phase名別の累積実行時間(読み取り専用コピー)。"""
        return dict(self._phase_cumulative_time)

    @property
    def phase_call_count(self) -> dict[str, int]:
        """Phase名別の呼び出し回数(読み取り専用コピー)。"""
        return dict(self._phase_call_count)

    @property
    def band_cumulative_time(self) -> dict[str, float]:
        """帯域名別の累積実行時間(読み取り専用コピー)。"""
        return dict(self._band_cumulative_time)

    @property
    def enrichment_cumulative_time(self) -> float:
        """enrichment生成の累積実行時間。"""
        return self._enrichment_cumulative_time

    @property
    def enrichment_call_count(self) -> int:
        """enrichment生成の呼び出し回数。"""
        return self._enrichment_call_count

    # ── ティック開始/完了 ──────────────────────────────────────────

    def begin_tick(self, tick: int) -> None:
        """ティック計測を開始する。

        各ティックの処理開始時に呼び出される。
        前回のcurrent_tick_profileを保持したままの場合はFIFOに移動する。

        Args:
            tick: ティック番号
        """
        if not self._enabled:
            return
        try:
            # 前回のティックプロファイルがあればFIFOに移動
            if self._current_tick_profile is not None:
                self._tick_buffer.append(self._current_tick_profile)
            self._current_tick_profile = TickProfile(tick)
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            self._current_tick_profile = None

    def end_tick(self) -> None:
        """ティック計測を完了する。

        ティック処理完了時に呼び出される。
        現在のティックプロファイルをFIFOバッファに追加し、
        ログストリームに出力する。
        """
        if not self._enabled:
            return
        try:
            profile = self._current_tick_profile
            if profile is None:
                return

            self._total_ticks_profiled += 1

            # FIFOバッファに追加(安全弁2: 上限到達時に最古が自動消失)
            self._tick_buffer.append(profile)
            self._current_tick_profile = None

            # ログ出力
            self._emit_tick_profile(profile)
        except Exception:
            # 安全弁1
            self._current_tick_profile = None

    # ── Phase実行時間の記録 ────────────────────────────────────────

    def record_phase(
        self,
        band_name: str,
        phase_name: str,
        elapsed: float,
        tick: int,
    ) -> None:
        """1つのPhase実行時間を記録する。

        Phase実行エンジンのハンドラ呼び出し前後で計測された経過時間を受け取る。
        計測はPhase実行エンジン内のコンテキストマネージャから呼ばれる。

        Args:
            band_name: 帯域名
            phase_name: Phase識別子
            elapsed: 経過時間(秒)
            tick: ティック番号
        """
        if not self._enabled:
            return
        try:
            record = PhaseTimingRecord(
                tick=tick,
                band_name=band_name,
                phase_name=phase_name,
                elapsed=elapsed,
            )

            # 現在のティックプロファイルに追加
            if self._current_tick_profile is not None:
                self._current_tick_profile.records.append(record)

            # セッション累積カウンタ更新
            self._phase_cumulative_time[phase_name] = (
                self._phase_cumulative_time.get(phase_name, 0.0) + elapsed
            )
            self._phase_call_count[phase_name] = (
                self._phase_call_count.get(phase_name, 0) + 1
            )
            self._band_cumulative_time[band_name] = (
                self._band_cumulative_time.get(band_name, 0.0) + elapsed
            )
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    # ── enrichment生成時間の記録 ───────────────────────────────────

    def record_enrichment(self, elapsed: float) -> None:
        """enrichment生成の所要時間を記録する。

        enrichment生成の開始から完了までの経過時間を1つの値として記録する。
        項目個別の計測は行わない。

        Args:
            elapsed: enrichment生成の経過時間(秒)
        """
        if not self._enabled:
            return
        try:
            # 現在のティックプロファイルに記録
            if self._current_tick_profile is not None:
                self._current_tick_profile.enrichment_elapsed = elapsed

            # セッション累積カウンタ更新
            self._enrichment_cumulative_time += elapsed
            self._enrichment_call_count += 1
        except Exception:
            # 安全弁1
            pass

    # ── 読み取り専用アクセサ ──────────────────────────────────────

    def get_latest_tick_profile(self) -> Optional[dict[str, Any]]:
        """直近1ティックの内訳記録を返す(ダッシュボード等の即時参照用)。

        Returns:
            直近のTickProfileの辞書表現。記録がない場合はNone。
        """
        try:
            if self._tick_buffer:
                return self._tick_buffer[-1].to_dict()
            return None
        except Exception:
            return None

    def get_summary(self) -> dict[str, Any]:
        """現在のプロファイリングサマリを読み取り専用で返す。

        外部ツール(ダッシュボード等)が呼び出す読み取り専用アクセサ。

        Returns:
            プロファイリングサマリの辞書。
        """
        try:
            # Phase別の平均実行時間を計算
            phase_avg_time: dict[str, float] = {}
            for phase_name, cum_time in self._phase_cumulative_time.items():
                count = self._phase_call_count.get(phase_name, 1)
                if count > 0:
                    phase_avg_time[phase_name] = round(cum_time / count, 6)

            # enrichment平均実行時間
            enrichment_avg_time = (
                round(
                    self._enrichment_cumulative_time / self._enrichment_call_count,
                    6,
                )
                if self._enrichment_call_count > 0
                else 0.0
            )

            return {
                "total_ticks_profiled": self._total_ticks_profiled,
                "phase_cumulative_time": {
                    k: round(v, 6) for k, v in self._phase_cumulative_time.items()
                },
                "phase_call_count": dict(self._phase_call_count),
                "phase_avg_time": phase_avg_time,
                "band_cumulative_time": {
                    k: round(v, 6) for k, v in self._band_cumulative_time.items()
                },
                "enrichment_cumulative_time": round(
                    self._enrichment_cumulative_time, 6
                ),
                "enrichment_call_count": self._enrichment_call_count,
                "enrichment_avg_time": enrichment_avg_time,
                "tick_buffer_size": len(self._tick_buffer),
                "latest_tick_profile": self.get_latest_tick_profile(),
            }
        except Exception:
            return {}

    # ── セッションサマリの出力 ────────────────────────────────────

    def emit_session_summary(self) -> None:
        """セッション終了時の累積プロファイリング情報を出力する。"""
        if not self._enabled:
            return
        try:
            summary = self.get_summary()
            record = {
                "type": "phase_profiling_session_summary",
                "timestamp": time.time(),
                **summary,
            }
            self._emit_json(record)
        except Exception:
            # 安全弁1
            pass

    # ── 内部: ティックプロファイルのログ出力 ──────────────────────

    def _emit_tick_profile(self, profile: TickProfile) -> None:
        """1ティック分のPhase実行時間内訳をログ出力する。

        安全弁7: 既存の実行時観測基盤のログ出力量制限機構に従う。
        """
        try:
            # 帯域別に集約
            band_breakdown: dict[str, list[dict[str, Any]]] = {}
            band_totals: dict[str, float] = {}
            for rec in profile.records:
                if rec.band_name not in band_breakdown:
                    band_breakdown[rec.band_name] = []
                    band_totals[rec.band_name] = 0.0
                band_breakdown[rec.band_name].append({
                    "phase": rec.phase_name,
                    "elapsed": round(rec.elapsed, 6),
                })
                band_totals[rec.band_name] += rec.elapsed

            record: dict[str, Any] = {
                "type": "phase_profiling_tick",
                "timestamp": time.time(),
                "tick": profile.tick,
                "band_breakdown": band_breakdown,
                "band_totals": {
                    k: round(v, 6) for k, v in band_totals.items()
                },
            }

            if profile.enrichment_elapsed is not None:
                record["enrichment_elapsed"] = round(
                    profile.enrichment_elapsed, 6
                )

            self._emit_json(record)
        except Exception:
            # 安全弁1
            pass

    # ── 内部: JSON構造化ログ出力 ──────────────────────────────────

    def _emit_json(self, record: dict[str, Any]) -> None:
        """JSON構造化ログをログストリームに出力する。

        安全弁7: 既存の実行時観測基盤のログ出力量制限機構に従う。
        """
        try:
            text = json.dumps(record, ensure_ascii=False, default=str)
            monitor_logger.debug(text)
        except Exception:
            # 安全弁1
            pass


# ── PhaseProfileTimer: Phase計測コンテキストマネージャ ─────────────


class PhaseProfileTimer:
    """Phase単位の実行時間を計測するコンテキストマネージャ。

    Phase実行エンジンのexecute_band()内で各Phaseハンドラの実行を囲む。
    既存のBandTimerと同じ設計パターン。

    with PhaseProfileTimer(profiler, "every_tick", "phase_1", tick):
        handler(orchestrator, user_id)

    計測点自体が例外を発生させた場合、元のPhase実行に影響を与えない(安全弁1)。
    無効化時は時刻取得すら行わない(安全弁6)。
    """

    def __init__(
        self,
        profiler: Optional[PhaseProfiler],
        band_name: str,
        phase_name: str,
        tick: int,
    ) -> None:
        self._profiler = profiler
        self._band_name = band_name
        self._phase_name = phase_name
        self._tick = tick
        self._start: float = 0.0

    def __enter__(self) -> "PhaseProfileTimer":
        try:
            # 安全弁6: 無効化時は時刻取得すら行わない
            if self._profiler and self._profiler.enabled:
                self._start = time.perf_counter()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if (
                self._profiler
                and self._profiler.enabled
                and self._start > 0
            ):
                elapsed = time.perf_counter() - self._start
                self._profiler.record_phase(
                    band_name=self._band_name,
                    phase_name=self._phase_name,
                    elapsed=elapsed,
                    tick=self._tick,
                )
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass
        # 元の例外を再送出(Falseを返す = 例外を抑制しない)
        return None


# ── EnrichmentProfileTimer: enrichment生成計測コンテキストマネージャ ──


class EnrichmentProfileTimer:
    """enrichment生成の実行時間を計測するコンテキストマネージャ。

    enrichment生成関数の呼び出し前後を囲む。
    既存のBandTimerと同じ設計パターン。

    with EnrichmentProfileTimer(profiler):
        enrichment_text = get_prompt_enrichment(...)

    計測点自体が例外を発生させた場合、元の処理に影響を与えない(安全弁1)。
    無効化時は時刻取得すら行わない(安全弁6)。
    """

    def __init__(
        self,
        profiler: Optional[PhaseProfiler],
    ) -> None:
        self._profiler = profiler
        self._start: float = 0.0

    def __enter__(self) -> "EnrichmentProfileTimer":
        try:
            # 安全弁6: 無効化時は時刻取得すら行わない
            if self._profiler and self._profiler.enabled:
                self._start = time.perf_counter()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if (
                self._profiler
                and self._profiler.enabled
                and self._start > 0
            ):
                elapsed = time.perf_counter() - self._start
                self._profiler.record_enrichment(elapsed)
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass
        # 元の例外を再送出(Falseを返す = 例外を抑制しない)
        return None
