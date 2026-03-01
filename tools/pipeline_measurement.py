"""
tools/pipeline_measurement.py - 知覚-判断-代弁パイプラインの計測

設計書: design_pipeline_measurement.md

パイプライン全体（外部APIコール＋内部処理）の時間経過を事実記録する。
3つの入力経路（画面知覚/テキスト対話/自発起動）ごとの時間特性の差異を記録する。

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)のみ
- psycheの内部状態・enrichment・判断・行動に一切参照されない
- 計測値に基づくフェーズのスキップ・短縮・並列化を行わない
- 計測結果に基づいて処理のスキップ・短縮・並列化を行わない
- orchestratorのPhase処理に組み込まれない
- enrichmentの項目として追加されない
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない

安全弁:
1. 計測失敗時の安全な無視(例外を捕捉しスキップ、本体処理続行)
2. FIFOバッファの上限(メモリ消費を制限)
3. ログ出力量の制限(既存の実行時観測基盤の制限機構に従う)
4. 永続化の非対象(save/loadフィールド追加なし、セッション境界で消失)
5. 環境変数による完全無効化(CYRENE_MONITOR=0で無効化)
6. psyche非参照(計測結果を参照するpsycheモジュール・enrichment項目が存在しない)
7. 思考エンジン側コード変更の最小化(with文と記録呼び出しのみ)
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Optional

# 独自ログ名前空間(既存のmonitorログと統合)
monitor_logger = logging.getLogger("cyrene.monitor")


# ── FIFOバッファのデフォルト上限(安全弁2) ────────────────────────
_DEFAULT_PIPELINE_BUFFER_MAX = 200


# ── フェーズ区間識別子 ──────────────────────────────────────────
# パイプライン内の各フェーズ区間の名前。
# 設計書の計測対象に対応。
PHASE_PERCEPTION_API = "perception_api"        # 知覚コール区間(外部API)
PHASE_PERCEPTION_PARSE = "perception_parse"    # 知覚構造化区間
PHASE_PSYCHE_UPDATE = "psyche_update"          # psyche全フェーズ更新区間
PHASE_MEMORY_RECALL = "memory_recall"          # 記憶検索区間
PHASE_POLICY_SELECT = "policy_select"          # 方針選択区間
PHASE_EXPRESSION_API = "expression_api"        # 代弁コール区間(外部API)
PHASE_PIPELINE_TOTAL = "pipeline_total"        # パイプライン全体区間


# ── 入力経路識別子 ──────────────────────────────────────────────
PATHWAY_VISION = "vision"
PATHWAY_TEXT = "text"
PATHWAY_INTERNAL = "internal"


# ── PipelineRecord ──────────────────────────────────────────────


class PipelineRecord:
    """1パイプライン実行の計測記録。

    各フェーズ区間の経過時間を保持する。
    存在しないフェーズ区間は記録なし(辞書に含まれない)。
    """

    __slots__ = ("pathway", "phase_times", "total_time", "timestamp")

    def __init__(self, pathway: str) -> None:
        self.pathway: str = pathway
        self.phase_times: dict[str, float] = {}
        self.total_time: float = 0.0
        self.timestamp: float = time.time()

    def to_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書表現を返す。"""
        return {
            "pathway": self.pathway,
            "phase_times": dict(self.phase_times),
            "total_time": round(self.total_time, 6),
            "timestamp": self.timestamp,
        }


# ── PipelineMeasurement ────────────────────────────────────────


class PipelineMeasurement:
    """知覚-判断-代弁パイプラインの計測。

    セッション開始時にインスタンスを生成し、セッション終了時に
    emit_pipeline_summary()を呼んで累積を出力する。
    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - psycheの内部状態を一切変更しない
    - 計測値に基づく条件分岐を一切持たない
    - 出力はログストリームへのJSON書き込みのみ
    """

    def __init__(
        self,
        buffer_max: int = _DEFAULT_PIPELINE_BUFFER_MAX,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            buffer_max: FIFOバッファの上限サイズ(安全弁2)。
            enabled: 明示的な有効/無効指定。Noneの場合は
                     既存のCYRENE_MONITOR環境変数で判定。
        """
        import os
        self._enabled = (
            enabled if enabled is not None
            else os.environ.get("CYRENE_MONITOR", "0") == "1"
        )

        # (A) パイプライン記録のFIFOバッファ
        self._buffer: deque[PipelineRecord] = deque(
            maxlen=max(1, buffer_max)
        )

        # (B) セッション累積カウンタ
        # 経路別のパイプライン実行回数
        self._pathway_count: dict[str, int] = {}
        # 経路別・フェーズ別の累積経過時間
        self._pathway_phase_cumulative: dict[str, dict[str, float]] = {}
        # 経路別のパイプライン全体の累積経過時間
        self._pathway_total_cumulative: dict[str, float] = {}

        # (C) 現在進行中のパイプライン計測(1実行分のみ)
        self._current_record: Optional[PipelineRecord] = None
        self._pipeline_start: float = 0.0

    @property
    def enabled(self) -> bool:
        """計測が有効かどうか。"""
        return self._enabled

    @property
    def buffer(self) -> list[dict[str, Any]]:
        """FIFOバッファの内容(読み取り専用コピー)。"""
        return [r.to_dict() for r in self._buffer]

    @property
    def pathway_count(self) -> dict[str, int]:
        """経路別のパイプライン実行回数(読み取り専用コピー)。"""
        return dict(self._pathway_count)

    @property
    def pathway_phase_cumulative(self) -> dict[str, dict[str, float]]:
        """経路別・フェーズ別の累積経過時間(読み取り専用コピー)。"""
        return {k: dict(v) for k, v in self._pathway_phase_cumulative.items()}

    @property
    def pathway_total_cumulative(self) -> dict[str, float]:
        """経路別のパイプライン全体の累積経過時間(読み取り専用コピー)。"""
        return dict(self._pathway_total_cumulative)

    @property
    def record_count(self) -> int:
        """FIFOバッファ内の記録数。"""
        return len(self._buffer)

    # ── パイプライン開始/完了 ──────────────────────────────────

    def begin_pipeline(self, pathway: str) -> None:
        """パイプライン計測を開始する。

        Args:
            pathway: 入力経路識別子("vision"/"text"/"internal")
        """
        if not self._enabled:
            return
        try:
            self._current_record = PipelineRecord(pathway)
            self._pipeline_start = time.perf_counter()
        except Exception:
            # 安全弁1
            self._current_record = None

    def end_pipeline(self) -> None:
        """パイプライン計測を完了し、記録を蓄積する。

        FIFOバッファへの追加、セッション累積カウンタの更新、
        ログストリームへのJSON出力を行う。
        """
        if not self._enabled:
            return
        try:
            record = self._current_record
            if record is None:
                return

            # パイプライン全体の経過時間
            if self._pipeline_start > 0:
                record.total_time = time.perf_counter() - self._pipeline_start
                record.phase_times[PHASE_PIPELINE_TOTAL] = record.total_time

            # FIFOバッファに追加
            self._buffer.append(record)

            pathway = record.pathway

            # セッション累積カウンタ更新
            self._pathway_count[pathway] = (
                self._pathway_count.get(pathway, 0) + 1
            )
            self._pathway_total_cumulative[pathway] = (
                self._pathway_total_cumulative.get(pathway, 0.0)
                + record.total_time
            )

            if pathway not in self._pathway_phase_cumulative:
                self._pathway_phase_cumulative[pathway] = {}
            phase_cum = self._pathway_phase_cumulative[pathway]
            for phase_name, elapsed in record.phase_times.items():
                phase_cum[phase_name] = phase_cum.get(phase_name, 0.0) + elapsed

            # ログ出力
            log_record = {
                "type": "pipeline_complete",
                "timestamp": time.time(),
                "pathway": pathway,
                "phase_times": {
                    k: round(v, 6) for k, v in record.phase_times.items()
                },
                "total_time": round(record.total_time, 6),
            }
            self._emit_json(log_record)

            # 現在の記録をクリア
            self._current_record = None
            self._pipeline_start = 0.0

        except Exception:
            # 安全弁1
            self._current_record = None
            self._pipeline_start = 0.0

    # ── フェーズ区間の記録 ────────────────────────────────────

    def record_phase(self, phase_name: str, elapsed: float) -> None:
        """フェーズ区間の経過時間を記録する。

        PhaseTimerコンテキストマネージャから呼ばれる。

        Args:
            phase_name: フェーズ区間識別子
            elapsed: 経過時間(秒)
        """
        if not self._enabled:
            return
        try:
            if self._current_record is not None:
                self._current_record.phase_times[phase_name] = elapsed
        except Exception:
            # 安全弁1
            pass

    # ── セッションサマリ出力 ──────────────────────────────────

    def emit_pipeline_summary(self) -> None:
        """セッション終了時のパイプライン計測累積情報を出力する。"""
        if not self._enabled:
            return
        try:
            record = {
                "type": "pipeline_session_summary",
                "timestamp": time.time(),
                "pathway_counts": dict(self._pathway_count),
                "pathway_total_cumulative": {
                    k: round(v, 6)
                    for k, v in self._pathway_total_cumulative.items()
                },
                "pathway_phase_cumulative": {
                    pathway: {
                        phase: round(t, 6) for phase, t in phases.items()
                    }
                    for pathway, phases
                    in self._pathway_phase_cumulative.items()
                },
                "buffer_size": len(self._buffer),
            }
            self._emit_json(record)
        except Exception:
            # 安全弁1
            pass

    # ── 読み取り専用アクセサ(シミュレータ等向け) ───────────────

    def get_summary(self) -> dict[str, Any]:
        """現在の計測サマリを読み取り専用で返す。

        外部ツール(シミュレータ等)が呼び出す読み取り専用アクセサ。

        Returns:
            計測サマリの辞書。
        """
        return {
            "pathway_counts": dict(self._pathway_count),
            "pathway_total_cumulative": {
                k: round(v, 6)
                for k, v in self._pathway_total_cumulative.items()
            },
            "pathway_phase_cumulative": {
                pathway: {
                    phase: round(t, 6) for phase, t in phases.items()
                }
                for pathway, phases
                in self._pathway_phase_cumulative.items()
            },
            "buffer_size": len(self._buffer),
            "latest_record": (
                self._buffer[-1].to_dict() if self._buffer else None
            ),
        }

    # ── 内部: JSON構造化ログ出力 ─────────────────────────────

    def _emit_json(self, record: dict[str, Any]) -> None:
        """JSON構造化ログをログストリームに出力する。"""
        try:
            text = json.dumps(record, ensure_ascii=False, default=str)
            monitor_logger.debug(text)
        except Exception:
            # 安全弁1
            pass


# ── PhaseTimer: フェーズ区間計測コンテキストマネージャ ──────────


class PhaseTimer:
    """パイプラインフェーズ区間の経過時間を計測するコンテキストマネージャ。

    既存のBandTimerと同じ設計パターン。
    思考エンジンの各フェーズをwith文で囲むだけで計測が完了する。

    with PhaseTimer(measurement, "perception_api"):
        ... 知覚APIコール ...

    計測点自体が例外を発生させた場合、元の処理に影響を与えない(安全弁1)。
    """

    def __init__(
        self,
        measurement: Optional[PipelineMeasurement],
        phase_name: str,
    ) -> None:
        self._measurement = measurement
        self._phase_name = phase_name
        self._start: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        try:
            if self._measurement and self._measurement.enabled:
                self._start = time.perf_counter()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if (
                self._measurement
                and self._measurement.enabled
                and self._start > 0
            ):
                elapsed = time.perf_counter() - self._start
                self._measurement.record_phase(self._phase_name, elapsed)
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass
        # 元の例外を再送出(Falseを返す = 例外を抑制しない)
        return None
