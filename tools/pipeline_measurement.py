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


# ── 知覚辞書カバレッジ計測 ──────────────────────────────────────

# 設計書: design_perception_coverage.md
#
# 知覚構造化の辞書照合における分類結果を事実記録する。
# 本機能はパイプライン計測基盤の拡張であり、以下の構造的分離を継承する:
# - psycheの内部状態・enrichment・判断・行動に一切参照されない
# - orchestratorのPhase処理に組み込まれない
# - enrichmentの項目として追加されない
# - save/loadの対象フィールドに一切追加しない
# - 辞書の追加・削除・変更を行わない(読み取り専用参照のみ)
# - 計測結果に基づいて知覚の振る舞いを動的に変更しない


_DEFAULT_COVERAGE_BUFFER_MAX = 200


class PerceptionCoverageRecord:
    """1回の知覚結果の分類記録。

    入力テキストは保持しない(安全弁7)。
    分類ラベルのみを記録する。
    """

    __slots__ = (
        "emotion_is_neutral",
        "intent_is_unknown",
        "keyword_hit",
        "llm_used",
        "timestamp",
    )

    def __init__(
        self,
        emotion_is_neutral: bool,
        intent_is_unknown: bool,
        keyword_hit: bool,
        llm_used: bool,
    ) -> None:
        self.emotion_is_neutral: bool = emotion_is_neutral
        self.intent_is_unknown: bool = intent_is_unknown
        self.keyword_hit: bool = keyword_hit
        self.llm_used: bool = llm_used
        self.timestamp: float = time.time()

    def to_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書表現を返す。"""
        return {
            "emotion_is_neutral": self.emotion_is_neutral,
            "intent_is_unknown": self.intent_is_unknown,
            "keyword_hit": self.keyword_hit,
            "llm_used": self.llm_used,
            "timestamp": self.timestamp,
        }


class PerceptionCoverageMeasurement:
    """知覚ヒューリスティック辞書の網羅性検証。

    知覚構造化の結果を分類し、辞書照合のカバレッジを事実として記録する。
    セッション開始時にインスタンスを生成し、セッション終了時に
    emit_coverage_summary()を呼んで累積を出力する。
    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - psycheの内部状態を一切変更しない
    - 計測値に基づく条件分岐を一切持たない
    - 辞書の追加・削除・変更を行わない
    - 出力はログストリームへのJSON書き込みと読み取り専用サマリのみ
    """

    def __init__(
        self,
        buffer_max: int = _DEFAULT_COVERAGE_BUFFER_MAX,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            buffer_max: FIFOバッファの上限サイズ(安全弁2)。
            enabled: 明示的な有効/無効指定。Noneの場合は
                     既存のCYRENE_MONITOR環境変数で判定(安全弁6)。
        """
        import os
        self._enabled = (
            enabled if enabled is not None
            else os.environ.get("CYRENE_MONITOR", "0") == "1"
        )

        # (A) FIFOバッファ(分類ラベルのみ、入力テキスト非保持: 安全弁7)
        self._buffer: deque[PerceptionCoverageRecord] = deque(
            maxlen=max(1, buffer_max)
        )

        # (B) セッション累積カウンタ(第2段)
        self._total_count: int = 0
        self._neutral_count: int = 0
        self._unknown_count: int = 0
        self._keyword_hit_count: int = 0
        self._llm_used_count: int = 0

    @property
    def enabled(self) -> bool:
        """計測が有効かどうか。"""
        return self._enabled

    @property
    def total_count(self) -> int:
        """総知覚回数。"""
        return self._total_count

    @property
    def neutral_count(self) -> int:
        """感情ラベルがneutralであった回数。"""
        return self._neutral_count

    @property
    def unknown_count(self) -> int:
        """意図ラベルがunknownであった回数。"""
        return self._unknown_count

    @property
    def keyword_hit_count(self) -> int:
        """キーワード照合で1つ以上の一致があった回数。"""
        return self._keyword_hit_count

    @property
    def llm_used_count(self) -> int:
        """LLM補助が利用された回数。"""
        return self._llm_used_count

    @property
    def buffer_size(self) -> int:
        """FIFOバッファ内の記録数。"""
        return len(self._buffer)

    # ── 第1段: 知覚結果の分類記録 ────────────────────────────────

    def record_perception(
        self,
        emotion: str,
        intent: str,
        topics: list[str],
        llm_used: bool,
    ) -> None:
        """知覚構造化の結果を分類し記録する。

        知覚構造化が完了するたびに呼ばれる。
        テキスト内容の意味解析・評価は行わない。
        入力テキストは受け取らない(安全弁7)。

        Args:
            emotion: 知覚構造化が出力した感情ラベル。
            intent: 知覚構造化が出力した意図ラベル。
            topics: 知覚構造化が出力した話題リスト。
            llm_used: LLM補助が利用されたか否か。
        """
        if not self._enabled:
            return
        try:
            emotion_is_neutral = (emotion == "neutral")
            intent_is_unknown = (intent == "unknown")
            keyword_hit = (len(topics) > 0)

            record = PerceptionCoverageRecord(
                emotion_is_neutral=emotion_is_neutral,
                intent_is_unknown=intent_is_unknown,
                keyword_hit=keyword_hit,
                llm_used=llm_used,
            )

            # FIFOバッファに追加
            self._buffer.append(record)

            # 第2段: 累積カウンタ更新
            self._total_count += 1
            if emotion_is_neutral:
                self._neutral_count += 1
            if intent_is_unknown:
                self._unknown_count += 1
            if keyword_hit:
                self._keyword_hit_count += 1
            if llm_used:
                self._llm_used_count += 1

            # ログ出力
            log_record = {
                "type": "perception_coverage",
                "timestamp": time.time(),
                "emotion_is_neutral": emotion_is_neutral,
                "intent_is_unknown": intent_is_unknown,
                "keyword_hit": keyword_hit,
                "llm_used": llm_used,
            }
            self._emit_json(log_record)

        except Exception:
            # 安全弁3: 計測失敗時の安全な無視
            pass

    # ── セッションサマリ出力 ──────────────────────────────────

    def emit_coverage_summary(self) -> None:
        """セッション終了時のカバレッジ計測累積情報を出力する。"""
        if not self._enabled:
            return
        try:
            record = {
                "type": "perception_coverage_session_summary",
                "timestamp": time.time(),
                "total_count": self._total_count,
                "neutral_count": self._neutral_count,
                "unknown_count": self._unknown_count,
                "keyword_hit_count": self._keyword_hit_count,
                "llm_used_count": self._llm_used_count,
                "buffer_size": len(self._buffer),
            }
            self._emit_json(record)
        except Exception:
            # 安全弁3
            pass

    # ── 第3段: 読み取り専用サマリ提供 ──────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """現在のカバレッジ計測サマリを読み取り専用で返す。

        Returns:
            計測サマリの辞書。
        """
        return {
            "total_count": self._total_count,
            "neutral_count": self._neutral_count,
            "unknown_count": self._unknown_count,
            "keyword_hit_count": self._keyword_hit_count,
            "llm_used_count": self._llm_used_count,
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
            # 安全弁3
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


# ── 知覚入力多様性記録 ────────────────────────────────────────────

# 設計書: design_perception_diversity.md
#
# 知覚パイプラインの入力と出力の構造的特性を事実として記録する。
# 本機能はパイプライン計測基盤の拡張であり、以下の構造的分離を継承する:
# - psycheの内部状態・enrichment・判断・行動に一切参照されない
# - orchestratorのPhase処理に組み込まれない
# - enrichmentの項目として追加されない
# - save/loadの対象フィールドに一切追加しない
# - 辞書の追加・削除・変更を行わない(読み取り専用参照のみ)
# - 記録内容に基づいて知覚の振る舞いを動的に調整しない
# - 入力テキストの全文を保持しない(テキスト長のみ記録)


_DEFAULT_DIVERSITY_BUFFER_MAX = 200


class PerceptionDiversityRecord:
    """1回の知覚結果の多様性断面記録。

    入力テキストは保持しない(安全弁3)。テキスト長(文字数)のみを記録する。
    """

    __slots__ = (
        "emotion_label",
        "intent_label",
        "topic_count",
        "input_length",
        "emotion_valence",
        "keyword_hit",
        "timestamp",
    )

    def __init__(
        self,
        emotion_label: str,
        intent_label: str,
        topic_count: int,
        input_length: int,
        emotion_valence: float,
        keyword_hit: bool,
    ) -> None:
        self.emotion_label: str = emotion_label
        self.intent_label: str = intent_label
        self.topic_count: int = topic_count
        self.input_length: int = input_length
        self.emotion_valence: float = emotion_valence
        self.keyword_hit: bool = keyword_hit
        self.timestamp: float = time.time()

    def to_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書表現を返す。"""
        return {
            "emotion_label": self.emotion_label,
            "intent_label": self.intent_label,
            "topic_count": self.topic_count,
            "input_length": self.input_length,
            "emotion_valence": round(self.emotion_valence, 4),
            "keyword_hit": self.keyword_hit,
            "timestamp": self.timestamp,
        }


class PerceptionDiversityMeasurement:
    """知覚パイプラインの入力多様性記録。

    知覚構造化が完了するたびに、出力されたラベル・話題数・入力長・感情価を
    事実として記録する。ラベル種類のセッション累積カウンタを保持する。

    セッション開始時にインスタンスを生成し、セッション終了時に
    emit_diversity_summary()を呼んで累積を出力する。
    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - psycheの内部状態を一切変更しない
    - 記録値に基づく条件分岐を一切持たない
    - 辞書の追加・削除・変更を行わない
    - 記録内容をenrichmentに露出しない
    - 出力はログストリームへのJSON書き込みと読み取り専用サマリのみ
    """

    def __init__(
        self,
        buffer_max: int = _DEFAULT_DIVERSITY_BUFFER_MAX,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            buffer_max: FIFOバッファの上限サイズ(安全弁1)。
            enabled: 明示的な有効/無効指定。Noneの場合は
                     既存のCYRENE_MONITOR環境変数で判定(安全弁5)。
        """
        import os
        self._enabled = (
            enabled if enabled is not None
            else os.environ.get("CYRENE_MONITOR", "0") == "1"
        )

        # (A) FIFOバッファ(入力テキスト非保持: 安全弁3)
        self._buffer: deque[PerceptionDiversityRecord] = deque(
            maxlen=max(1, buffer_max)
        )

        # (B) 感情ラベル累積カウンタ
        self._emotion_label_counts: dict[str, int] = {}

        # (C) 意図ラベル累積カウンタ
        self._intent_label_counts: dict[str, int] = {}

        # (D) 入力長の累積情報
        self._input_length_min: Optional[int] = None
        self._input_length_max: Optional[int] = None
        self._input_length_sum: int = 0
        self._input_length_count: int = 0

    @property
    def enabled(self) -> bool:
        """計測が有効かどうか。"""
        return self._enabled

    @property
    def buffer_size(self) -> int:
        """FIFOバッファ内の記録数。"""
        return len(self._buffer)

    @property
    def emotion_label_counts(self) -> dict[str, int]:
        """感情ラベル累積カウンタ(読み取り専用コピー)。"""
        return dict(self._emotion_label_counts)

    @property
    def intent_label_counts(self) -> dict[str, int]:
        """意図ラベル累積カウンタ(読み取り専用コピー)。"""
        return dict(self._intent_label_counts)

    @property
    def total_count(self) -> int:
        """総記録回数。"""
        return self._input_length_count

    # ── 知覚結果の断面記録 ──────────────────────────────────────

    def record_perception_diversity(
        self,
        emotion_label: str,
        intent_label: str,
        topic_count: int,
        input_length: int,
        emotion_valence: float,
        keyword_hit: bool,
    ) -> None:
        """知覚構造化の結果を多様性断面として記録する。

        知覚構造化が完了するたびに呼ばれる。
        入力テキストの全文は受け取らない(安全弁3)。

        Args:
            emotion_label: 知覚構造化が出力した感情ラベル。
            intent_label: 知覚構造化が出力した意図ラベル。
            topic_count: 知覚構造化が出力した話題リストの要素数。
            input_length: 入力テキストの長さ(文字数)。
            emotion_valence: 感情価の数値。
            keyword_hit: 辞書照合によるキーワード一致の有無。
        """
        if not self._enabled:
            return
        try:
            record = PerceptionDiversityRecord(
                emotion_label=emotion_label,
                intent_label=intent_label,
                topic_count=topic_count,
                input_length=input_length,
                emotion_valence=emotion_valence,
                keyword_hit=keyword_hit,
            )

            # FIFOバッファに追加
            self._buffer.append(record)

            # 感情ラベル累積カウンタ更新
            self._emotion_label_counts[emotion_label] = (
                self._emotion_label_counts.get(emotion_label, 0) + 1
            )

            # 意図ラベル累積カウンタ更新
            self._intent_label_counts[intent_label] = (
                self._intent_label_counts.get(intent_label, 0) + 1
            )

            # 入力長の累積情報更新
            if self._input_length_min is None or input_length < self._input_length_min:
                self._input_length_min = input_length
            if self._input_length_max is None or input_length > self._input_length_max:
                self._input_length_max = input_length
            self._input_length_sum += input_length
            self._input_length_count += 1

            # ログ出力
            log_record = {
                "type": "perception_diversity",
                "timestamp": time.time(),
                "emotion_label": emotion_label,
                "intent_label": intent_label,
                "topic_count": topic_count,
                "input_length": input_length,
                "emotion_valence": round(emotion_valence, 4),
                "keyword_hit": keyword_hit,
            }
            self._emit_json(log_record)

        except Exception:
            # 安全弁2: 記録失敗時の安全な無視
            pass

    # ── セッションサマリ出力 ────────────────────────────────────

    def emit_diversity_summary(self) -> None:
        """セッション終了時の多様性記録累積情報を出力する。"""
        if not self._enabled:
            return
        try:
            avg_length = (
                self._input_length_sum / self._input_length_count
                if self._input_length_count > 0
                else 0.0
            )
            record = {
                "type": "perception_diversity_session_summary",
                "timestamp": time.time(),
                "total_count": self._input_length_count,
                "emotion_label_counts": dict(self._emotion_label_counts),
                "intent_label_counts": dict(self._intent_label_counts),
                "emotion_label_unique_count": len(self._emotion_label_counts),
                "intent_label_unique_count": len(self._intent_label_counts),
                "input_length_min": self._input_length_min,
                "input_length_max": self._input_length_max,
                "input_length_avg": round(avg_length, 2),
                "buffer_size": len(self._buffer),
            }
            self._emit_json(record)
        except Exception:
            # 安全弁2
            pass

    # ── 読み取り専用サマリ提供 ──────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """現在の多様性記録サマリを読み取り専用で返す。

        Returns:
            記録サマリの辞書。
        """
        avg_length = (
            self._input_length_sum / self._input_length_count
            if self._input_length_count > 0
            else 0.0
        )
        return {
            "total_count": self._input_length_count,
            "emotion_label_counts": dict(self._emotion_label_counts),
            "intent_label_counts": dict(self._intent_label_counts),
            "emotion_label_unique_count": len(self._emotion_label_counts),
            "intent_label_unique_count": len(self._intent_label_counts),
            "input_length_min": self._input_length_min,
            "input_length_max": self._input_length_max,
            "input_length_avg": round(avg_length, 2),
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
            # 安全弁2
            pass
