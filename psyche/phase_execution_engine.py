"""
psyche/phase_execution_engine.py - 宣言的Phase実行エンジン（段階2プロトタイプ）

宣言的定義（phase_declaration.py）に基づいてPhase実行を駆動する実行エンジン。
段階2では10ティック帯域（Phase 27/28/29）のみを対象とする。

設計原則:
- Phase処理ロジック自体は変更しない。「呼び出し方」のみを標準化する
- 既存の手続き的コードをフォールバック経路として保持する
- 有効フラグにより宣言的実行と手続き的実行を任意に切替可能
- save/load非影響: 内部状態は永続化しない
- enrichment非接続: 内部状態はenrichmentに接続しない
- 帯域限定: 10ティック帯域以外のPhase定義を参照する経路を持たない
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from .phase_declaration import (
    Band,
    BAND_EVERY_10_TICKS,
    PHASE_BY_ID,
    PhaseDefinition,
)

if TYPE_CHECKING:
    pass  # PsycheOrchestrator is not imported to avoid circular deps

logger = logging.getLogger(__name__)


# ── Phase実行結果 ──────────────────────────────────────────────


class PhaseStatus:
    """Phase実行の成否を表す列挙的定数。"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseExecutionLog:
    """最新帯域実行のPhase別成否ログ。

    デバッグ・検証目的でのみ使用する。
    enrichmentやpsyche内部状態には一切接続しない。
    上書き型であり、蓄積されない。
    """
    phase_results: dict[str, str] = field(default_factory=dict)
    # phase_id -> PhaseStatus.SUCCESS/FAILED/SKIPPED


# ── Phase処理関数の型 ──────────────────────────────────────────


# Phase処理関数: orchestratorインスタンスとuser_idを受け取り、戻り値なし
PhaseHandler = Callable[[Any, str], None]


# ── Phase実行エンジン ──────────────────────────────────────────


class PhaseExecutionEngine:
    """宣言的定義に基づくPhase実行エンジン。

    10ティック帯域（Phase 27/28/29）のみを対象とする。
    他帯域のPhase定義を参照する経路を持たない。

    Attributes:
        _enabled: 実行エンジン有効フラグ。無効時は呼び出し元がフォールバックする
        _handler_table: Phase識別子→処理関数の対応表
        _band_phase_ids: 10ティック帯域の所属Phase一覧（帯域内順序）
        _last_log: 最新実行のPhase別成否ログ
    """

    def __init__(self) -> None:
        """実行エンジンを初期化する。

        登録テーブルは空の状態で作成される。
        register_handler() で処理関数を登録した後に execute_band() を呼ぶ。
        """
        self._enabled: bool = True
        self._handler_table: dict[str, PhaseHandler] = {}
        self._band_phase_ids: tuple[str, ...] = BAND_EVERY_10_TICKS.phase_ids
        self._last_log: PhaseExecutionLog = PhaseExecutionLog()

    # ── 有効フラグ制御 ──────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """実行エンジンの有効/無効状態を返す。"""
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        """実行エンジンの有効/無効を切り替える。

        Phase処理結果に基づいて自動変化する経路は存在しない。
        外部からの明示的切替のみで変化する。
        """
        self._enabled = value

    # ── Phase処理関数の登録 ──────────────────────────────────────

    def register_handler(self, phase_id: str, handler: PhaseHandler) -> None:
        """Phase処理関数を登録する。

        登録は初期化時に一度だけ行われることを想定する。
        登録対象は10ティック帯域のPhaseのみ。

        Args:
            phase_id: Phase識別子（宣言的定義のphase_idと一致）
            handler: 該当Phaseの処理内容を実行する関数

        Raises:
            ValueError: 指定phase_idが10ティック帯域に所属しない場合
        """
        if phase_id not in self._band_phase_ids:
            raise ValueError(
                f"Phase '{phase_id}' is not in the 10-tick band. "
                f"Valid phase IDs: {self._band_phase_ids}"
            )
        self._handler_table[phase_id] = handler

    def is_fully_registered(self) -> bool:
        """全10ティック帯域Phaseの処理関数が登録済みかを返す。"""
        return all(
            pid in self._handler_table
            for pid in self._band_phase_ids
        )

    def get_registered_phase_ids(self) -> tuple[str, ...]:
        """登録済みのPhase識別子一覧を返す。"""
        return tuple(self._handler_table.keys())

    # ── 帯域実行 ────────────────────────────────────────────────

    def execute_band(self, orchestrator: Any, user_id: str) -> PhaseExecutionLog:
        """10ティック帯域の全Phaseを宣言的定義の順序で実行する。

        帯域定義から所属Phase一覧を取得し、帯域内順序に従って
        各Phaseを順次実行する。

        各Phaseの実行は標準化された手順に従う:
        1. 宣言的定義からエラー吸収フラグを確認
        2. 処理関数を呼び出す
        3. エラー吸収が有効な場合、例外をキャッチしてログに記録

        Args:
            orchestrator: PsycheOrchestratorインスタンス
            user_id: 対話相手ID

        Returns:
            PhaseExecutionLog: 各Phase実行の成否ログ
        """
        log = PhaseExecutionLog()

        for phase_id in self._band_phase_ids:
            handler = self._handler_table.get(phase_id)
            if handler is None:
                log.phase_results[phase_id] = PhaseStatus.SKIPPED
                logger.debug(
                    "Phase %s skipped: no handler registered", phase_id
                )
                continue

            phase_def = PHASE_BY_ID[phase_id]
            if phase_def.error_absorbed:
                try:
                    handler(orchestrator, user_id)
                    log.phase_results[phase_id] = PhaseStatus.SUCCESS
                except Exception as e:
                    log.phase_results[phase_id] = PhaseStatus.FAILED
                    logger.debug(
                        "Phase %s (%s) skipped: %s",
                        phase_id, phase_def.display_name, e,
                    )
            else:
                # エラー吸収なし: 例外はそのまま伝播
                handler(orchestrator, user_id)
                log.phase_results[phase_id] = PhaseStatus.SUCCESS

        self._last_log = log
        return log

    # ── 実行ログ参照 ────────────────────────────────────────────

    @property
    def last_log(self) -> PhaseExecutionLog:
        """最新の帯域実行ログを返す。上書き型であり蓄積されない。"""
        return self._last_log

    # ── 検証用アクセサ ──────────────────────────────────────────

    def get_band_phase_definitions(self) -> tuple[PhaseDefinition, ...]:
        """10ティック帯域のPhase定義を帯域内順序で返す。

        検証・テスト用途のみ。実行ロジックには使用しない。
        """
        return tuple(
            PHASE_BY_ID[pid]
            for pid in self._band_phase_ids
        )
