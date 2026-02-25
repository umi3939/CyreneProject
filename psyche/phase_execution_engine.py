"""
psyche/phase_execution_engine.py - 宣言的Phase実行エンジン（段階3: 帯域横断対応）

宣言的定義（phase_declaration.py）に基づいてPhase実行を駆動する実行エンジン。
段階2では10ティック帯域（Phase 27/28/29）のみを対象としていたが、
段階3では帯域識別子を受け取って当該帯域のPhaseを駆動する帯域横断構造に拡張する。

対象帯域:
- 10ティック帯域（Phase 27/28/29）: 段階2から継続
- 3ティック帯域（Phase 8-14j, 17 Phase）: 段階3で追加

設計原則:
- Phase処理ロジック自体は変更しない。「呼び出し方」のみを標準化する
- 既存の手続き的コードをフォールバック経路として保持する
- 帯域別の有効フラグにより、帯域ごとに宣言的実行と手続き的実行を任意に切替可能
- save/load非影響: 内部状態は永続化しない
- enrichment非接続: 内部状態はenrichmentに接続しない
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from .phase_declaration import (
    Band,
    BandDefinition,
    BAND_EVERY_3_TICKS,
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


# ── 登録可能な帯域の集合 ──────────────────────────────────────────

# 段階3では3ティック帯域と10ティック帯域のみが対象。
# 将来の帯域追加時にはこの集合に追加するだけで既存帯域の動作に影響しない。
_SUPPORTED_BANDS: dict[Band, BandDefinition] = {
    Band.EVERY_10_TICKS: BAND_EVERY_10_TICKS,
    Band.EVERY_3_TICKS: BAND_EVERY_3_TICKS,
}


# ── Phase実行エンジン ──────────────────────────────────────────


class PhaseExecutionEngine:
    """宣言的定義に基づく帯域横断Phase実行エンジン。

    帯域識別子を受け取って当該帯域のPhaseを駆動する。
    段階3では10ティック帯域（3 Phase）と3ティック帯域（17 Phase）を対象とする。

    帯域間の構造的独立性:
    - 帯域別Phase登録テーブルは帯域ごとに分離して管理する
    - 帯域別有効フラグは帯域ごとに独立
    - 帯域別実行ログは帯域ごとに独立
    - ある帯域の操作が他帯域に影響する経路は存在しない

    Attributes:
        _band_enabled: 帯域別有効フラグ
        _band_handlers: 帯域別Phase登録テーブル
        _band_phase_order: 帯域別Phase順序表
        _band_last_log: 帯域別最新実行ログ
    """

    def __init__(self) -> None:
        """実行エンジンを初期化する。

        帯域別の登録テーブルは空の状態で作成される。
        register_handler() で処理関数を登録した後に execute_band() を呼ぶ。
        """
        # 帯域別有効フラグ: 初期値は全帯域で有効
        self._band_enabled: dict[Band, bool] = {
            band: True for band in _SUPPORTED_BANDS
        }

        # 帯域別Phase登録テーブル
        self._band_handlers: dict[Band, dict[str, PhaseHandler]] = {
            band: {} for band in _SUPPORTED_BANDS
        }

        # 帯域別Phase順序表: 宣言的定義から導出
        self._band_phase_order: dict[Band, tuple[str, ...]] = {
            band: band_def.phase_ids
            for band, band_def in _SUPPORTED_BANDS.items()
        }

        # 後方互換性: 段階2のテストが _band_phase_ids を直接参照する
        self._band_phase_ids: tuple[str, ...] = BAND_EVERY_10_TICKS.phase_ids

        # 帯域別最新実行ログ: 帯域ごとに独立、上書き型
        self._band_last_log: dict[Band, PhaseExecutionLog] = {
            band: PhaseExecutionLog() for band in _SUPPORTED_BANDS
        }

    # ── 帯域別有効フラグ制御 ──────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """10ティック帯域の有効/無効状態を返す（後方互換性）。"""
        return self._band_enabled[Band.EVERY_10_TICKS]

    def set_enabled(self, value: bool) -> None:
        """10ティック帯域の有効/無効を切り替える（後方互換性）。

        Phase処理結果に基づいて自動変化する経路は存在しない。
        外部からの明示的切替のみで変化する。
        """
        self._band_enabled[Band.EVERY_10_TICKS] = value

    def is_band_enabled(self, band: Band) -> bool:
        """指定帯域の有効/無効状態を返す。

        Args:
            band: 帯域識別子

        Returns:
            有効ならTrue
        """
        return self._band_enabled.get(band, False)

    def set_band_enabled(self, band: Band, value: bool) -> None:
        """指定帯域の有効/無効を切り替える。

        帯域ごとに独立して動作する。ある帯域の有効フラグ変更が
        他帯域の有効フラグに影響する経路は存在しない。

        Phase処理結果に基づいて自動変化する経路は存在しない。
        外部からの明示的切替のみで変化する。

        Args:
            band: 帯域識別子
            value: 有効にする場合True
        """
        if band not in _SUPPORTED_BANDS:
            raise ValueError(
                f"Band '{band}' is not supported. "
                f"Supported bands: {list(_SUPPORTED_BANDS.keys())}"
            )
        self._band_enabled[band] = value

    # ── Phase処理関数の登録 ──────────────────────────────────────

    def register_handler(self, phase_id: str, handler: PhaseHandler) -> None:
        """Phase処理関数を登録する。

        登録は初期化時に一度だけ行われることを想定する。
        Phase識別子は宣言的定義の帯域所属と照合され、対象帯域への登録が確認される。

        Args:
            phase_id: Phase識別子（宣言的定義のphase_idと一致）
            handler: 該当Phaseの処理内容を実行する関数

        Raises:
            ValueError: 指定phase_idがサポート帯域に所属しない場合
        """
        # Phase識別子から帯域を特定
        target_band = self._find_band_for_phase(phase_id)
        if target_band is None:
            # 後方互換性: 段階2のエラーメッセージ形式を維持
            raise ValueError(
                f"Phase '{phase_id}' is not in the 10-tick band. "
                f"Valid phase IDs: {tuple(self._band_phase_order[Band.EVERY_10_TICKS])}"
            )
        self._band_handlers[target_band][phase_id] = handler

    def _find_band_for_phase(self, phase_id: str) -> Optional[Band]:
        """Phase識別子が所属する帯域を返す。未対応帯域ならNone。"""
        for band, phase_ids in self._band_phase_order.items():
            if phase_id in phase_ids:
                return band
        return None

    def is_fully_registered(self) -> bool:
        """全10ティック帯域Phaseの処理関数が登録済みかを返す（後方互換性）。"""
        return self.is_band_fully_registered(Band.EVERY_10_TICKS)

    def is_band_fully_registered(self, band: Band) -> bool:
        """指定帯域の全Phaseの処理関数が登録済みかを返す。

        Args:
            band: 帯域識別子

        Returns:
            全Phase登録済みならTrue
        """
        if band not in _SUPPORTED_BANDS:
            return False
        handlers = self._band_handlers[band]
        return all(
            pid in handlers
            for pid in self._band_phase_order[band]
        )

    def get_registered_phase_ids(self) -> tuple[str, ...]:
        """10ティック帯域の登録済みPhase識別子一覧を返す（後方互換性）。"""
        return self.get_band_registered_phase_ids(Band.EVERY_10_TICKS)

    def get_band_registered_phase_ids(self, band: Band) -> tuple[str, ...]:
        """指定帯域の登録済みPhase識別子一覧を返す。

        Args:
            band: 帯域識別子

        Returns:
            登録済みPhase IDのタプル
        """
        if band not in _SUPPORTED_BANDS:
            return ()
        return tuple(self._band_handlers[band].keys())

    # ── 帯域実行 ────────────────────────────────────────────────

    def execute_band(
        self,
        orchestrator: Any,
        user_id: str,
        band: Optional[Band] = None,
    ) -> PhaseExecutionLog:
        """指定帯域の全Phaseを宣言的定義の順序で実行する。

        帯域定義から所属Phase一覧を取得し、帯域内順序に従って
        各Phaseを順次実行する。

        各Phaseの実行は標準化された手順に従う:
        1. 宣言的定義からエラー吸収フラグを確認
        2. 処理関数を呼び出す
        3. エラー吸収が有効な場合、例外をキャッチしてログに記録

        Args:
            orchestrator: PsycheOrchestratorインスタンス
            user_id: 対話相手ID
            band: 帯域識別子。省略時は10ティック帯域（後方互換性）

        Returns:
            PhaseExecutionLog: 各Phase実行の成否ログ
        """
        if band is None:
            band = Band.EVERY_10_TICKS

        if band not in _SUPPORTED_BANDS:
            raise ValueError(
                f"Band '{band}' is not supported. "
                f"Supported bands: {list(_SUPPORTED_BANDS.keys())}"
            )

        phase_ids = self._band_phase_order[band]
        handlers = self._band_handlers[band]
        log = PhaseExecutionLog()

        for phase_id in phase_ids:
            handler = handlers.get(phase_id)
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

        self._band_last_log[band] = log
        return log

    # ── 実行ログ参照 ────────────────────────────────────────────

    @property
    def last_log(self) -> PhaseExecutionLog:
        """10ティック帯域の最新実行ログを返す（後方互換性）。上書き型であり蓄積されない。"""
        return self._band_last_log[Band.EVERY_10_TICKS]

    def get_band_last_log(self, band: Band) -> PhaseExecutionLog:
        """指定帯域の最新実行ログを返す。上書き型であり蓄積されない。

        Args:
            band: 帯域識別子

        Returns:
            PhaseExecutionLog
        """
        if band not in _SUPPORTED_BANDS:
            return PhaseExecutionLog()
        return self._band_last_log[band]

    # ── 検証用アクセサ ──────────────────────────────────────────

    def get_band_phase_definitions(
        self,
        band: Optional[Band] = None,
    ) -> tuple[PhaseDefinition, ...]:
        """指定帯域のPhase定義を帯域内順序で返す。

        検証・テスト用途のみ。実行ロジックには使用しない。

        Args:
            band: 帯域識別子。省略時は10ティック帯域（後方互換性）
        """
        if band is None:
            band = Band.EVERY_10_TICKS
        if band not in _SUPPORTED_BANDS:
            return ()
        return tuple(
            PHASE_BY_ID[pid]
            for pid in self._band_phase_order[band]
        )

    def get_supported_bands(self) -> tuple[Band, ...]:
        """サポートされている帯域一覧を返す。"""
        return tuple(_SUPPORTED_BANDS.keys())
