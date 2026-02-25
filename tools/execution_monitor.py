"""
tools/execution_monitor.py - 実運用向けログ・モニタリング基盤

Phase実行時間計測、enrichment圧縮比記録、API呼び出し記録、状態スナップショット出力。
設計書: design_logging_monitoring.md

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)のみ
- 内部システムの状態変数を一切変更しない(READ-ONLY観測のみ)
- 判断・行動・選択に一切介入しない
- 計測値に基づく条件分岐を一切持たない
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない

安全弁:
1. 計測失敗時の安全な無視(例外を捕捉しログスキップ、本体処理続行)
2. スナップショットの負荷制御(設定サイクル間隔でのみ実行、間隔下限あり)
3. ログ出力量の制限(1サイクルあたりの上限超過時は帯域時間のみ出力)
4. 永続化の対象外(save/loadフィールド追加なし)
5. 環境変数による完全無効化(CYRENE_MONITOR=0で全計測点を無効化)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Optional

# 独自ログ名前空間(既存ログとの混在防止)
monitor_logger = logging.getLogger("cyrene.monitor")

# ── 環境変数制御 ──────────────────────────────────────────────────

# 安全弁5: 環境変数による完全無効化
# CYRENE_MONITOR=1 で有効化(デフォルト無効)
# CYRENE_MONITOR=0 または未設定で無効化

def is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    インポート時ではなく呼び出し時に環境変数を確認する。
    テストフィクスチャでの動的設定変更に対応するため。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── スナップショット間隔の下限(安全弁2) ─────────────────────────

_SNAPSHOT_INTERVAL_MIN = 5  # サイクル数の下限

# ── 1サイクルあたりのログ出力量上限(安全弁3) ──────────────────────

_MAX_LOG_CHARS_PER_CYCLE = 50_000


# ── ExecutionMonitor ──────────────────────────────────────────────


class ExecutionMonitor:
    """実行時観測のセッション累積と記録。

    セッション開始時にインスタンスを生成し、セッション終了時に
    emit_session_summary()を呼んで累積を出力する。
    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない
    - 計測値に基づく条件分岐を一切持たない
    - 出力はログストリームへのJSON書き込みのみ
    """

    def __init__(
        self,
        snapshot_interval: int = 50,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            snapshot_interval: 全フィールドダンプのサイクル間隔。
                安全弁2により下限が適用される。
            enabled: 明示的な有効/無効指定。Noneの場合は環境変数で判定。
        """
        # 安全弁2: スナップショット間隔の下限適用
        self._snapshot_interval = max(snapshot_interval, _SNAPSHOT_INTERVAL_MIN)

        # 有効/無効判定(明示指定 > 環境変数)
        self._enabled = enabled if enabled is not None else is_monitor_enabled()

        # ── A. セッション累積カウンタ ──
        # API呼び出し累積回数(知覚/代弁の2種別)
        self._api_call_count: dict[str, int] = {
            "perception": 0,
            "expression": 0,
        }

        # API累積トークン消費(入力/出力の2種別 x 知覚/代弁の2種別)
        self._api_token_count: dict[str, dict[str, int]] = {
            "perception": {"input": 0, "output": 0},
            "expression": {"input": 0, "output": 0},
        }

        # 帯域別累積実行時間
        self._band_cumulative_time: dict[str, float] = {}

        # 処理サイクル累積回数
        self._cycle_count: int = 0

        # ── B. 直近の計測値(1サイクル分のみ保持) ──
        # 直近の帯域別実行時間
        self._last_band_times: dict[str, float] = {}

        # 直近の圧縮前後文字数
        self._last_compression_chars: tuple[int, int] = (0, 0)

        # 直近のAPI応答メタデータ
        self._last_api_meta: dict[str, Any] = {}

        # ── C. スナップショット制御 ──
        self._last_snapshot_cycle: int = 0

        # ── サイクル内ログ文字数カウンタ(安全弁3) ──
        self._cycle_log_chars: int = 0

        # ── セッション開始時刻 ──
        self._session_start_time: float = time.time()

        # ── enrichment分布モニター(セッション境界で消失) ──
        self._enrichment_dist: EnrichmentDistributionMonitor = (
            EnrichmentDistributionMonitor()
        )

    @property
    def enabled(self) -> bool:
        """モニタリングが有効かどうか。"""
        return self._enabled

    @property
    def snapshot_interval(self) -> int:
        """スナップショット間隔(サイクル数)。"""
        return self._snapshot_interval

    @property
    def cycle_count(self) -> int:
        """処理サイクル累積回数。"""
        return self._cycle_count

    @property
    def api_call_count(self) -> dict[str, int]:
        """API呼び出し累積回数(読み取り専用コピー)。"""
        return dict(self._api_call_count)

    @property
    def api_token_count(self) -> dict[str, dict[str, int]]:
        """API累積トークン消費(読み取り専用コピー)。"""
        return {k: dict(v) for k, v in self._api_token_count.items()}

    @property
    def band_cumulative_time(self) -> dict[str, float]:
        """帯域別累積実行時間(読み取り専用コピー)。"""
        return dict(self._band_cumulative_time)

    @property
    def last_band_times(self) -> dict[str, float]:
        """直近の帯域別実行時間(読み取り専用コピー)。"""
        return dict(self._last_band_times)

    @property
    def last_compression_chars(self) -> tuple[int, int]:
        """直近の圧縮前後文字数(圧縮前, 圧縮後)。"""
        return self._last_compression_chars

    @property
    def enrichment_distribution(self) -> "EnrichmentDistributionMonitor":
        """enrichment分布モニター(読み取り専用アクセサ)。"""
        return self._enrichment_dist

    # ── enrichment分布の記録 ────────────────────────────────────────

    def record_enrichment_distribution(
        self,
        tick_count: int,
        sections_data: list[dict],
        compressed_text: str,
    ) -> None:
        """enrichment項目の出力分布を記述・記録する。

        enrichment項目収集完了後かつ圧縮パイプライン適用後に呼び出す。

        Args:
            tick_count: 現在のティックカウント
            sections_data: セクション定義のリスト
            compressed_text: 圧縮パイプライン適用後のテキスト全体
        """
        if not self._enabled:
            return
        try:
            self._enrichment_dist.record_enrichment_distribution(
                tick_count=tick_count,
                sections_data=sections_data,
                compressed_text=compressed_text,
                monitor=self,
            )
        except Exception:
            # 安全弁1
            pass

    # ── 帯域実行時間の記録 ────────────────────────────────────────

    def record_band_time(self, band_name: str, elapsed: float) -> None:
        """帯域の実行時間を記録する。

        Args:
            band_name: 帯域識別子(例: "every_tick", "every_3_ticks" 等)
            elapsed: 実行時間(秒)
        """
        if not self._enabled:
            return
        try:
            # 直近値の更新(1サイクル分のみ)
            self._last_band_times[band_name] = elapsed

            # 累積値の更新
            self._band_cumulative_time[band_name] = (
                self._band_cumulative_time.get(band_name, 0.0) + elapsed
            )
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    # ── 圧縮比の記録 ──────────────────────────────────────────────

    def record_compression(
        self, before_chars: int, after_chars: int, ratio: float
    ) -> None:
        """enrichment圧縮の前後文字数と圧縮率を記録する。

        Args:
            before_chars: 圧縮前の文字数
            after_chars: 圧縮後の文字数
            ratio: 圧縮率(build_compressed_enrichmentの戻り値)
        """
        if not self._enabled:
            return
        try:
            self._last_compression_chars = (before_chars, after_chars)

            record = {
                "type": "enrichment_compression",
                "timestamp": time.time(),
                "before_chars": before_chars,
                "after_chars": after_chars,
                "ratio": round(ratio, 4),
            }
            self._emit_json(record)
        except Exception:
            # 安全弁1
            pass

    # ── API呼び出しの記録 ─────────────────────────────────────────

    def record_api_call(
        self,
        call_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """外部API呼び出しの記録。

        Args:
            call_type: 呼び出し種別("perception" or "expression")
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
        """
        if not self._enabled:
            return
        try:
            # 累積カウンタ更新
            if call_type in self._api_call_count:
                self._api_call_count[call_type] += 1
            else:
                self._api_call_count[call_type] = 1

            # トークン累積更新
            if call_type not in self._api_token_count:
                self._api_token_count[call_type] = {"input": 0, "output": 0}
            self._api_token_count[call_type]["input"] += input_tokens
            self._api_token_count[call_type]["output"] += output_tokens

            # 直近メタデータ更新
            self._last_api_meta = {
                "call_type": call_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": time.time(),
            }

            record = {
                "type": "api_call",
                "timestamp": time.time(),
                "call_type": call_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cumulative_calls": dict(self._api_call_count),
            }
            self._emit_json(record)
        except Exception:
            # 安全弁1
            pass

    # ── サイクル完了の記録 ─────────────────────────────────────────

    def record_cycle_complete(self, tick_count: int) -> None:
        """1処理サイクルの完了を記録する。

        帯域別実行時間をまとめてログ出力する。

        Args:
            tick_count: 現在のティックカウント
        """
        if not self._enabled:
            return
        try:
            self._cycle_count += 1
            self._cycle_log_chars = 0  # サイクル内ログカウンタリセット

            record = {
                "type": "cycle_complete",
                "timestamp": time.time(),
                "tick_count": tick_count,
                "cycle_count": self._cycle_count,
                "band_times": dict(self._last_band_times),
            }
            self._emit_json(record)

            # 直近帯域時間をリセット(次サイクル用)
            self._last_band_times = {}
        except Exception:
            # 安全弁1
            pass

    # ── 全フィールドスナップショット ──────────────────────────────

    def maybe_emit_snapshot(
        self,
        tick_count: int,
        state_reader: Any,
    ) -> bool:
        """スナップショット間隔に達していれば全フィールドダンプを出力する。

        全66フィールドの現在値をログストリームに出力する。
        フィールドの取捨選択は行わず、全フィールドを等価に出力する。

        Args:
            tick_count: 現在のティックカウント
            state_reader: 全フィールドを読み取るための呼び出し可能オブジェクト。
                          引数なしで呼び出され、dict[str, Any]を返す。

        Returns:
            スナップショットを出力したかどうか。
        """
        if not self._enabled:
            return False
        try:
            # 安全弁2: スナップショット間隔チェック
            cycles_since = self._cycle_count - self._last_snapshot_cycle
            if cycles_since < self._snapshot_interval:
                return False

            # 全フィールド読み取り(READ-ONLY)
            fields = state_reader()
            if not isinstance(fields, dict):
                return False

            self._last_snapshot_cycle = self._cycle_count

            record = {
                "type": "state_snapshot",
                "timestamp": time.time(),
                "tick_count": tick_count,
                "cycle_count": self._cycle_count,
                "field_count": len(fields),
                "fields": fields,
            }
            self._emit_json(record)
            return True
        except Exception:
            # 安全弁1
            return False

    # ── セッションサマリの出力 ─────────────────────────────────────

    def emit_session_summary(self) -> None:
        """セッション終了時の累積情報を出力する。"""
        if not self._enabled:
            return
        try:
            elapsed = time.time() - self._session_start_time
            record = {
                "type": "session_summary",
                "timestamp": time.time(),
                "session_duration_seconds": round(elapsed, 2),
                "total_cycles": self._cycle_count,
                "api_call_counts": dict(self._api_call_count),
                "api_token_totals": {
                    k: dict(v) for k, v in self._api_token_count.items()
                },
                "band_cumulative_times": dict(self._band_cumulative_time),
            }
            self._emit_json(record)
        except Exception:
            # 安全弁1
            pass

    # ── 内部: JSON構造化ログ出力 ──────────────────────────────────

    def _emit_json(self, record: dict[str, Any]) -> None:
        """JSON構造化ログをログストリームに出力する。

        安全弁3: 1サイクルあたりのログ出力量に上限を設ける。
        上限超過時は帯域別実行時間のみを出力し、他の項目を省略する。
        """
        try:
            text = json.dumps(record, ensure_ascii=False, default=str)

            # 安全弁3: ログ出力量チェック
            self._cycle_log_chars += len(text)
            if self._cycle_log_chars > _MAX_LOG_CHARS_PER_CYCLE:
                # 上限超過: band_timesのみ出力許可
                if record.get("type") not in ("cycle_complete",):
                    return

            monitor_logger.debug(text)
        except Exception:
            # 安全弁1
            pass


# ── ヘルパー: 帯域計測コンテキストマネージャ ──────────────────────


class BandTimer:
    """帯域実行時間を計測するコンテキストマネージャ。

    with BandTimer(monitor, "every_tick"):
        ... 帯域処理 ...

    計測点自体が例外を投げた場合、元の処理に影響を与えない。
    """

    def __init__(self, monitor: Optional[ExecutionMonitor], band_name: str) -> None:
        self._monitor = monitor
        self._band_name = band_name
        self._start: float = 0.0

    def __enter__(self) -> "BandTimer":
        try:
            if self._monitor and self._monitor.enabled:
                self._start = time.perf_counter()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self._monitor and self._monitor.enabled and self._start > 0:
                elapsed = time.perf_counter() - self._start
                self._monitor.record_band_time(self._band_name, elapsed)
        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass
        # 元の例外を再送出(Falseを返す = 例外を抑制しない)
        return None


# ── ヘルパー: 状態スナップショット読み取り ─────────────────────────


def read_orchestrator_fields(orchestrator: Any) -> dict[str, Any]:
    """orchestratorの全永続化フィールドをREAD-ONLYで読み取る。

    saveメソッドと同等の読み取りパスを使用するが、
    saveメソッド自体は呼ばない。永続化フォーマットではなく
    ログ出力用のフォーマットで構造化する。

    全66フィールドを等価に出力する。フィールドの取捨選択は行わない。

    Args:
        orchestrator: PsycheOrchestratorインスタンス

    Returns:
        全フィールドの現在値を格納した辞書。
    """
    fields: dict[str, Any] = {}
    try:
        # tick_count
        fields["tick_count"] = orchestrator._tick_count

        # Core psyche state
        try:
            p = orchestrator._psyche
            fields["psyche_emotion"] = p.emotion_summary()
            fields["psyche_mood_valence"] = p.mood.valence
            fields["psyche_mood_arousal"] = p.mood.arousal
            fields["psyche_drives_social"] = p.drives.social
            fields["psyche_drives_curiosity"] = p.drives.curiosity
            fields["psyche_drives_expression"] = p.drives.expression
            fields["psyche_fear_level"] = p.fear_level
            fields["psyche_dominant_emotion"] = p.dominant_emotion
            fields["psyche_dominant_emotion_value"] = p.dominant_emotion_value
        except Exception:
            fields["psyche"] = "read_error"

        # Dynamics
        try:
            d = orchestrator._dynamics
            if d:
                fields["dynamics_phase"] = d.current_phase
                fields["dynamics_accumulated"] = (
                    orchestrator._amplitude_state.current_amplitude
                    if orchestrator._amplitude_state else 0.0
                )
        except Exception:
            fields["dynamics"] = "read_error"

        # Loop state (STM)
        try:
            ls = orchestrator._loop_state
            if ls and ls.memory:
                fields["stm_entry_count"] = len(ls.memory.entries)
            else:
                fields["stm_entry_count"] = 0
        except Exception:
            fields["stm"] = "read_error"

        # Value orientation
        try:
            vo = orchestrator._value_orientation
            if vo:
                fields["value_orientation_axes"] = {
                    axis.name: round(axis.value, 4)
                    for axis in vo.axes
                } if hasattr(vo, 'axes') else str(vo)
        except Exception:
            fields["value_orientation"] = "read_error"

        # Stability valve
        try:
            sv = orchestrator._stability_valve
            if sv:
                fields["stability_valve_count"] = (
                    sv.observation_count
                    if hasattr(sv, 'observation_count') else 0
                )
        except Exception:
            fields["stability_valve"] = "read_error"

        # Tendency system
        try:
            ts = orchestrator._tendency_sys
            if ts and ts.state:
                fields["tendency_count"] = len(ts.state.tendencies)
        except Exception:
            fields["tendency"] = "read_error"

        # Self-model
        try:
            fields["self_model_available"] = (
                orchestrator._last_self_view is not None
            )
        except Exception:
            pass

        # Temporal self-difference
        try:
            fields["temporal_diff_available"] = (
                orchestrator._last_diff_summary is not None
            )
        except Exception:
            pass

        # Continuity strain
        try:
            fields["strain_available"] = (
                orchestrator._last_strain is not None
            )
        except Exception:
            pass

        # Self-image
        try:
            fields["self_image_available"] = (
                orchestrator._last_self_image is not None
            )
        except Exception:
            pass

        # Identity coherence
        try:
            fields["coherence_available"] = (
                orchestrator._last_coherence is not None
            )
        except Exception:
            pass

        # Self-narrative
        try:
            fields["narrative_available"] = (
                orchestrator._last_narrative is not None
            )
        except Exception:
            pass

        # Episodic memory
        try:
            ep = orchestrator._last_episodes
            if ep:
                fields["episodic_episode_count"] = (
                    len(ep.episodes) if ep.has_episodes else 0
                )
            else:
                fields["episodic_episode_count"] = 0
        except Exception:
            fields["episodic_memory"] = "read_error"

        # Emotional memory binding
        try:
            fields["binding_available"] = (
                orchestrator._last_bindings is not None
            )
        except Exception:
            pass

        # Introspection
        try:
            fields["introspection_available"] = (
                orchestrator._last_trace is not None
            )
        except Exception:
            pass

        # Consumption
        try:
            fields["consumption_available"] = (
                orchestrator._last_consumption is not None
            )
        except Exception:
            pass

        # Expectations
        try:
            fields["expectations_available"] = (
                orchestrator._last_expectations is not None
            )
        except Exception:
            pass

        # Intrinsic motivation
        try:
            fields["motivation_available"] = (
                orchestrator._last_motives is not None
            )
        except Exception:
            pass

        # Other agent model
        try:
            fields["other_model_available"] = (
                orchestrator._last_other_model is not None
            )
        except Exception:
            pass

        # Proto-goal vector
        try:
            vg = orchestrator._vector_gen
            if vg and vg.state:
                fields["proto_vector_count"] = len(vg.state.vectors)
        except Exception:
            fields["proto_vector"] = "read_error"

        # Goal candidates
        try:
            cg = orchestrator._candidate_gen
            if cg and cg.state:
                fields["goal_candidate_count"] = len(cg.state.candidates)
        except Exception:
            fields["goal_candidates"] = "read_error"

        # Transient goal
        try:
            tg = orchestrator._transient_goal_mgr
            if tg and tg.state:
                fields["transient_goal_active"] = (
                    tg.state.active_goal is not None
                )
        except Exception:
            fields["transient_goal"] = "read_error"

        # Scoped goal
        try:
            sg = orchestrator._scoped_goal_sys
            fields["scoped_goal_active"] = sg.has_active_scope
        except Exception:
            fields["scoped_goal"] = "read_error"

        # Dispersion state
        try:
            fields["dispersion_active_count"] = len(
                get_dispersion_active_units_safe(orchestrator._dispersion_state)
            )
        except Exception:
            fields["dispersion"] = "read_error"

        # Policy expander
        try:
            pe = orchestrator._policy_expander
            if pe and hasattr(pe, 'state'):
                fields["policy_expander_available"] = True
        except Exception:
            fields["policy_expander"] = "read_error"

        # Memory integrator
        try:
            fields["memory_integrator_available"] = (
                orchestrator._memory_integrator is not None
            )
        except Exception:
            pass

        # Real feed
        try:
            fields["real_feed_available"] = (
                orchestrator._last_feed_result is not None
            )
        except Exception:
            pass

        # Text dialogue
        try:
            fields["text_dialogue_available"] = (
                orchestrator._text_dialogue_processor is not None
            )
        except Exception:
            pass

        # Spontaneous activation
        try:
            fields["spontaneous_available"] = (
                orchestrator._last_activation_result is not None
            )
        except Exception:
            pass

        # VO validation
        try:
            fields["vo_validation_available"] = (
                orchestrator._last_vo_validation is not None
            )
        except Exception:
            pass

        # Forgetting fixation
        try:
            fields["forgetting_fixation_available"] = (
                orchestrator._last_forgetting_fixation is not None
            )
        except Exception:
            pass

        # Action result
        try:
            fields["action_result_available"] = (
                orchestrator._last_action_result is not None
            )
        except Exception:
            pass

        # Dialogue learning
        try:
            fields["dialogue_learning_available"] = (
                orchestrator._last_dialogue_learning is not None
            )
        except Exception:
            pass

        # Meta-emotion
        try:
            fields["meta_emotion_available"] = (
                orchestrator._last_meta_emotion is not None
            )
        except Exception:
            pass

        # STM-emotion coupling
        try:
            fields["stm_coupling_available"] = (
                orchestrator._last_coupling is not None
            )
        except Exception:
            pass

        # Decision bias (cached from Phase 30-35)
        try:
            fields["decision_bias_available"] = (
                orchestrator._last_decision_bias is not None
            )
        except Exception:
            pass

        # Tone modifier (cached from Phase 30-35)
        try:
            fields["tone_modifier_available"] = (
                orchestrator._last_tone_mod is not None
            )
        except Exception:
            pass

        # Sensitivity bias (cached from Phase 30-35)
        try:
            fields["sensitivity_bias_available"] = (
                orchestrator._last_sensitivity_bias is not None
            )
        except Exception:
            pass

        # Selected policy label
        try:
            fields["last_selected_policy_label"] = (
                orchestrator._last_selected_policy_label
            )
        except Exception:
            pass

        # Enrichment cache size
        try:
            fields["enrichment_cache_size"] = len(
                orchestrator._enrichment_prev_cache
            )
        except Exception:
            pass

        # Session gap
        try:
            fields["session_gap_seconds"] = orchestrator._session_gap_seconds
        except Exception:
            pass

        # Multi-path recall
        try:
            fields["multi_path_recall_available"] = (
                orchestrator._multi_path_recall is not None
            )
        except Exception:
            pass

        # Spontaneous recall
        try:
            fields["spontaneous_recall_available"] = (
                orchestrator._spontaneous_recall is not None
            )
        except Exception:
            pass

        # Introspection cross-section
        try:
            fields["introspection_cross_section_available"] = (
                orchestrator._introspection_cross_section is not None
            )
        except Exception:
            pass

        # Temporal cognition
        try:
            fields["temporal_cognition_available"] = (
                orchestrator._temporal_cognition is not None
            )
        except Exception:
            pass

        # Perceptual context
        try:
            fields["perceptual_context_available"] = (
                orchestrator._perceptual_context is not None
            )
        except Exception:
            pass

        # Selection attribution
        try:
            fields["selection_attribution_available"] = (
                orchestrator._selection_attribution_recorder is not None
            )
        except Exception:
            pass

        # Reference frequency
        try:
            fields["reference_frequency_available"] = (
                orchestrator._reference_frequency_state is not None
            )
        except Exception:
            pass

        # Persistent commitment
        try:
            pc = orchestrator._persistent_commitment
            if pc and hasattr(pc, 'state') and pc.state:
                fields["persistent_commitment_count"] = len([
                    it for it in pc.state.items if not it.released
                ])
        except Exception:
            fields["persistent_commitment"] = "read_error"

        # Contradiction
        try:
            fields["contradiction_available"] = (
                orchestrator._last_contradiction_result is not None
            )
        except Exception:
            pass

        # Emotional backdrop
        try:
            fields["emotional_backdrop_available"] = (
                orchestrator._last_backdrop_result is not None
            )
        except Exception:
            pass

        # Drive variation
        try:
            fields["drive_variation_available"] = (
                orchestrator._last_drive_variation_result is not None
            )
        except Exception:
            pass

        # Cooccurrence
        try:
            fields["cooccurrence_available"] = (
                orchestrator._last_cooccurrence_result is not None
            )
        except Exception:
            pass

        # Other boundary accumulation
        try:
            fields["boundary_accumulation_available"] = (
                orchestrator._last_boundary_accumulation is not None
            )
        except Exception:
            pass

    except Exception:
        fields["_read_error"] = "top_level_error"

    return fields


# ── EnrichmentDistributionMonitor ──────────────────────────────────
#
# enrichment項目の出力分布記述と構造的監視。
# 設計書: design_enrichment_monitoring.md
#
# 本機能の構造的分離:
# - enrichment項目の出力を読み取り専用で観測する(一方向のデータフロー)
# - enrichmentテキスト、圧縮パイプライン、psycheモジュール、
#   判断・行動の選択に出力を供給する経路が存在しない
# - 出力先は既存の観測記録基盤(ExecutionMonitor)と読み取り専用アクセサのみ
# - psycheモジュールの内部状態を一切変更しない
# - save/loadの対象フィールドに一切追加しない
# - orchestratorのPhase処理に組み込まれない
#
# 安全弁:
# 1. 計測失敗時の安全な無視(例外を捕捉しスキップ)
# 2. 時間窓の上限によるメモリ制御(FIFOリスト上限)
# 3. 重複検出の計算量制御(比較回数上限)
# 4. 記録出力量の制限(既存の出力量制御機構に従う)
# 5. 環境変数による完全無効化(既存の観測機構と同一)
# 6. 永続化の非対象(全内部状態はセッション境界で消失)
# 7. psycheモジュール非変更


# 既知の空パターン(enrichment_compression.pyの空状態検出と整合)
_KNOWN_EMPTY_PATTERNS: frozenset[str] = frozenset({
    "",
    "(なし)",
    "(空)",
    "(蓄積前)",
    "(未蓄積)",
    "(安定)",
})

# 分布履歴FIFOのデフォルト上限
_DISTRIBUTION_HISTORY_DEFAULT_MAX = 100

# 重複検出の比較回数上限(安全弁3)
_DUPLICATE_COMPARISON_LIMIT = 500

# 重複検出間隔(ティック数)
_DUPLICATE_CHECK_INTERVAL = 10

# 部分一致判定の閾値(文字列長に対する共通部分の比率)
_DUPLICATE_SIMILARITY_THRESHOLD = 0.8

# スナップショット時の項目別詳細出力間隔(ティック数)
_ITEM_DETAIL_SNAPSHOT_INTERVAL = 10


class EnrichmentDistributionMonitor:
    """enrichment項目の出力分布記述と構造的監視。

    ExecutionMonitorと統合して動作し、enrichment項目の出力特性を
    観測・記述する。psyche内部には一切の変更を加えない外部観測ツール。

    全内部状態はインスタンス破棄時に消失する(永続化対象外)。
    """

    def __init__(
        self,
        history_max: int = _DISTRIBUTION_HISTORY_DEFAULT_MAX,
        duplicate_check_interval: int = _DUPLICATE_CHECK_INTERVAL,
        item_detail_interval: int = _ITEM_DETAIL_SNAPSHOT_INTERVAL,
    ) -> None:
        """初期化。

        Args:
            history_max: 分布履歴FIFOの上限サイズ(安全弁2)
            duplicate_check_interval: 重複検出の実行間隔(ティック数)
            item_detail_interval: 項目別詳細の出力間隔(ティック数)
        """
        # (a) 前回テキストキャッシュ: ラベル→直前ティックの出力テキスト
        self._prev_texts: dict[str, str] = {}

        # (b) 時間窓内の分布履歴: FIFO
        self._history: deque[dict[str, Any]] = deque(maxlen=max(1, history_max))

        # (c) 項目別累積カウンタ: ラベル→{non_empty, changed, observed}
        self._item_counters: dict[str, dict[str, int]] = {}

        # (d) 重複検出結果キャッシュ
        self._duplicate_pairs: list[tuple[str, str, float]] = []
        self._last_duplicate_check_tick: int = 0
        self._duplicate_check_interval = max(1, duplicate_check_interval)

        # 項目別詳細出力間隔
        self._item_detail_interval = max(1, item_detail_interval)
        self._last_item_detail_tick: int = 0

        # 観測ティック数
        self._observation_count: int = 0

    @property
    def observation_count(self) -> int:
        """観測したティック数。"""
        return self._observation_count

    @property
    def history(self) -> list[dict[str, Any]]:
        """時間窓内の分布履歴(読み取り専用コピー)。"""
        return list(self._history)

    @property
    def item_counters(self) -> dict[str, dict[str, int]]:
        """項目別累積カウンタ(読み取り専用コピー)。"""
        return {k: dict(v) for k, v in self._item_counters.items()}

    @property
    def duplicate_pairs(self) -> list[tuple[str, str, float]]:
        """直近の重複検出結果(読み取り専用コピー)。"""
        return list(self._duplicate_pairs)

    def record_enrichment_distribution(
        self,
        tick_count: int,
        sections_data: list[dict],
        compressed_text: str,
        monitor: Optional["ExecutionMonitor"] = None,
    ) -> None:
        """enrichment項目の出力分布を記述・記録する。

        第1段(項目別出力特性記述)→第2段(集計)→第3段(履歴蓄積+重複検出)を
        順に実行する。

        Args:
            tick_count: 現在のティックカウント
            sections_data: セクション定義のリスト。各要素は:
                {"header": str, "items": list[tuple[str, str]]}
            compressed_text: 圧縮パイプライン適用後のテキスト全体
            monitor: 親のExecutionMonitorインスタンス(ログ出力用)
        """
        try:
            self._observation_count += 1

            # ── 第1段: 項目別の出力特性記述 ──
            item_details: list[dict[str, Any]] = []
            section_summaries: list[dict[str, Any]] = []

            for section in sections_data:
                header = section.get("header", "")
                items = section.get("items", [])
                section_non_empty = 0
                section_changed = 0
                section_total = len(items)

                for label, text in items:
                    # 空判定
                    is_empty = self._is_empty(text)
                    non_empty = not is_empty

                    # 変動判定(前回テキストとの文字列同一性比較)
                    changed = False
                    if label in self._prev_texts:
                        changed = (self._prev_texts[label] != text)
                    else:
                        # 初回観測は変動扱い
                        changed = True

                    # テキスト長
                    text_len = len(text)

                    # 項目別詳細
                    item_details.append({
                        "label": label,
                        "section": header,
                        "non_empty": non_empty,
                        "changed": changed,
                        "text_length": text_len,
                    })

                    # 累積カウンタ更新
                    if label not in self._item_counters:
                        self._item_counters[label] = {
                            "non_empty": 0,
                            "changed": 0,
                            "observed": 0,
                        }
                    self._item_counters[label]["observed"] += 1
                    if non_empty:
                        self._item_counters[label]["non_empty"] += 1
                        section_non_empty += 1
                    if changed:
                        self._item_counters[label]["changed"] += 1
                        section_changed += 1

                    # 前回テキストキャッシュ更新(1ティック分のみ)
                    self._prev_texts[label] = text

                section_summaries.append({
                    "header": header,
                    "non_empty": section_non_empty,
                    "changed": section_changed,
                    "total": section_total,
                })

            # ── 第2段: 全体集計 ──
            total_items = sum(s["total"] for s in section_summaries)
            total_non_empty = sum(s["non_empty"] for s in section_summaries)
            total_changed = sum(s["changed"] for s in section_summaries)
            compressed_chars = len(compressed_text)

            # ── 第3段: 履歴蓄積 ──
            history_entry: dict[str, Any] = {
                "tick": tick_count,
                "sections": section_summaries,
                "total_items": total_items,
                "total_non_empty": total_non_empty,
                "total_changed": total_changed,
                "compressed_chars": compressed_chars,
            }
            self._history.append(history_entry)

            # 重複検出(一定間隔で実行)
            if (tick_count - self._last_duplicate_check_tick
                    >= self._duplicate_check_interval):
                self._detect_duplicates(sections_data, tick_count)

            # ── ログ出力 ──
            if monitor and monitor.enabled:
                # 通常ティック: セクション別集計のみ
                summary_record = {
                    "type": "enrichment_distribution",
                    "timestamp": time.time(),
                    "tick_count": tick_count,
                    "total_items": total_items,
                    "total_non_empty": total_non_empty,
                    "total_changed": total_changed,
                    "compressed_chars": compressed_chars,
                    "sections": section_summaries,
                }
                monitor._emit_json(summary_record)

                # 項目別詳細はスナップショット間隔でのみ出力(記録出力量の制御)
                if (tick_count - self._last_item_detail_tick
                        >= self._item_detail_interval):
                    self._last_item_detail_tick = tick_count
                    detail_record = {
                        "type": "enrichment_item_detail",
                        "timestamp": time.time(),
                        "tick_count": tick_count,
                        "items": item_details,
                    }
                    monitor._emit_json(detail_record)

                # 重複検出結果がある場合に出力
                if self._duplicate_pairs:
                    dup_record = {
                        "type": "enrichment_duplicate_pairs",
                        "timestamp": time.time(),
                        "tick_count": tick_count,
                        "pairs": [
                            {"label_a": a, "label_b": b, "similarity": round(s, 4)}
                            for a, b, s in self._duplicate_pairs
                        ],
                    }
                    monitor._emit_json(dup_record)

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    def get_distribution_summary(self) -> dict[str, Any]:
        """現在の分布サマリを読み取り専用で返す。

        外部の分析ツール(シミュレータ等)が呼び出す読み取り専用アクセサ。

        Returns:
            分布サマリの辞書。キー:
            - observation_count: 観測ティック数
            - item_counters: 項目別累積カウンタ
            - history_length: 履歴バッファ内のエントリ数
            - latest_entry: 最新の履歴エントリ(存在する場合)
            - duplicate_pairs: 直近の重複検出結果
        """
        result: dict[str, Any] = {
            "observation_count": self._observation_count,
            "item_counters": self.item_counters,
            "history_length": len(self._history),
            "latest_entry": dict(self._history[-1]) if self._history else None,
            "duplicate_pairs": self.duplicate_pairs,
        }
        return result

    def _is_empty(self, text: str) -> bool:
        """テキストが空状態を示すかを判定する。

        空文字列、既知の空パターン(「(未蓄積)」「(安定)」等)を空として扱う。
        """
        stripped = text.strip()
        if not stripped:
            return True
        # ラベル付き形式 "ラベル: (未蓄積)" からマーカー部分を抽出
        if ": " in stripped:
            after_colon = stripped.split(": ", 1)[1].strip()
            if after_colon in _KNOWN_EMPTY_PATTERNS:
                return True
        if stripped in _KNOWN_EMPTY_PATTERNS:
            return True
        return False

    def _detect_duplicates(
        self,
        sections_data: list[dict],
        tick_count: int,
    ) -> None:
        """項目間の出力テキスト重複を検出する。

        同一ティック内で異なる項目が返した出力テキスト間の部分一致度を検査する。
        高い部分一致が検出された項目対をリストとして記録する。
        重複の「理由」や「解消方法」は推測しない。

        安全弁3: 比較回数に上限を設け、計算量爆発を防止する。
        """
        try:
            self._last_duplicate_check_tick = tick_count

            # 全項目の(ラベル, テキスト)を収集(空項目を除く)
            items: list[tuple[str, str]] = []
            for section in sections_data:
                for label, text in section.get("items", []):
                    if not self._is_empty(text):
                        items.append((label, text))

            pairs: list[tuple[str, str, float]] = []
            comparison_count = 0

            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    # 安全弁3: 比較回数上限
                    comparison_count += 1
                    if comparison_count > _DUPLICATE_COMPARISON_LIMIT:
                        # 上限到達、打ち切り
                        self._duplicate_pairs = pairs
                        return

                    label_a, text_a = items[i]
                    label_b, text_b = items[j]

                    similarity = self._compute_similarity(text_a, text_b)
                    if similarity >= _DUPLICATE_SIMILARITY_THRESHOLD:
                        pairs.append((label_a, label_b, similarity))

            # 完全に上書き(次回検出実行時にキャッシュは完全に上書き)
            self._duplicate_pairs = pairs

        except Exception:
            # 安全弁1
            pass

    @staticmethod
    def _compute_similarity(text_a: str, text_b: str) -> float:
        """2つのテキスト間の類似度を算出する。

        文字レベルの共通部分比率で部分一致度を計測する。
        短い方のテキスト長に対する一致文字数の比率を返す。

        Args:
            text_a: テキストA
            text_b: テキストB

        Returns:
            0.0〜1.0の類似度
        """
        if not text_a or not text_b:
            return 0.0
        if text_a == text_b:
            return 1.0

        # 短い方のテキスト長を基準に類似度を算出
        shorter = min(len(text_a), len(text_b))
        if shorter == 0:
            return 0.0

        # 共通部分文字数(順序を考慮した最長共通部分列の近似)
        # 計算量を抑えるため、文字のバイグラム集合の重複度で近似する
        set_a = set(text_a[i:i+2] for i in range(len(text_a) - 1)) if len(text_a) > 1 else {text_a}
        set_b = set(text_b[i:i+2] for i in range(len(text_b) - 1)) if len(text_b) > 1 else {text_b}

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        if union == 0:
            return 0.0

        return intersection / union


def get_dispersion_active_units_safe(dispersion_state: Any) -> list:
    """dispersion_stateからactive_unitsを安全に取得する。"""
    try:
        if dispersion_state is None:
            return []
        if hasattr(dispersion_state, 'active_units'):
            return list(dispersion_state.active_units)
        return []
    except Exception:
        return []
