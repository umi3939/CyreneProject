"""
tools/anomaly_detection.py - 実運用時の動態停止検出と警告フレームワーク

設計書: design_anomaly_detection.md, design_gradual_fixation_detection.md

既存の記録ツール群（帰還経路モニター、実行時観測基盤、enrichment分布モニター、
パイプライン計測ツール）が蓄積したデータと、オーケストレータが保持する状態を
READ-ONLYで参照し、「動態の停止」という構造的事実を検出して開発者に通知する。

緩やかな固定化進行の検出:
既存の「変化率ゼロ」検出に加え、ウィンドウ内における状態値の散らばり（分散）が
時間的に一方向に減少し続けているという事実を記録する。スナップショットバッファを
設定可能な個数のサブウィンドウに分割し、各サブウィンドウ内の分散が単調減少している
場合に「緩やかな収束が進行中」という事実を記録する。対象は感情・駆動信号のみ。
検出結果はログストリームへの記録のみであり、内部システムへの供給経路は存在しない。

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)のみ
- 内部システムの状態変数を一切変更しない(READ-ONLY観測のみ)
- 判断・行動・選択に一切介入しない
- 計測値に基づく条件分岐を一切持たない(停止検出の結果を内部処理に供給しない)
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない
- enrichmentに反映されない
- psycheの状態更新経路に接続されない
- 検出結果を内部処理のオーケストレータのPhase処理、enrichment項目、
  記憶系統、判断・行動の選択、帰還経路に供給しない
- 動態停止の「原因」を推定・帰属しない
- 動態停止の「解消方法」を提示しない
- 将来の動態停止を「予測」しない
- 「正常な状態範囲」「あるべき動作パターン」「望ましい活性度」を定義しない

安全弁:
1. 計測失敗時の安全な無視(例外を捕捉しログスキップ、本体処理続行)
2. 環境変数による完全無効化(既存のモニタリング基盤と同一: CYRENE_MONITOR)
3. 修復機能の構造的排除(状態変更メソッドが存在しない)
4. enrichment非接続: 検出結果はenrichmentの構成・内容・項目数に一切影響しない
5. psyche状態非接続: 検出結果はpsycheの状態変数に一切書き込まない
6. 永続化非対象: 全内部状態はセッション境界で消失する
7. FIFOバッファの上限制御: メモリ消費が無制限に増加しない
8. ログ出力の冗長抑制: 停止/収束状態への遷移時と解消時にのみ出力する
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Optional

# 独自ログ名前空間(既存ログとの混在防止)
_logger = logging.getLogger("cyrene.monitor.anomaly_detection")


# ── 信号種別識別子 ──────────────────────────────────────────────
SIGNAL_EMOTION = "emotion"
SIGNAL_DRIVE = "drive"
SIGNAL_RETURN_PATHWAY = "return_pathway"
SIGNAL_ENRICHMENT_VARIATION = "enrichment_variation"

_ALL_SIGNALS = frozenset({
    SIGNAL_EMOTION,
    SIGNAL_DRIVE,
    SIGNAL_RETURN_PATHWAY,
    SIGNAL_ENRICHMENT_VARIATION,
})

# 緩やかな固定化検出の対象信号(分散の単調減少検出)
# 帰還経路発火(真偽値)とenrichment変動(整数)は分散算出の対象外
_GRADUAL_CONVERGENCE_SIGNALS = frozenset({
    SIGNAL_EMOTION,
    SIGNAL_DRIVE,
})


# ── FIFOバッファのデフォルト上限(安全弁7) ─────────────────────
_DEFAULT_BUFFER_MAX = 30

# ── 緩やかな固定化検出のデフォルトサブウィンドウ分割数 ──────────
_DEFAULT_SUB_WINDOW_COUNT = 5


# ── 環境変数制御 ──────────────────────────────────────────────────

def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存の実行時観測基盤と同じ環境変数(CYRENE_MONITOR)に従う。
    インポート時ではなく呼び出し時に環境変数を確認する。
    テストフィクスチャでの動的設定変更に対応するため。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── スナップショットデータ構造 ───────────────────────────────────


class _Snapshot:
    """1回の読み取り時点における状態断面。

    内部使用のみ。外部に公開しない。
    """

    __slots__ = (
        "tick_number",
        "timestamp",
        "emotion_values",
        "drive_values",
        "return_pathway_fired",
        "enrichment_variation_count",
    )

    def __init__(
        self,
        tick_number: int,
        emotion_values: dict[str, float],
        drive_values: dict[str, float],
        return_pathway_fired: bool,
        enrichment_variation_count: int,
    ) -> None:
        self.tick_number = tick_number
        self.timestamp = time.time()
        self.emotion_values = dict(emotion_values)
        self.drive_values = dict(drive_values)
        self.return_pathway_fired = return_pathway_fired
        self.enrichment_variation_count = enrichment_variation_count

    def to_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書表現。"""
        return {
            "tick_number": self.tick_number,
            "timestamp": self.timestamp,
            "emotion_values": self.emotion_values,
            "drive_values": self.drive_values,
            "return_pathway_fired": self.return_pathway_fired,
            "enrichment_variation_count": self.enrichment_variation_count,
        }


# ── AnomalyDetector ──────────────────────────────────────────────


class AnomalyDetector:
    """動態停止の検出と警告出力。

    既存の記録ツール群およびオーケストレータの状態読み取り関数のみを参照する。
    参照先への書き込みメソッドは本クラスから呼び出されない。

    出力先はログストリームのみであり、オーケストレータのPhase処理、
    enrichment項目、記憶系統、判断選択に接続するインタフェースが存在しない。

    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない
    - 検出結果を内部処理に供給しない
    - 出力はログストリームへのJSON書き込みのみ
    - 修復・復旧・リセットを行わない
    """

    def __init__(
        self,
        buffer_max: int = _DEFAULT_BUFFER_MAX,
        enabled: Optional[bool] = None,
        sub_window_count: int = _DEFAULT_SUB_WINDOW_COUNT,
    ) -> None:
        """初期化。

        Args:
            buffer_max: スナップショットFIFOバッファの上限サイズ(安全弁7)。
                バッファサイズが検出感度を決定する。
            enabled: 明示的な有効/無効指定。Noneの場合は環境変数で判定。
            sub_window_count: 緩やかな固定化検出のサブウィンドウ分割数。
                バッファ全体をこの数の等分サブウィンドウに分割し、
                各サブウィンドウ内の分散の単調減少を検出する。
                buffer_maxより大きい値は初期化時に検証しエラーにはせず
                検出をスキップする(安全弁8)。
        """
        # 安全弁2: 環境変数による完全無効化
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # (A) スナップショットFIFOバッファ
        self._buffer: deque[_Snapshot] = deque(maxlen=max(1, buffer_max))
        self._buffer_max = max(1, buffer_max)

        # (B) 停止検出状態フラグ(信号種別ごと)
        # 「前回の検出で停止状態であったか」を示す真偽値
        # 初期値は全て「非停止」
        self._stall_flags: dict[str, bool] = {
            signal: False for signal in _ALL_SIGNALS
        }

        # (C) セッション累積カウンタ
        # 信号種別ごとの動態停止検出回数
        self._stall_detected_count: dict[str, int] = {
            signal: 0 for signal in _ALL_SIGNALS
        }
        # 信号種別ごとの動態停止解消回数
        self._stall_resolved_count: dict[str, int] = {
            signal: 0 for signal in _ALL_SIGNALS
        }
        # 全体のスナップショット取得回数
        self._snapshot_count: int = 0

        # ── 緩やかな固定化進行検出(design_gradual_fixation_detection.md) ──

        # (D) サブウィンドウ分割数(生成時に指定、変更されない)
        self._sub_window_count = max(2, sub_window_count)

        # (E) 緩やかな収束検出状態フラグ(信号種別ごと)
        # 「前回の検出で緩やかな収束が進行中であったか」を示す真偽値
        self._gradual_convergence_flags: dict[str, bool] = {
            signal: False for signal in _GRADUAL_CONVERGENCE_SIGNALS
        }

        # (F) 緩やかな収束検出累積カウンタ
        self._gradual_convergence_detected_count: dict[str, int] = {
            signal: 0 for signal in _GRADUAL_CONVERGENCE_SIGNALS
        }
        self._gradual_convergence_resolved_count: dict[str, int] = {
            signal: 0 for signal in _GRADUAL_CONVERGENCE_SIGNALS
        }

        # (G) 最新の散らばり列(信号種別ごと)
        # 最後に算出したサブウィンドウごとの散らばりの列
        self._latest_variance_series: dict[str, list[float]] = {
            signal: [] for signal in _GRADUAL_CONVERGENCE_SIGNALS
        }

    @property
    def enabled(self) -> bool:
        """モニタリングが有効かどうか。"""
        return self._enabled

    @property
    def buffer_size(self) -> int:
        """現在のバッファ内スナップショット数。"""
        return len(self._buffer)

    @property
    def buffer_max(self) -> int:
        """バッファの上限サイズ。"""
        return self._buffer_max

    @property
    def stall_flags(self) -> dict[str, bool]:
        """信号種別ごとの停止検出状態フラグ(読み取り専用コピー)。"""
        return dict(self._stall_flags)

    @property
    def stall_detected_count(self) -> dict[str, int]:
        """信号種別ごとの動態停止検出累積回数(読み取り専用コピー)。"""
        return dict(self._stall_detected_count)

    @property
    def stall_resolved_count(self) -> dict[str, int]:
        """信号種別ごとの動態停止解消累積回数(読み取り専用コピー)。"""
        return dict(self._stall_resolved_count)

    @property
    def snapshot_count(self) -> int:
        """全体のスナップショット取得回数。"""
        return self._snapshot_count

    @property
    def sub_window_count(self) -> int:
        """緩やかな固定化検出のサブウィンドウ分割数。"""
        return self._sub_window_count

    @property
    def gradual_convergence_flags(self) -> dict[str, bool]:
        """信号種別ごとの緩やかな収束検出状態フラグ(読み取り専用コピー)。"""
        return dict(self._gradual_convergence_flags)

    @property
    def gradual_convergence_detected_count(self) -> dict[str, int]:
        """信号種別ごとの緩やかな収束検出累積回数(読み取り専用コピー)。"""
        return dict(self._gradual_convergence_detected_count)

    @property
    def gradual_convergence_resolved_count(self) -> dict[str, int]:
        """信号種別ごとの緩やかな収束解消累積回数(読み取り専用コピー)。"""
        return dict(self._gradual_convergence_resolved_count)

    @property
    def latest_variance_series(self) -> dict[str, list[float]]:
        """信号種別ごとの最新の散らばり列(読み取り専用コピー)。"""
        return {k: list(v) for k, v in self._latest_variance_series.items()}

    # ── 第1段: 窓内スナップショットの蓄積 ──────────────────────────

    def record_snapshot(
        self,
        tick_number: int,
        emotion_values: dict[str, float],
        drive_values: dict[str, float],
        return_pathway_fired: bool,
        enrichment_variation_count: int,
    ) -> None:
        """参照先から状態の断面を読み取り、FIFOバッファに蓄積する。

        オーケストレータの1サイクル処理完了後に呼び出される。
        サイクル処理の途中で呼び出してはならない。

        蓄積される断面は事実の記録であり、断面値の加工・正規化・
        スコアリングは行わない。

        Args:
            tick_number: 現在のティック番号
            emotion_values: 感情ベクトルの各次元値(読み取り専用コピー)
            drive_values: 駆動ベクトルの各次元値(読み取り専用コピー)
            return_pathway_fired: 当該間隔内で帰還経路の発火があったか
            enrichment_variation_count: enrichment項目の変動数
        """
        if not self._enabled:
            return
        try:
            snapshot = _Snapshot(
                tick_number=tick_number,
                emotion_values=emotion_values,
                drive_values=drive_values,
                return_pathway_fired=return_pathway_fired,
                enrichment_variation_count=enrichment_variation_count,
            )
            self._buffer.append(snapshot)
            self._snapshot_count += 1

            # 第2段→第3段を実行
            self._detect_and_alert(tick_number)

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    # ── 第2段: 変化率ゼロの検出 ─────────────────────────────────────

    def _detect_and_alert(self, tick_number: int) -> None:
        """FIFOバッファ内のスナップショットを走査し、動態停止を検出する。

        バッファが満杯でない場合は検出を行わない(不完全なウィンドウからの
        偽検出を防止する)。

        検出は「変化率がゼロであるという事実」の記述であり、
        その状態が「異常」かどうかの判定は含まない。
        """
        try:
            # バッファが満杯でなければ検出をスキップ
            if len(self._buffer) < self._buffer_max:
                return

            snapshots = list(self._buffer)

            # 各信号種別ごとに停止を検出
            current_stall: dict[str, bool] = {}
            stall_values: dict[str, Any] = {}

            # (1) 感情ベクトルの全次元が連続して同一値を示しているか
            emotion_stalled, emotion_val = self._check_emotion_stall(snapshots)
            current_stall[SIGNAL_EMOTION] = emotion_stalled
            if emotion_stalled:
                stall_values[SIGNAL_EMOTION] = emotion_val

            # (2) 駆動ベクトルの全次元が連続して同一値を示しているか
            drive_stalled, drive_val = self._check_drive_stall(snapshots)
            current_stall[SIGNAL_DRIVE] = drive_stalled
            if drive_stalled:
                stall_values[SIGNAL_DRIVE] = drive_val

            # (3) 帰還経路の全経路が連続して発火ゼロを示しているか
            pathway_stalled = self._check_return_pathway_stall(snapshots)
            current_stall[SIGNAL_RETURN_PATHWAY] = pathway_stalled

            # (4) enrichment項目の変動数が連続してゼロを示しているか
            enrichment_stalled = self._check_enrichment_variation_stall(snapshots)
            current_stall[SIGNAL_ENRICHMENT_VARIATION] = enrichment_stalled

            # 第3段: 遷移検出と警告出力
            self._emit_alerts(tick_number, current_stall, stall_values)

            # 第4段: 緩やかな固定化進行検出
            self._detect_gradual_convergence(tick_number, snapshots)

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    def _check_emotion_stall(
        self, snapshots: list[_Snapshot]
    ) -> tuple[bool, Optional[dict[str, float]]]:
        """感情ベクトルの全次元が連続して同一値を示しているか検出する。

        Returns:
            (停止しているか, 停止時の値)
        """
        if not snapshots:
            return False, None

        reference = snapshots[0].emotion_values
        for s in snapshots[1:]:
            if s.emotion_values != reference:
                return False, None

        return True, dict(reference)

    def _check_drive_stall(
        self, snapshots: list[_Snapshot]
    ) -> tuple[bool, Optional[dict[str, float]]]:
        """駆動ベクトルの全次元が連続して同一値を示しているか検出する。

        Returns:
            (停止しているか, 停止時の値)
        """
        if not snapshots:
            return False, None

        reference = snapshots[0].drive_values
        for s in snapshots[1:]:
            if s.drive_values != reference:
                return False, None

        return True, dict(reference)

    def _check_return_pathway_stall(
        self, snapshots: list[_Snapshot]
    ) -> bool:
        """帰還経路の全経路が連続して発火ゼロを示しているか検出する。"""
        for s in snapshots:
            if s.return_pathway_fired:
                return False
        return True

    def _check_enrichment_variation_stall(
        self, snapshots: list[_Snapshot]
    ) -> bool:
        """enrichment項目の変動数が連続してゼロを示しているか検出する。"""
        for s in snapshots:
            if s.enrichment_variation_count > 0:
                return False
        return True

    # ── 第4段: 緩やかな固定化進行検出 ────────────────────────────────

    def _detect_gradual_convergence(
        self, tick_number: int, snapshots: list[_Snapshot]
    ) -> None:
        """サブウィンドウ内の分散が単調減少しているかを検出する。

        設計書: design_gradual_fixation_detection.md

        対象は感情・駆動信号のみ。帰還経路発火(真偽値)とenrichment変動(整数)は
        分散の単調減少検出の対象外。

        安全弁8: サブウィンドウの構成に必要な最低限のスナップショット数に
        達していない場合は検出を行わない。
        """
        try:
            n = len(snapshots)
            swc = self._sub_window_count

            # バッファ不足時のスキップ(安全弁8):
            # 各サブウィンドウに最低2件のスナップショットが必要
            # (分散を意味ある形で算出するため)
            if n < swc * 2:
                return

            # サブウィンドウのサイズ(等分、余りは末尾に含める)
            base_size = n // swc

            current_convergence: dict[str, bool] = {}
            current_variance_series: dict[str, list[float]] = {}

            for signal in _GRADUAL_CONVERGENCE_SIGNALS:
                variance_series = self._compute_variance_series(
                    snapshots, signal, swc, base_size, n
                )
                current_variance_series[signal] = variance_series
                self._latest_variance_series[signal] = list(variance_series)

                # 単調減少の判定: 全ての隣接ペアで後者が前者以下
                is_converging = self._is_monotonically_decreasing(variance_series)
                current_convergence[signal] = is_converging

            # 遷移検出と記録出力
            self._emit_gradual_convergence_alerts(
                tick_number, current_convergence, current_variance_series
            )

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    def _compute_variance_series(
        self,
        snapshots: list[_Snapshot],
        signal: str,
        sub_window_count: int,
        base_size: int,
        total: int,
    ) -> list[float]:
        """サブウィンドウごとの分散列を算出する。

        各サブウィンドウ内で、対象信号の全次元値の分散(平均偏差の二乗和の平均)を
        算出する。複数次元がある場合は全次元の分散の合計を使用する。

        Args:
            snapshots: スナップショットの一覧
            signal: 信号種別(SIGNAL_EMOTION or SIGNAL_DRIVE)
            sub_window_count: サブウィンドウ分割数
            base_size: 各サブウィンドウの基本サイズ
            total: スナップショット総数

        Returns:
            サブウィンドウごとの分散値の列
        """
        variances: list[float] = []

        for i in range(sub_window_count):
            start = i * base_size
            # 末尾のサブウィンドウに余りを含める
            if i == sub_window_count - 1:
                end = total
            else:
                end = (i + 1) * base_size

            sub_snapshots = snapshots[start:end]
            if not sub_snapshots:
                variances.append(0.0)
                continue

            # 各次元の値列を収集
            if signal == SIGNAL_EMOTION:
                all_values_by_dim = self._extract_dimension_values(
                    sub_snapshots, "emotion_values"
                )
            else:  # SIGNAL_DRIVE
                all_values_by_dim = self._extract_dimension_values(
                    sub_snapshots, "drive_values"
                )

            # 全次元の分散の合計
            total_variance = 0.0
            for dim_values in all_values_by_dim.values():
                if len(dim_values) < 2:
                    continue
                mean = sum(dim_values) / len(dim_values)
                var = sum((v - mean) ** 2 for v in dim_values) / len(dim_values)
                total_variance += var

            variances.append(total_variance)

        return variances

    def _extract_dimension_values(
        self,
        snapshots: list[_Snapshot],
        attr_name: str,
    ) -> dict[str, list[float]]:
        """スナップショット列から指定属性の次元ごとの値列を抽出する。

        Args:
            snapshots: スナップショットの一覧
            attr_name: 属性名("emotion_values" or "drive_values")

        Returns:
            次元名→値のリスト
        """
        result: dict[str, list[float]] = {}
        for s in snapshots:
            values = getattr(s, attr_name)
            for dim, val in values.items():
                if dim not in result:
                    result[dim] = []
                result[dim].append(val)
        return result

    @staticmethod
    def _is_monotonically_decreasing(series: list[float]) -> bool:
        """列が単調に減少しているか(各要素が直前の要素以下であるか)を判定する。

        列が2未満の場合はFalseを返す(単調減少の判定に十分なペアがない)。
        全要素が同一の場合もTrue(非増加=減少or同値)だが、
        全要素がゼロの場合はFalseを返す(既に最小値であり収束を検出する意味がない)。

        Args:
            series: 分散値の列

        Returns:
            単調減少であればTrue
        """
        if len(series) < 2:
            return False

        # 全要素がゼロの場合は「収束進行中」とは判定しない
        if all(v == 0.0 for v in series):
            return False

        for i in range(1, len(series)):
            if series[i] > series[i - 1]:
                return False

        return True

    def _emit_gradual_convergence_alerts(
        self,
        tick_number: int,
        current_convergence: dict[str, bool],
        variance_series: dict[str, list[float]],
    ) -> None:
        """緩やかな収束状態の遷移を検出し、遷移時にのみ記録ログを出力する。

        安全弁7: 収束継続中は出力しない(冗長抑制)。

        Args:
            tick_number: 現在のティック番号
            current_convergence: 今回の検出結果(信号種別→収束進行中か)
            variance_series: 信号種別ごとのサブウィンドウ分散列
        """
        try:
            for signal in _GRADUAL_CONVERGENCE_SIGNALS:
                was_converging = self._gradual_convergence_flags.get(signal, False)
                is_converging = current_convergence.get(signal, False)

                if is_converging and not was_converging:
                    # 非収束→収束: 収束進行検出ログを出力
                    self._gradual_convergence_flags[signal] = True
                    self._gradual_convergence_detected_count[signal] = (
                        self._gradual_convergence_detected_count.get(signal, 0) + 1
                    )

                    record: dict[str, Any] = {
                        "type": "gradual_convergence_detected",
                        "timestamp": time.time(),
                        "signal": signal,
                        "tick_number": tick_number,
                        "sub_window_count": self._sub_window_count,
                        "variance_series": variance_series.get(signal, []),
                    }
                    self._emit_json(record)

                elif not is_converging and was_converging:
                    # 収束→非収束: 解消ログを出力
                    self._gradual_convergence_flags[signal] = False
                    self._gradual_convergence_resolved_count[signal] = (
                        self._gradual_convergence_resolved_count.get(signal, 0) + 1
                    )

                    record = {
                        "type": "gradual_convergence_resolved",
                        "timestamp": time.time(),
                        "signal": signal,
                        "tick_number": tick_number,
                    }
                    self._emit_json(record)

                # 収束継続中 or 非収束継続中: ログ出力しない(安全弁7)

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    # ── 第3段: 警告の出力 ──────────────────────────────────────────

    def _emit_alerts(
        self,
        tick_number: int,
        current_stall: dict[str, bool],
        stall_values: dict[str, Any],
    ) -> None:
        """停止状態の遷移を検出し、遷移時にのみ警告/解消ログを出力する。

        安全弁8: 同一の停止状態が継続している間、毎スナップショットで
        警告を出力するのではなく、停止状態への遷移時と解消時にのみ出力する。

        Args:
            tick_number: 現在のティック番号
            current_stall: 今回の検出結果(信号種別→停止しているか)
            stall_values: 停止時の値(信号種別→停止時の値)
        """
        try:
            for signal in _ALL_SIGNALS:
                was_stalled = self._stall_flags.get(signal, False)
                is_stalled = current_stall.get(signal, False)

                if is_stalled and not was_stalled:
                    # 非停止→停止: 停止検出ログを出力
                    self._stall_flags[signal] = True
                    self._stall_detected_count[signal] = (
                        self._stall_detected_count.get(signal, 0) + 1
                    )

                    record: dict[str, Any] = {
                        "type": "dynamics_stall_detected",
                        "timestamp": time.time(),
                        "signal": signal,
                        "tick_number": tick_number,
                        "consecutive_snapshots": len(self._buffer),
                    }
                    if signal in stall_values:
                        record["stall_value"] = stall_values[signal]

                    self._emit_json(record)

                elif not is_stalled and was_stalled:
                    # 停止→非停止: 解消ログを出力
                    self._stall_flags[signal] = False
                    self._stall_resolved_count[signal] = (
                        self._stall_resolved_count.get(signal, 0) + 1
                    )

                    record = {
                        "type": "dynamics_stall_resolved",
                        "timestamp": time.time(),
                        "signal": signal,
                        "tick_number": tick_number,
                    }
                    self._emit_json(record)

                # 停止継続中 or 非停止継続中: ログ出力しない(安全弁8)

        except Exception:
            # 安全弁1: 計測失敗時の安全な無視
            pass

    # ── セッションサマリ出力 ─────────────────────────────────────

    def emit_session_summary(self) -> Optional[dict[str, Any]]:
        """セッション終了時のセッション累積情報を出力する。

        Returns:
            セッションサマリの辞書。無効時はNone。
        """
        if not self._enabled:
            return None
        try:
            summary: dict[str, Any] = {
                "type": "anomaly_detection_session_summary",
                "timestamp": time.time(),
                "snapshot_count": self._snapshot_count,
                "buffer_max": self._buffer_max,
                "stall_detected_counts": dict(self._stall_detected_count),
                "stall_resolved_counts": dict(self._stall_resolved_count),
                "current_stall_flags": dict(self._stall_flags),
                "sub_window_count": self._sub_window_count,
                "gradual_convergence_detected_counts": dict(
                    self._gradual_convergence_detected_count
                ),
                "gradual_convergence_resolved_counts": dict(
                    self._gradual_convergence_resolved_count
                ),
                "current_gradual_convergence_flags": dict(
                    self._gradual_convergence_flags
                ),
            }

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
            "snapshot_count": self._snapshot_count,
            "buffer_size": len(self._buffer),
            "buffer_max": self._buffer_max,
            "stall_detected_counts": dict(self._stall_detected_count),
            "stall_resolved_counts": dict(self._stall_resolved_count),
            "current_stall_flags": dict(self._stall_flags),
            "latest_snapshot": (
                self._buffer[-1].to_dict() if self._buffer else None
            ),
            "sub_window_count": self._sub_window_count,
            "gradual_convergence_detected_counts": dict(
                self._gradual_convergence_detected_count
            ),
            "gradual_convergence_resolved_counts": dict(
                self._gradual_convergence_resolved_count
            ),
            "current_gradual_convergence_flags": dict(
                self._gradual_convergence_flags
            ),
            "latest_variance_series": {
                k: list(v) for k, v in self._latest_variance_series.items()
            },
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


# ── ヘルパー: オーケストレータからのスナップショット読み取り ──────


def collect_snapshot_from_orchestrator(
    orchestrator: Any,
    return_pathway_monitor: Any = None,
    enrichment_distribution_monitor: Any = None,
    last_tick_had_pathway_firing: Optional[bool] = None,
    last_tick_enrichment_changed: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """オーケストレータと既存ツール群から状態断面を読み取り、辞書で返す。

    既存の状態スナップショット読み取り関数と同等のパスを使用する。
    新しい読み取りパスを追加しない。

    全ての読み取りはREAD-ONLYであり、参照先への書き込みを行わない。

    Args:
        orchestrator: PsycheOrchestratorインスタンス
        return_pathway_monitor: ReturnPathwayMonitorインスタンス(Optional)
        enrichment_distribution_monitor: EnrichmentDistributionMonitorインスタンス(Optional)
        last_tick_had_pathway_firing: 直接指定する帰還経路発火有無(Optional)
        last_tick_enrichment_changed: 直接指定するenrichment変動数(Optional)

    Returns:
        スナップショット情報の辞書。読み取り失敗時はNone。
    """
    try:
        # ティック番号
        tick_number = orchestrator._tick_count

        # 感情ベクトル
        psyche = orchestrator._psyche
        emotion_values = psyche.emotions.as_dict()

        # 駆動ベクトル
        drive_values = psyche.drives.as_dict()

        # 帰還経路の発火有無
        pathway_fired = False
        if last_tick_had_pathway_firing is not None:
            pathway_fired = last_tick_had_pathway_firing
        elif return_pathway_monitor is not None:
            try:
                last_record = return_pathway_monitor.last_tick_record
                pathway_fired = last_record is not None
            except Exception:
                pathway_fired = False

        # enrichment変動数
        enrichment_changed = 0
        if last_tick_enrichment_changed is not None:
            enrichment_changed = last_tick_enrichment_changed
        elif enrichment_distribution_monitor is not None:
            try:
                dist_summary = enrichment_distribution_monitor.get_distribution_summary()
                latest = dist_summary.get("latest_entry")
                if latest:
                    enrichment_changed = latest.get("total_changed", 0)
            except Exception:
                enrichment_changed = 0

        return {
            "tick_number": tick_number,
            "emotion_values": emotion_values,
            "drive_values": drive_values,
            "return_pathway_fired": pathway_fired,
            "enrichment_variation_count": enrichment_changed,
        }

    except Exception:
        # 安全弁1: 読み取り失敗時はNoneを返す
        return None
