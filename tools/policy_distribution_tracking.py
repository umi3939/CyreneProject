"""
tools/policy_distribution_tracking.py - ポリシー選択分布の経時的変化追跡

ポリシー選択分布のセッション内での経時的変化を追跡する。
既存の分析基盤(policy_selection_analysis.py)を拡張する形で、
分布スナップショットの定期的蓄積、文脈別選択分布の構成、
分布推移の数値的記述を行う。

設計書: design_policy_distribution_tracking.md

本機能の構造的分離:
- psycheモジュール群には配置しない。外部ツール群(tools/)に配置する。
- 方針選択処理の入力（状態・知覚・記憶・責任影響・判断バイアス・拡張入力）を一切変更しない
- 方針選択処理の出力（候補リスト・選択結果）を一切変更しない
- ポリシー選択ログ基盤の蓄積データを読み取るのみであり、書き込まない
- prompt enrichment への出力経路を構造的に持たない
- 方針選択処理への逆流経路を構造的に持たない
- 全内部状態はセッション境界で消失する（永続化対象外）
- save/loadの対象フィールドに一切追加しない
- 本機能の有効/無効がpsycheの動作に影響しない（完全に透過的）
- 記録された分布データが将来の分析に影響を与える自己参照構造を持たない

安全弁:
1. 規範的判断の構造的排除: 数値的事実のみ出力。評価的語彙を含めない。
   選択分布に「望ましい形」「目標分布」を設定しない
2. 推奨生成の禁止: 分析結果に基づく推奨を生成する仕組みを持たない
3. FIFO上限によるメモリ制御: スナップショットリスト・文脈別分布・推移記録の
   全てにFIFO上限を設ける。上限超過時は古いデータから消失する
4. 環境変数による完全無効化: CYRENE_MONITOR=1 で有効化
5. セッション境界での完全消失: save/loadフィールドへの追加を行わない
6. 文脈条件の組み合わせ爆発防止: 文脈条件の種類数に上限を設ける
7. 分析結果の状態非依存性: 各分析はその時点のログ基盤の生データのみから
   算出される。過去の分析結果が将来の分析に影響しない
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

monitor_logger = logging.getLogger("cyrene.monitor")


# ── 環境変数制御 ──────────────────────────────────────────────────


def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存の実行時観測基盤と同じ環境変数(CYRENE_MONITOR)に従う。
    安全弁4: 無効時は全ての処理をスキップし、空の結果を返す。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── 設定 ──────────────────────────────────────────────────────────


@dataclass
class PolicyDistributionTrackingConfig:
    """ポリシー選択分布追跡の設定パラメータ。

    Attributes:
        snapshot_interval: スナップショットを記録するティック間隔。
        max_snapshots: 分布スナップショットのFIFO上限（安全弁3）。
        max_transitions: 推移記録のFIFO上限（安全弁3）。
        max_context_keys: 文脈条件の種類数上限（安全弁6）。
        max_context_entries: 各文脈条件あたりのエントリ数FIFO上限（安全弁3）。
    """
    snapshot_interval: int = 50
    max_snapshots: int = 30
    max_transitions: int = 29
    max_context_keys: int = 27
    max_context_entries: int = 200

    def __post_init__(self) -> None:
        """設定値のバリデーション。不正な値はデフォルトに戻す。"""
        if self.snapshot_interval < 1:
            self.snapshot_interval = 50
        if self.max_snapshots < 1:
            self.max_snapshots = 30
        if self.max_transitions < 1:
            self.max_transitions = 29
        if self.max_context_keys < 1:
            self.max_context_keys = 27
        if self.max_context_entries < 1:
            self.max_context_entries = 200


# ── 文脈段階値の離散化 ─────────────────────────────────────────────


def _discretize_drive_level(max_drive: float) -> str:
    """駆動の最大値を段階値に離散化する。

    3段階: low / mid / high
    """
    if max_drive < 0.33:
        return "low"
    elif max_drive < 0.67:
        return "mid"
    else:
        return "high"


def _discretize_valence(valence: float) -> str:
    """ムードのvalenceを極性段階値に離散化する。

    3段階: negative / neutral / positive
    """
    if valence < -0.2:
        return "negative"
    elif valence > 0.2:
        return "positive"
    else:
        return "neutral"


def _discretize_arousal(arousal: float) -> str:
    """覚醒度を段階値に離散化する。

    3段階: low / mid / high
    """
    if arousal < 0.33:
        return "low"
    elif arousal < 0.67:
        return "mid"
    else:
        return "high"


def build_context_key(
    max_drive: float = 0.0,
    valence: float = 0.0,
    arousal: float = 0.0,
) -> str:
    """内部状態スナップショットから文脈条件キーを構築する。

    Args:
        max_drive: 駆動の最大値
        valence: ムードのvalence
        arousal: 覚醒度

    Returns:
        文脈条件キー文字列（例: "drive=mid|valence=positive|arousal=high"）
    """
    d = _discretize_drive_level(max_drive)
    v = _discretize_valence(valence)
    a = _discretize_arousal(arousal)
    return f"drive={d}|valence={v}|arousal={a}"


# ── 分布スナップショット ───────────────────────────────────────────


@dataclass
class DistributionSnapshot:
    """1つの時点でのポリシー選択分布のスナップショット。

    安全弁1: 数値的事実のみを格納する。評価的判断を含めない。
    安全弁7: 過去のスナップショットが将来のスナップショットの算出に影響しない。

    Attributes:
        tick: 記録時点のティック番号
        label_counts: ポリシーラベル別の選択回数
        label_ratios: ポリシーラベル別の正規化比率
        concentration_level: 集中度の段階値
        record_count: スナップショットを構成した記録件数
    """
    tick: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    label_ratios: dict[str, float] = field(default_factory=dict)
    concentration_level: str = "none"
    record_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """スナップショットを辞書形式に変換する。"""
        return {
            "tick": self.tick,
            "label_counts": dict(self.label_counts),
            "label_ratios": {
                k: round(v, 6) for k, v in self.label_ratios.items()
            },
            "concentration_level": self.concentration_level,
            "record_count": self.record_count,
        }


# ── 推移記録 ───────────────────────────────────────────────────────


@dataclass
class DistributionTransition:
    """隣接するスナップショット間の変化量を記述する。

    安全弁1: 因果帰属は行わない（「なぜ変化したか」は記述しない）。

    Attributes:
        from_tick: 変化元のティック番号
        to_tick: 変化先のティック番号
        label_ratio_deltas: ポリシーラベル別比率の変化量
    """
    from_tick: int = 0
    to_tick: int = 0
    label_ratio_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """推移記録を辞書形式に変換する。"""
        return {
            "from_tick": self.from_tick,
            "to_tick": self.to_tick,
            "label_ratio_deltas": {
                k: round(v, 6) for k, v in self.label_ratio_deltas.items()
            },
        }


# ── 文脈別選択分布 ─────────────────────────────────────────────────


@dataclass
class ContextDistribution:
    """1つの文脈条件に対するポリシーラベル別選択回数。

    Attributes:
        context_key: 文脈条件キー
        label_counts: ポリシーラベル別の選択回数
        total_count: 全選択回数
    """
    context_key: str = ""
    label_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """文脈別分布を辞書形式に変換する。"""
        total = self.total_count if self.total_count > 0 else 1
        ratios = {
            k: round(v / total, 6) for k, v in self.label_counts.items()
        }
        return {
            "context_key": self.context_key,
            "label_counts": dict(self.label_counts),
            "label_ratios": ratios,
            "total_count": self.total_count,
        }


# ── 集中度の段階値算出 ─────────────────────────────────────────────


def _compute_concentration_level(label_counts: dict[str, int]) -> str:
    """最頻ポリシーの比率から集中度の段階値を算出する。

    安全弁1: 段階値のみ。「偏りすぎ」等の評価的語彙を含めない。

    Returns:
        集中度段階値: "none" / "low" / "mid" / "high" / "very_high"
    """
    if not label_counts:
        return "none"

    total = sum(label_counts.values())
    if total == 0:
        return "none"

    max_count = max(label_counts.values())
    ratio = max_count / total

    if ratio < 0.3:
        return "low"
    elif ratio < 0.5:
        return "mid"
    elif ratio < 0.7:
        return "high"
    else:
        return "very_high"


# ── 時間的安定度の段階値算出 ───────────────────────────────────────


def _compute_stability_level(transitions: list[DistributionTransition]) -> str:
    """隣接間変化量の大きさから時間的安定度の段階値を算出する。

    安全弁1: 段階値のみ。評価的語彙を含めない。

    Returns:
        安定度段階値: "none" / "stable" / "moderate" / "volatile"
    """
    if not transitions:
        return "none"

    total_change = 0.0
    count = 0
    for t in transitions:
        for delta in t.label_ratio_deltas.values():
            total_change += abs(delta)
            count += 1

    if count == 0:
        return "none"

    avg_change = total_change / count

    if avg_change < 0.05:
        return "stable"
    elif avg_change < 0.15:
        return "moderate"
    else:
        return "volatile"


# ── PolicyDistributionTracking 本体 ──────────────────────────────


class PolicyDistributionTracking:
    """ポリシー選択分布の経時的変化追跡を行う本体クラス。

    セッション開始時にインスタンスを生成し、セッション終了時に破棄する。
    全内部状態はインスタンス破棄時に消失する（永続化対象外）。
    安全弁5: セッション境界で全データが消失する。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない（READ-ONLY観測のみ）
    - prompt enrichment への出力経路を持たない
    - 方針選択処理への逆流経路を持たない
    - ポリシー選択ログ基盤への書き込みを行わない
    """

    def __init__(
        self,
        config: Optional[PolicyDistributionTrackingConfig] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            config: 設定。Noneの場合はデフォルト設定を使用する。
            enabled: 有効/無効の明示的指定。Noneの場合は環境変数で判定する。
        """
        self._config = config or PolicyDistributionTrackingConfig()
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # データ構造1: 分布スナップショットのFIFOリスト（安全弁3）
        self._snapshots: list[DistributionSnapshot] = []

        # データ構造2: 文脈別選択分布の辞書（安全弁6: 種類数上限あり）
        self._context_distributions: dict[str, ContextDistribution] = {}

        # データ構造3: 推移記録のFIFOリスト（安全弁3）
        self._transitions: list[DistributionTransition] = []

        # 最後にスナップショットを記録したティック番号
        self._last_snapshot_tick: int = 0

    @property
    def enabled(self) -> bool:
        """追跡が有効かどうかを返す。"""
        return self._enabled

    @property
    def config(self) -> PolicyDistributionTrackingConfig:
        """設定を読み取り専用で返す。"""
        return self._config

    @property
    def snapshot_count(self) -> int:
        """現在保持しているスナップショットの件数を返す。"""
        return len(self._snapshots)

    @property
    def transition_count(self) -> int:
        """現在保持している推移記録の件数を返す。"""
        return len(self._transitions)

    @property
    def context_key_count(self) -> int:
        """現在保持している文脈条件の種類数を返す。"""
        return len(self._context_distributions)

    # ── 処理A: 定期的な分布スナップショットの蓄積 ─────────────────

    def record_snapshot(
        self,
        tick: int,
        entries: list[dict[str, Any]],
    ) -> Optional[DistributionSnapshot]:
        """ポリシー選択ログの直近記録から分布スナップショットを蓄積する。

        一定ティック間隔で呼び出され、直近の選択記録から分布を算出し
        スナップショットとして蓄積する。

        安全弁7: 各スナップショットはその時点のログ基盤の生データのみから
        算出される。過去のスナップショットが将来のスナップショットの算出に
        影響しない。

        Args:
            tick: 現在のティック番号
            entries: ポリシー選択ログのエントリ辞書リスト。
                PolicySelectionLog.get_entries() の戻り値を想定する。

        Returns:
            記録されたスナップショット。記録条件を満たさない場合はNone。
        """
        if not self._enabled:
            return None

        if not entries:
            return None

        # 定期的なティック間隔チェック
        if self._last_snapshot_tick > 0:
            elapsed = tick - self._last_snapshot_tick
            if elapsed < self._config.snapshot_interval:
                return None

        # ポリシーラベル別の選択回数を算出する
        label_counts: dict[str, int] = defaultdict(int)
        for entry in entries:
            label = entry.get("selected_label", "")
            if label:
                label_counts[label] += 1

        label_counts_dict = dict(label_counts)

        # 正規化比率を算出する
        total = sum(label_counts_dict.values())
        label_ratios: dict[str, float] = {}
        if total > 0:
            for label, count in label_counts_dict.items():
                label_ratios[label] = count / total

        # 集中度の段階値を算出する
        concentration = _compute_concentration_level(label_counts_dict)

        snapshot = DistributionSnapshot(
            tick=tick,
            label_counts=label_counts_dict,
            label_ratios=label_ratios,
            concentration_level=concentration,
            record_count=len(entries),
        )

        # 蓄積する（安全弁3: FIFO上限）
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._config.max_snapshots:
            self._snapshots = self._snapshots[-self._config.max_snapshots:]

        self._last_snapshot_tick = tick

        # 推移記録の自動生成（処理C）
        if len(self._snapshots) >= 2:
            prev = self._snapshots[-2]
            curr = self._snapshots[-1]
            self._record_transition(prev, curr)

        return snapshot

    # ── 処理B: 文脈別の選択分布の構成 ────────────────────────────

    def record_context_entry(
        self,
        selected_label: str,
        max_drive: float = 0.0,
        valence: float = 0.0,
        arousal: float = 0.0,
    ) -> Optional[str]:
        """選択記録を文脈別に分類して蓄積する。

        各選択記録を、記録時点の内部状態スナップショットの段階値で分類し、
        文脈条件ごとに独立したポリシーラベル別選択回数を蓄積する。

        安全弁6: 文脈条件の種類数に上限を設ける。

        Args:
            selected_label: 選択されたポリシーラベル
            max_drive: 駆動の最大値
            valence: ムードのvalence
            arousal: 覚醒度

        Returns:
            使用した文脈条件キー。無効時はNone。
        """
        if not self._enabled:
            return None

        if not selected_label:
            return None

        context_key = build_context_key(max_drive, valence, arousal)

        # 安全弁6: 文脈条件の種類数上限チェック
        if context_key not in self._context_distributions:
            if len(self._context_distributions) >= self._config.max_context_keys:
                return None
            self._context_distributions[context_key] = ContextDistribution(
                context_key=context_key,
            )

        dist = self._context_distributions[context_key]

        # 安全弁3: 各文脈条件のエントリ数FIFO上限
        if dist.total_count >= self._config.max_context_entries:
            # 最も古い（最も少ない）ラベルの回数を1減らして空きを作る
            if dist.label_counts:
                min_label = min(dist.label_counts, key=dist.label_counts.get)  # type: ignore[arg-type]
                dist.label_counts[min_label] -= 1
                if dist.label_counts[min_label] <= 0:
                    del dist.label_counts[min_label]
                dist.total_count -= 1

        # 選択回数を加算する
        if selected_label not in dist.label_counts:
            dist.label_counts[selected_label] = 0
        dist.label_counts[selected_label] += 1
        dist.total_count += 1

        return context_key

    # ── 処理C: 分布推移の数値的記述 ──────────────────────────────

    def _record_transition(
        self,
        prev: DistributionSnapshot,
        curr: DistributionSnapshot,
    ) -> DistributionTransition:
        """隣接するスナップショット間の変化量を算出し記録する。

        安全弁1: 因果帰属は行わない（「なぜ変化したか」は記述しない）。

        Args:
            prev: 変化元のスナップショット
            curr: 変化先のスナップショット

        Returns:
            推移記録
        """
        # ポリシーラベル別比率の変化量を算出する
        all_labels = set(prev.label_ratios.keys()) | set(curr.label_ratios.keys())
        label_ratio_deltas: dict[str, float] = {}
        for label in all_labels:
            prev_ratio = prev.label_ratios.get(label, 0.0)
            curr_ratio = curr.label_ratios.get(label, 0.0)
            label_ratio_deltas[label] = curr_ratio - prev_ratio

        transition = DistributionTransition(
            from_tick=prev.tick,
            to_tick=curr.tick,
            label_ratio_deltas=label_ratio_deltas,
        )

        # 蓄積する（安全弁3: FIFO上限）
        self._transitions.append(transition)
        if len(self._transitions) > self._config.max_transitions:
            self._transitions = self._transitions[-self._config.max_transitions:]

        return transition

    # ── アクセサ（読み取り専用） ──────────────────────────────────

    def get_snapshots(self) -> list[dict[str, Any]]:
        """分布スナップショットを辞書形式で返す。読み取り専用アクセサ。"""
        return [s.to_dict() for s in self._snapshots]

    def get_transitions(self) -> list[dict[str, Any]]:
        """推移記録を辞書形式で返す。読み取り専用アクセサ。"""
        return [t.to_dict() for t in self._transitions]

    def get_context_distributions(self) -> dict[str, dict[str, Any]]:
        """文脈別選択分布を辞書形式で返す。読み取り専用アクセサ。"""
        return {
            key: dist.to_dict()
            for key, dist in self._context_distributions.items()
        }

    def get_stability_level(self) -> str:
        """現在の推移記録から時間的安定度の段階値を返す。

        安全弁1: 段階値のみ。評価的語彙を含めない。
        """
        return _compute_stability_level(self._transitions)

    # ── レポート生成 ─────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """全追跡結果を構造化辞書として整理し返却する。

        安全弁1: 評価的語彙を出力に含めない。事実記述のみ。
        安全弁2: 推奨を生成しない。

        Returns:
            構造化されたレポート辞書
        """
        if not self._enabled:
            return {
                "type": "policy_distribution_tracking_report",
                "enabled": False,
                "timestamp": time.time(),
            }

        report: dict[str, Any] = {
            "type": "policy_distribution_tracking_report",
            "enabled": True,
            "timestamp": time.time(),
            "snapshots": self.get_snapshots(),
            "transitions": self.get_transitions(),
            "context_distributions": self.get_context_distributions(),
            "snapshot_count": self.snapshot_count,
            "transition_count": self.transition_count,
            "context_key_count": self.context_key_count,
            "stability_level": self.get_stability_level(),
        }

        # ログストリームにJSON形式で出力する
        try:
            text = json.dumps(report, ensure_ascii=False, default=str)
            monitor_logger.debug(text)
        except Exception:
            pass

        return report

    # ── 便利メソッド: ログ基盤からの直接追跡 ─────────────────────

    def track_from_log(
        self,
        log_instance: Any,
        tick: int,
    ) -> Optional[DistributionSnapshot]:
        """PolicySelectionLog インスタンスから直接ログを読み取りスナップショットを記録する。

        読み取り専用アクセサ(get_entries)のみを使用する。
        ポリシー選択ログ基盤の内部状態を変更しない。

        Args:
            log_instance: PolicySelectionLog のインスタンス。
                get_entries() メソッドを持つこと。
            tick: 現在のティック番号

        Returns:
            記録されたスナップショット。条件を満たさない場合はNone。
        """
        if not self._enabled:
            return None

        try:
            entries = log_instance.get_entries()
        except Exception:
            return None

        return self.record_snapshot(tick, entries)

    # ── 既存分析基盤との統合 ─────────────────────────────────────

    def extend_analysis_report(
        self,
        analysis_report: dict[str, Any],
    ) -> dict[str, Any]:
        """既存の分析基盤のレポートに追跡結果を追加フィールドとして含める。

        既存のレポート構造を破壊しない（追加フィールドとして含める）。

        Args:
            analysis_report: PolicySelectionAnalysis.generate_report()の戻り値

        Returns:
            追跡結果が追加されたレポート辞書
        """
        if not self._enabled:
            analysis_report["distribution_tracking"] = {
                "enabled": False,
            }
            return analysis_report

        analysis_report["distribution_tracking"] = {
            "enabled": True,
            "snapshots": self.get_snapshots(),
            "transitions": self.get_transitions(),
            "context_distributions": self.get_context_distributions(),
            "snapshot_count": self.snapshot_count,
            "transition_count": self.transition_count,
            "context_key_count": self.context_key_count,
            "stability_level": self.get_stability_level(),
        }

        return analysis_report


# ── ファクトリ関数 ────────────────────────────────────────────────


def create_policy_distribution_tracking(
    config: Optional[PolicyDistributionTrackingConfig] = None,
    enabled: Optional[bool] = None,
) -> PolicyDistributionTracking:
    """PolicyDistributionTracking のファクトリ関数。

    Args:
        config: 設定。Noneの場合はデフォルト設定を使用する。
        enabled: 有効/無効の明示的指定。Noneの場合は環境変数で判定する。

    Returns:
        PolicyDistributionTracking インスタンス
    """
    return PolicyDistributionTracking(config=config, enabled=enabled)
