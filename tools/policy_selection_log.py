"""
tools/policy_selection_log.py - ポリシー選択ログの構造化と分析基盤

方針選択処理のスコア内訳（断面別寄与量）を心理処理パイプラインの外側で
蓄積・集計し、開発者向けの分析手段として構造化する。

設計書: design_policy_selection_log.md

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)とアクセサ返却のみ
- prompt enrichment への接続経路を一切持たない
- 方針選択処理の候補生成・スコアリング・最終選択への入力経路を持たない
- 偏りの検出結果に基づく推奨・判定を行わない
- 内部システムの状態変数を一切変更しない(READ-ONLY観測のみ)
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない

安全弁:
1. enrichment経路の構造的遮断: enrichment出力を生成する関数を持たない
2. 永続化の対象外: save/loadフィールド追加なし
3. 集計の事実記述限定: 数値的事実のみ、評価的判断を含めない
4. 蓄積量のFIFO上限: 設定により上限を制御
5. 環境変数による完全無効化: CYRENE_MONITOR=1 で有効化
"""

# Python 3.10未満でも型ヒントの新記法を使えるようにする
from __future__ import annotations

# JSON形式でのログ出力に使う
import json
# ログ出力の仕組みを使う
import logging
# 環境変数の読み取りに使う
import os
# タイムスタンプの生成に使う
import time
# デフォルト値付き辞書（キーが存在しない場合に自動で初期値を設定する辞書）
from collections import defaultdict
# データクラス（構造体のようなもの）を定義するためのデコレータ
from dataclasses import dataclass, field
# 型ヒント用
from typing import Any, Optional

# 既存の実行時観測基盤(execution_monitor.py)と同じログ名前空間を使用する
# これにより、同じログファイルに出力される
monitor_logger = logging.getLogger("cyrene.monitor")


# ── 環境変数制御 ──────────────────────────────────────────────────


def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存の実行時観測基盤と同じ環境変数(CYRENE_MONITOR)に従う。
    安全弁5: 無効時は蓄積・集計・出力を全て省略する。
    """
    # 環境変数 CYRENE_MONITOR が "1" のときだけ有効にする
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── 設定 ──────────────────────────────────────────────────────────


@dataclass
class PolicySelectionLogConfig:
    """ポリシー選択ログの設定パラメータ。

    Attributes:
        max_log_entries: 蓄積リストのFIFO上限（安全弁4）。
            この件数を超えると古い記録が押し出される。
        aggregation_window: 窓内集計に使用する直近件数。
            集計はこの件数分の直近記録に対してのみ行う。
    """
    # 蓄積リストに保持する最大エントリ数（安全弁4: FIFO上限）
    max_log_entries: int = 500
    # 窓内集計で参照する直近エントリ数
    aggregation_window: int = 50

    def __post_init__(self) -> None:
        """設定値のバリデーション。不正な値はデフォルトに戻す。"""
        # 最大エントリ数が1未満なら500にリセット
        if self.max_log_entries < 1:
            self.max_log_entries = 500
        # 集計窓サイズが1未満なら50にリセット
        if self.aggregation_window < 1:
            self.aggregation_window = 50
        # 集計窓サイズが最大エントリ数を超えていたら、最大エントリ数に合わせる
        if self.aggregation_window > self.max_log_entries:
            self.aggregation_window = self.max_log_entries


# ── ログエントリ ──────────────────────────────────────────────────


@dataclass
class ScoreLogEntry:
    """1回のポリシー選択に対するスコアログエントリ。

    一度生成された記録は変更されない（不変）。
    方針選択処理が実行されるたびに1件生成される。

    Attributes:
        tick: ティック番号（何回目の処理サイクルか）
        timestamp: タイムスタンプ（Unix時刻、浮動小数点）
        selected_label: 最終的に選択されたポリシーの名前
        candidates: 全候補の情報リスト（各候補のラベル、スコア、内訳）
        candidate_count: 候補の総数
        selected_count: 動的選出で実際に返された候補の数（3〜5件）
    """
    # この記録が作られた時点のティック番号
    tick: int
    # この記録が作られた時点のUnix時刻
    timestamp: float
    # 最終的に選ばれたポリシーの名前（例: "共感する"）
    selected_label: str
    # 全候補の情報（各候補にはラベル・スコア・内訳が含まれる）
    candidates: list[dict[str, Any]]
    # 候補の総数（拡張候補を含む全体数）
    candidate_count: int
    # 動的選出で実際に返された候補の数（thought.pyの3-5件選出結果）
    selected_count: int

    def to_dict(self) -> dict[str, Any]:
        """エントリを辞書形式に変換する。外部ツールへの受け渡し用。"""
        return {
            "tick": self.tick,                         # ティック番号
            "timestamp": self.timestamp,               # タイムスタンプ
            "selected_label": self.selected_label,     # 選択されたポリシー名
            "candidates": self.candidates,             # 全候補の情報
            "candidate_count": self.candidate_count,   # 候補総数
            "selected_count": self.selected_count,     # 選出された候補数
        }


# ── 窓内集計キャッシュ ────────────────────────────────────────────


@dataclass
class AggregationCache:
    """窓内集計結果のキャッシュ。

    新規記録の追加時に再計算される。
    安全弁3: 数値的な事実のみを格納する。評価的判断（「偏りが大きい」等）を含めない。

    Attributes:
        label_selection_counts: ポリシーラベルごとに何回選択されたかの回数
        section_contribution_totals: 断面ごとのスコア寄与量の合計
        section_contribution_variances: 断面ごとのスコア寄与量の分散（ばらつき）
        top_gap_history: 各記録での1位と2位のスコア差の推移
        max_selection_reached_count: 選出数が上限(5件)に到達した回数
        window_size: 集計に使用した記録の件数
    """
    # ポリシーラベル別の選択回数（例: {"共感する": 5, "質問で会話を広げる": 3}）
    label_selection_counts: dict[str, int] = field(default_factory=dict)
    # 断面別の寄与量合計（例: {"drive_goal_match": 12.5, "fear_bias": -3.0}）
    section_contribution_totals: dict[str, float] = field(default_factory=dict)
    # 断面別の寄与量の分散（各断面のスコア寄与がどれだけばらついているか）
    section_contribution_variances: dict[str, float] = field(default_factory=dict)
    # 最高スコアと2位スコアの差分を時系列で記録したリスト
    top_gap_history: list[float] = field(default_factory=list)
    # 候補が選出数上限（5件）に到達した回数
    max_selection_reached_count: int = 0
    # この集計に使用した記録の件数（窓サイズ以下になることがある）
    window_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """集計結果を辞書形式に変換する。レポート出力用。"""
        return {
            # ポリシーラベル別の選択回数をそのまま辞書で返す
            "label_selection_counts": dict(self.label_selection_counts),
            # 断面別の寄与量合計（小数点以下6桁に丸める）
            "section_contribution_totals": {
                k: round(v, 6) for k, v in self.section_contribution_totals.items()
            },
            # 断面別の寄与量の分散（小数点以下6桁に丸める）
            "section_contribution_variances": {
                k: round(v, 6) for k, v in self.section_contribution_variances.items()
            },
            # 1位と2位の差分推移（小数点以下4桁に丸める）
            "top_gap_history": [round(g, 4) for g in self.top_gap_history],
            # 選出数上限到達回数
            "max_selection_reached_count": self.max_selection_reached_count,
            # 集計に使用した記録数
            "window_size": self.window_size,
        }


# ── PolicySelectionLog 本体 ───────────────────────────────────────


class PolicySelectionLog:
    """ポリシー選択ログの蓄積・集計・出力を行う本体クラス。

    セッション開始時にインスタンスを生成し、セッション終了時に破棄する。
    全内部状態はインスタンス破棄時に消失する（永続化対象外）。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない（READ-ONLY観測のみ）
    - prompt enrichment への出力経路を持たない（安全弁1: 構造的遮断）
    - 方針選択処理への逆流経路を持たない
    - 出力はログストリームへのJSON書き込みとアクセサ返却のみ
    """

    def __init__(
        self,
        config: Optional[PolicySelectionLogConfig] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            config: 設定。Noneの場合はデフォルト設定を使用する。
            enabled: 有効/無効の明示的指定。Noneの場合は環境変数で判定する。
        """
        # 設定を保持する（指定がなければデフォルト設定を使う）
        self._config = config or PolicySelectionLogConfig()
        # 有効/無効を判定する（明示指定 > 環境変数の優先順位）
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # スコアログ蓄積リスト（安全弁4: FIFO上限で古い記録を押し出す）
        self._entries: list[ScoreLogEntry] = []

        # 窓内集計キャッシュ（計算量を減らすため、結果をキャッシュする）
        self._cache: Optional[AggregationCache] = None
        # キャッシュが最新かどうかのフラグ（新しい記録が追加されると無効になる）
        self._cache_valid: bool = False

    @property
    def enabled(self) -> bool:
        """ログ記録が有効かどうかを返す。"""
        return self._enabled

    @property
    def entry_count(self) -> int:
        """現在蓄積されているログエントリの件数を返す。"""
        return len(self._entries)

    @property
    def config(self) -> PolicySelectionLogConfig:
        """設定を読み取り専用で返す。"""
        return self._config

    # ── ログの蓄積 ────────────────────────────────────────────────

    def record(
        self,
        tick: int,
        selected_label: str,
        candidates: list[dict[str, Any]],
        selected_count: int,
    ) -> None:
        """スコア内訳情報を1件蓄積する。

        方針選択処理が候補に付与したスコアの内訳情報を受け取り、
        ティック番号・タイムスタンプとともに時系列順に蓄積する。

        Args:
            tick: ティック番号（何回目の処理サイクルか）
            selected_label: 最終的に選択されたポリシーの名前
            candidates: 全候補の情報リスト。各候補は以下を含む:
                - policy_label: ポリシーラベル（名前）
                - _score: 最終スコア（浮動小数点数）
                - _score_breakdown: 断面別寄与量の辞書（オプショナル）
            selected_count: 動的選出で返された候補の数（3〜5件）
        """
        # 無効時は何もしない（安全弁5: 環境変数による完全無効化）
        if not self._enabled:
            return

        try:
            # 候補情報を構造化して、必要な情報だけを抽出する
            candidate_records: list[dict[str, Any]] = []
            # 全候補を1件ずつ処理する
            for c in candidates:
                # 各候補からラベルとスコアを取り出す
                record: dict[str, Any] = {
                    "policy_label": c.get("policy_label", ""),  # ポリシー名
                    "score": c.get("_score", 0.0),              # 最終スコア
                }
                # 断面別寄与量の内訳があれば追加する（オプショナル）
                breakdown = c.get("_score_breakdown")
                if breakdown is not None and isinstance(breakdown, dict):
                    # 内訳辞書をコピーして格納する
                    record["score_breakdown"] = dict(breakdown)
                # 構造化した候補情報をリストに追加
                candidate_records.append(record)

            # ログエントリを1件作成する
            entry = ScoreLogEntry(
                tick=tick,                              # ティック番号
                timestamp=time.time(),                  # 現在時刻をタイムスタンプとして記録
                selected_label=selected_label,          # 選択されたポリシー名
                candidates=candidate_records,           # 全候補の構造化情報
                candidate_count=len(candidates),        # 候補の総数
                selected_count=selected_count,          # 動的選出で返された候補数
            )

            # 蓄積リストにエントリを追加する
            self._entries.append(entry)

            # 安全弁4: FIFO上限。蓄積リストが上限を超えたら古い記録を押し出す
            if len(self._entries) > self._config.max_log_entries:
                # 末尾から max_log_entries 件分だけ残す（先頭の古い記録が消える）
                self._entries = self._entries[-self._config.max_log_entries:]

            # 新しい記録が追加されたので、集計キャッシュを無効化する
            self._cache_valid = False

            # ログストリームにJSON形式で個別エントリを出力する
            self._emit_entry_log(entry)

        except Exception:
            # 記録失敗時は安全に無視する（本体処理に影響を与えない）
            pass

    # ── 偏り検出のための集計 ──────────────────────────────────────

    def get_aggregation(self) -> AggregationCache:
        """窓内集計結果を返す。

        キャッシュが有効な場合はキャッシュされた結果をそのまま返す。
        新しい記録が追加されてキャッシュが無効な場合は再計算する。

        安全弁3: 数値的な事実のみを返す。評価的判断を含めない。

        Returns:
            AggregationCache: 窓内集計結果
        """
        # キャッシュが有効かつ存在する場合はそのまま返す（計算を省略）
        if self._cache_valid and self._cache is not None:
            return self._cache

        # キャッシュが無効なので再計算する
        self._cache = self._compute_aggregation()
        # 計算完了したのでキャッシュを有効にする
        self._cache_valid = True
        return self._cache

    def _compute_aggregation(self) -> AggregationCache:
        """窓内集計を実行する。

        蓄積リストの直近N件（設定値）に対して以下を算出する:
        - ポリシーラベル別の選択回数
        - 断面別の寄与量合計
        - 断面別の寄与量の分散
        - 最高スコアと2位スコアの差分の推移
        - 候補が選出数上限に到達した頻度
        """
        # 蓄積リストの直近N件を窓（ウィンドウ）として取り出す
        window = self._entries[-self._config.aggregation_window:]
        # 窓が空の場合は空の集計結果を返す
        if not window:
            return AggregationCache()

        # ── ポリシーラベル別の選択回数を数える ──
        # defaultdict(int) は存在しないキーにアクセスした時に0を返す辞書
        label_counts: dict[str, int] = defaultdict(int)
        # 窓内の各エントリについて、選択されたラベルの回数を加算する
        for entry in window:
            label_counts[entry.selected_label] += 1

        # ── 断面別の寄与量の合計と分散を計算する ──
        # 断面名をキーとして寄与量の合計を蓄積する辞書
        section_sums: dict[str, float] = defaultdict(float)
        # 断面名をキーとして寄与量の個々の値をリストで蓄積する辞書（分散計算用）
        section_values: dict[str, list[float]] = defaultdict(list)

        # 窓内の全エントリの全候補を走査する
        for entry in window:
            for cand in entry.candidates:
                # 候補のスコア内訳（断面別寄与量）を取得する
                breakdown = cand.get("score_breakdown")
                # 内訳がない場合や辞書でない場合はスキップ
                if not breakdown or not isinstance(breakdown, dict):
                    continue
                # 内訳の各断面について合計と値リストを更新する
                for section_name, contrib in breakdown.items():
                    # 数値のみを対象とする（文字列等は無視）
                    if isinstance(contrib, (int, float)):
                        section_sums[section_name] += contrib
                        section_values[section_name].append(contrib)

        # ── 分散の計算 ──
        # 分散 = 各値と平均値との差の二乗の平均
        section_variances: dict[str, float] = {}
        for section_name, values in section_values.items():
            # 値が2つ未満の場合は分散を0とする
            if len(values) < 2:
                section_variances[section_name] = 0.0
                continue
            # 平均値を計算
            n = len(values)
            mean = sum(values) / n
            # 分散を計算（各値と平均の差の二乗の平均）
            variance = sum((v - mean) ** 2 for v in values) / n
            section_variances[section_name] = variance

        # ── 最高スコアと2位スコアの差分の推移 ──
        top_gaps: list[float] = []
        for entry in window:
            # 各エントリの全候補のスコアを降順にソートする
            scores = sorted(
                [c.get("score", 0.0) for c in entry.candidates],
                reverse=True,
            )
            # 候補が2つ以上ある場合は1位と2位の差を記録
            if len(scores) >= 2:
                top_gaps.append(scores[0] - scores[1])
            # 候補が1つだけの場合はそのスコアを記録
            elif len(scores) == 1:
                top_gaps.append(scores[0])

        # ── 候補が選出数上限に到達した頻度 ──
        # thought.pyの動的選出の上限は5件（MAX_SELECT）
        # selected_count が5以上の場合を「上限到達」とカウントする
        max_reached = sum(
            1 for entry in window if entry.selected_count >= 5
        )

        # 全ての集計結果をまとめて返す
        return AggregationCache(
            label_selection_counts=dict(label_counts),          # ラベル別選択回数
            section_contribution_totals=dict(section_sums),     # 断面別寄与量合計
            section_contribution_variances=section_variances,   # 断面別寄与量分散
            top_gap_history=top_gaps,                           # 1位-2位差分推移
            max_selection_reached_count=max_reached,            # 上限到達回数
            window_size=len(window),                            # 集計対象の件数
        )

    # ── レポート出力 ──────────────────────────────────────────────

    def emit_report(self) -> dict[str, Any]:
        """蓄積と集計の結果を構造化された辞書として返す。

        出力はJSON形式であり、既存の実行時観測基盤のログ名前空間を使用する。
        安全弁3: 評価的判断を含めない。事実記述のみ。

        Returns:
            構造化されたレポート辞書
        """
        # 窓内集計結果を取得する
        agg = self.get_aggregation()

        # レポートを構造化辞書として作成する
        report: dict[str, Any] = {
            "type": "policy_selection_report",     # レコード種別の識別子
            "timestamp": time.time(),              # レポート生成時刻
            "total_entries": len(self._entries),    # 蓄積リスト内の全エントリ数
            "aggregation": agg.to_dict(),          # 窓内集計結果
        }

        # ログが有効な場合はログストリームにJSON形式で出力する
        if self._enabled:
            try:
                # 辞書をJSON文字列に変換する
                text = json.dumps(report, ensure_ascii=False, default=str)
                # ログストリームに出力する（debugレベル）
                monitor_logger.debug(text)
            except Exception:
                # 出力失敗時は安全に無視する
                pass

        # 辞書としても返す（外部ツールがプログラム的に利用するため）
        return report

    def get_entries(
        self,
        last_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """蓄積されたログエントリを辞書形式で返す。

        長期シミュレーション等の外部ツールが呼び出して分析結果を取得する
        読み取り専用のアクセサ。

        Args:
            last_n: 直近N件のみ返す。Noneの場合は全件を返す。

        Returns:
            ログエントリの辞書のリスト
        """
        # last_n が指定されている場合は直近N件のみ取り出す
        if last_n is not None and last_n > 0:
            entries = self._entries[-last_n:]
        else:
            # 指定がない場合は全件を返す
            entries = self._entries

        # 各エントリを辞書形式に変換して返す
        return [e.to_dict() for e in entries]

    # ── 内部: ログ出力 ────────────────────────────────────────────

    def _emit_entry_log(self, entry: ScoreLogEntry) -> None:
        """個別のスコアログエントリをログストリームにJSON形式で出力する。

        各ポリシー選択の都度呼ばれ、スコア内訳を含む完全な記録を出力する。
        """
        try:
            # 出力用のレコード辞書を作成する
            record = {
                "type": "policy_selection_log",         # レコード種別の識別子
                "timestamp": entry.timestamp,           # エントリのタイムスタンプ
                "tick": entry.tick,                     # ティック番号
                "selected_label": entry.selected_label, # 選択されたポリシー名
                "candidate_count": entry.candidate_count,  # 候補総数
                "selected_count": entry.selected_count,    # 選出された候補数
                "candidates": entry.candidates,            # 全候補の詳細情報
            }
            # 辞書をJSON文字列に変換する
            text = json.dumps(record, ensure_ascii=False, default=str)
            # ログストリームにdebugレベルで出力する
            monitor_logger.debug(text)
        except Exception:
            # 出力失敗時は安全に無視する（本体処理に影響を与えない）
            pass


# ── ファクトリ関数 ────────────────────────────────────────────────


def create_policy_selection_log(
    config: Optional[PolicySelectionLogConfig] = None,
    enabled: Optional[bool] = None,
) -> PolicySelectionLog:
    """PolicySelectionLog のファクトリ関数。

    orchestrator が初期化時に呼び出してインスタンスを生成する。

    Args:
        config: 設定。Noneの場合はデフォルト設定を使用する。
        enabled: 有効/無効の明示的指定。Noneの場合は環境変数で判定する。

    Returns:
        PolicySelectionLog インスタンス
    """
    # 指定された設定と有効/無効フラグでインスタンスを生成して返す
    return PolicySelectionLog(config=config, enabled=enabled)
