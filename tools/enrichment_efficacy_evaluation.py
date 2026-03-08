"""
tools/enrichment_efficacy_evaluation.py - enrichment実効性の定量評価フレームワーク

既存のEnrichmentEffectivenessEvaluator(tools/execution_monitor.py内)の
項目別語彙断片照合結果を上位集約し、項目別・セクション別・全体の
表出率を時系列で構造化蓄積するフレームワーク。

設計書: design_enrichment_efficacy_evaluation.md

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)とアクセサ返却のみ
- 情報記述群(enrichment)の構成・項目数・内容を変更しない
- 心理システムの内部状態を一切変更しない(READ-ONLY観測のみ)
- 判断・行動・選択に一切介入しない
- 帰還経路への入力経路は構造的に存在しない
- 方針選択への入力経路は構造的に存在しない
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない

安全弁:
1. 情報記述群経路の構造的遮断: enrichment出力を生成する関数を持たない
2. 永続化の対象外: save/loadフィールド追加なし、セッション境界で消失
3. 評価的判定の排除: 表出率に対する「良い/悪い」「十分/不十分」「必要/不要」
   の判定を出力しない
4. 蓄積量のFIFO上限: 全バッファにFIFO上限を設定し、無限蓄積を防止する
5. 環境変数による完全無効化: 既存のモニタリング基盤の有効/無効制御に従う
6. 照合精度の限界の常時併記: 表出率の数値を単独で出力せず、照合方式の精度限界を
   示す情報を常に同時に出力する
7. パターン抽出の禁止: 蓄積された表出率の時系列から傾向・パターン・規則性を
   抽出する処理を持たない。蓄積と数値参照のみ
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Optional

# 既存のモニタリング基盤と同一のログ名前空間
_logger = logging.getLogger("cyrene.monitor.enrichment_efficacy")


# ── 環境変数制御(安全弁5) ──────────────────────────────────────────

def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存のモニタリング基盤と同一の環境変数(CYRENE_MONITOR)に従う。
    安全弁5: 無効時は記録生成・蓄積・ログ出力をすべて省略する。
    インポート時ではなく呼び出し時に環境変数を確認する。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── FIFO上限の定数(安全弁4) ────────────────────────────────────────

# 項目別照合結果バッファのデフォルト上限
_ITEM_BUFFER_DEFAULT_MAX = 100

# セクション別集約バッファのデフォルト上限
_SECTION_BUFFER_DEFAULT_MAX = 100

# 全体集約バッファのデフォルト上限
_OVERALL_BUFFER_DEFAULT_MAX = 100


# ── EnrichmentEfficacyEvaluation ──────────────────────────────────


class EnrichmentEfficacyEvaluation:
    """enrichment実効性の定量評価フレームワーク。

    既存のEnrichmentEffectivenessEvaluator(tools/execution_monitor.py内)の
    項目別照合結果を上位集約し、3段の処理を行う:

    第一段: 照合結果の構造化蓄積
      - 項目別: 各項目の表出有無をFIFOで蓄積
      - セクション別: セクション内表出項目数の比率をFIFOで蓄積
      - 全体: 全項目中の表出項目数の比率をFIFOで蓄積

    第二段: 照合精度の限界の構造的記述
      - 語彙断片が空であった項目の数
      - 語彙断片の平均長
      - 非表出のうち語彙断片が存在した項目の数

    第三段: セッション内累積記述の提供
      - 項目別の表出率
      - セクション別の平均表出率
      - 全体の平均表出率
      - 照合不可能項目の割合
      - 語彙断片の総数と平均長

    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない
    - 計測値に基づく条件分岐を一切持たない(蓄積と参照のみ)
    - 出力はログストリームへのJSON書き込みとアクセサ返却のみ
    - 情報記述群の構成を変更しない(安全弁1)
    - 表出率に対する評価的判定を出力しない(安全弁3)
    - パターン抽出を行わない(安全弁7)
    """

    def __init__(
        self,
        item_buffer_max: int = _ITEM_BUFFER_DEFAULT_MAX,
        section_buffer_max: int = _SECTION_BUFFER_DEFAULT_MAX,
        overall_buffer_max: int = _OVERALL_BUFFER_DEFAULT_MAX,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            item_buffer_max: 項目別照合結果FIFOの上限(安全弁4)
            section_buffer_max: セクション別集約FIFOの上限(安全弁4)
            overall_buffer_max: 全体集約FIFOの上限(安全弁4)
            enabled: 明示的な有効/無効指定。Noneの場合は環境変数で判定。
        """
        # 安全弁4: FIFO上限(最低1)
        self._item_buffer_max = max(1, item_buffer_max)
        self._section_buffer_max = max(1, section_buffer_max)
        self._overall_buffer_max = max(1, overall_buffer_max)

        # 有効/無効判定(明示指定 > 環境変数)
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # ── 第一段: 照合結果の構造化蓄積 ──

        # 1. 項目別照合結果バッファ: 項目識別子→deque[bool]
        self._item_buffers: dict[str, deque] = {}

        # 2. セクション別集約バッファ: セクション識別子→deque[float]
        self._section_buffers: dict[str, deque] = {}

        # 3. 全体集約バッファ: deque[float]
        self._overall_buffer: deque = deque(maxlen=self._overall_buffer_max)

        # ── 第二段: 語彙断片統計(照合精度の限界記述用) ──

        # 項目別の語彙断片数キャッシュ: 項目識別子→断片数
        self._item_fragment_counts: dict[str, int] = {}

        # 項目別の語彙断片長合計キャッシュ: 項目識別子→文字数合計
        self._item_fragment_total_lengths: dict[str, int] = {}

        # 項目→セクション対応
        self._item_section_map: dict[str, str] = {}

        # ── 照合回数カウンタ ──
        self._evaluation_count: int = 0

    @property
    def enabled(self) -> bool:
        """モニタリングが有効かどうか。"""
        return self._enabled

    @property
    def evaluation_count(self) -> int:
        """照合実行回数。"""
        return self._evaluation_count

    # ── 語彙断片統計の更新 ────────────────────────────────────────────

    def update_vocab_statistics(
        self,
        vocab_cache: dict[str, list[str]],
        item_section_map: dict[str, str],
    ) -> None:
        """語彙断片統計を更新する。

        EnrichmentEffectivenessEvaluatorの語彙断片キャッシュから
        断片数・断片長の統計を抽出する。この情報は第二段(照合精度の限界記述)
        に使用される。

        Args:
            vocab_cache: 項目識別子→語彙断片リスト
            item_section_map: 項目識別子→セクション識別子
        """
        if not self._enabled:
            return
        try:
            for item_id, fragments in vocab_cache.items():
                self._item_fragment_counts[item_id] = len(fragments)
                total_len = sum(len(f) for f in fragments)
                self._item_fragment_total_lengths[item_id] = total_len

            # 項目→セクション対応を更新
            self._item_section_map.update(item_section_map)
        except Exception:
            # 安全弁: 更新失敗時の安全な無視
            pass

    # ── 第一段: 照合結果の構造化蓄積 ──────────────────────────────────

    def record_manifestation_results(
        self,
        results: dict[str, bool],
        item_section_map: dict[str, str],
    ) -> None:
        """照合結果を構造化蓄積する。

        EnrichmentEffectivenessEvaluator.evaluate_manifestation()の戻り値を
        受け取り、項目別・セクション別・全体の3層で蓄積する。

        蓄積は事実の記録であり、蓄積された数値に対する統計的判定
        (有意差検定、閾値比較、正常範囲判定等)は行わない。

        Args:
            results: 項目識別子→表出有無の辞書
            item_section_map: 項目識別子→セクション識別子の対応
        """
        if not self._enabled:
            return
        try:
            self._evaluation_count += 1

            # 項目→セクション対応を更新
            self._item_section_map.update(item_section_map)

            # ── 項目別蓄積 ──
            for item_id, is_manifest in results.items():
                if item_id not in self._item_buffers:
                    self._item_buffers[item_id] = deque(
                        maxlen=self._item_buffer_max
                    )
                self._item_buffers[item_id].append(is_manifest)

            # ── セクション別集約 ──
            # セクションごとに項目をグループ化し、表出率を算出
            section_items: dict[str, list[bool]] = {}
            for item_id, is_manifest in results.items():
                section = item_section_map.get(
                    item_id, self._item_section_map.get(item_id, "")
                )
                if section not in section_items:
                    section_items[section] = []
                section_items[section].append(is_manifest)

            for section, manifest_list in section_items.items():
                if section not in self._section_buffers:
                    self._section_buffers[section] = deque(
                        maxlen=self._section_buffer_max
                    )
                total = len(manifest_list)
                manifest_count = sum(1 for v in manifest_list if v)
                rate = manifest_count / total if total > 0 else 0.0
                self._section_buffers[section].append(rate)

            # ── 全体集約 ──
            total_items = len(results)
            total_manifest = sum(1 for v in results.values() if v)
            overall_rate = total_manifest / total_items if total_items > 0 else 0.0
            self._overall_buffer.append(overall_rate)

            # ── ログ出力 ──
            self._emit_record_log(results)

        except Exception:
            # 安全弁: 蓄積失敗時の安全な無視
            pass

    # ── 第二段: 照合精度の限界の構造的記述 ─────────────────────────────

    def get_precision_limitations(self) -> dict[str, Any]:
        """照合精度の限界を構造的に記述する。

        照合方式が部分文字列一致に基づくものであるため、
        以下の構造的限界を常に併記する(安全弁6):

        - empty_fragment_count: 語彙断片が空であった項目の数
        - avg_fragment_length: 語彙断片の平均長
        - non_manifest_with_fragments_count: 非表出のうち語彙断片が存在した項目の数

        Returns:
            精度限界を記述する辞書。
        """
        try:
            # 語彙断片が空(0個)であった項目の数
            empty_fragment_count = sum(
                1 for count in self._item_fragment_counts.values()
                if count == 0
            )

            # 語彙断片の平均長
            total_fragments = sum(self._item_fragment_counts.values())
            total_fragment_length = sum(
                self._item_fragment_total_lengths.values()
            )
            avg_fragment_length = (
                round(total_fragment_length / total_fragments, 2)
                if total_fragments > 0
                else 0.0
            )

            # 非表出のうち語彙断片が存在した項目の数
            # (直近の照合結果で非表出かつ語彙断片が1つ以上あった項目)
            non_manifest_with_fragments = 0
            for item_id, buf in self._item_buffers.items():
                if buf and not buf[-1]:  # 直近が非表出
                    frag_count = self._item_fragment_counts.get(item_id, 0)
                    if frag_count > 0:
                        non_manifest_with_fragments += 1

            return {
                "empty_fragment_count": empty_fragment_count,
                "total_fragment_count": total_fragments,
                "avg_fragment_length": avg_fragment_length,
                "non_manifest_with_fragments_count": non_manifest_with_fragments,
                "total_items_tracked": len(self._item_fragment_counts),
            }
        except Exception:
            return {
                "empty_fragment_count": 0,
                "total_fragment_count": 0,
                "avg_fragment_length": 0.0,
                "non_manifest_with_fragments_count": 0,
                "total_items_tracked": 0,
            }

    # ── 第三段: セッション内累積記述の提供 ─────────────────────────────

    def get_cumulative_description(self) -> dict[str, Any]:
        """セッション内の累積記述を生成する。

        この記述は事実の数値列挙であり、「高い」「低い」「十分」「不十分」等の
        評価的ラベルを一切含まない(安全弁3)。

        蓄積された表出率の時系列から傾向・パターン・規則性を抽出する処理を
        持たない(安全弁7)。蓄積と数値参照のみ。

        Returns:
            累積記述の辞書。照合精度の限界を常時併記する(安全弁6)。
        """
        try:
            # ── 項目別の表出率 ──
            item_rates: dict[str, float] = {}
            for item_id, buf in self._item_buffers.items():
                if len(buf) > 0:
                    item_rates[item_id] = round(
                        sum(1 for v in buf if v) / len(buf), 4
                    )
                else:
                    item_rates[item_id] = 0.0

            # ── セクション別の平均表出率 ──
            section_avg_rates: dict[str, float] = {}
            for section, buf in self._section_buffers.items():
                if len(buf) > 0:
                    section_avg_rates[section] = round(
                        sum(buf) / len(buf), 4
                    )
                else:
                    section_avg_rates[section] = 0.0

            # ── 全体の平均表出率 ──
            overall_avg_rate = 0.0
            if len(self._overall_buffer) > 0:
                overall_avg_rate = round(
                    sum(self._overall_buffer) / len(self._overall_buffer), 4
                )

            # ── 照合不可能項目の割合 ──
            total_tracked = len(self._item_fragment_counts)
            empty_fragments = sum(
                1 for count in self._item_fragment_counts.values()
                if count == 0
            )
            unmatchable_rate = (
                round(empty_fragments / total_tracked, 4)
                if total_tracked > 0
                else 0.0
            )

            # ── 語彙断片の総数と平均長 ──
            total_fragments = sum(self._item_fragment_counts.values())
            total_fragment_length = sum(
                self._item_fragment_total_lengths.values()
            )
            avg_fragment_length = (
                round(total_fragment_length / total_fragments, 2)
                if total_fragments > 0
                else 0.0
            )

            # ── 照合精度の限界の常時併記(安全弁6) ──
            precision_limitations = self.get_precision_limitations()

            return {
                "evaluation_count": self._evaluation_count,
                "item_rates": item_rates,
                "item_count": len(item_rates),
                "section_avg_rates": section_avg_rates,
                "section_count": len(section_avg_rates),
                "overall_avg_rate": overall_avg_rate,
                "overall_buffer_length": len(self._overall_buffer),
                "unmatchable_rate": unmatchable_rate,
                "total_fragments": total_fragments,
                "avg_fragment_length": avg_fragment_length,
                # 安全弁6: 照合精度の限界を常時併記
                "precision_limitations": precision_limitations,
            }
        except Exception:
            return {
                "evaluation_count": self._evaluation_count,
                "item_rates": {},
                "item_count": 0,
                "section_avg_rates": {},
                "section_count": 0,
                "overall_avg_rate": 0.0,
                "overall_buffer_length": 0,
                "unmatchable_rate": 0.0,
                "total_fragments": 0,
                "avg_fragment_length": 0.0,
                "precision_limitations": self.get_precision_limitations(),
            }

    # ── 項目別の表出率取得 ────────────────────────────────────────────

    def get_item_rate(self, item_id: str) -> float:
        """特定項目の表出率を返す。

        FIFO内の表出回数/照合回数を返す。
        項目が存在しない場合は0.0を返す。

        Args:
            item_id: 項目識別子

        Returns:
            0.0~1.0の表出率
        """
        try:
            buf = self._item_buffers.get(item_id)
            if not buf or len(buf) == 0:
                return 0.0
            return sum(1 for v in buf if v) / len(buf)
        except Exception:
            return 0.0

    # ── セクション別の平均表出率取得 ───────────────────────────────────

    def get_section_avg_rate(self, section: str) -> float:
        """特定セクションの平均表出率を返す。

        FIFO内のセクション表出率の平均を返す。
        セクションが存在しない場合は0.0を返す。

        Args:
            section: セクション識別子

        Returns:
            0.0~1.0の平均表出率
        """
        try:
            buf = self._section_buffers.get(section)
            if not buf or len(buf) == 0:
                return 0.0
            return sum(buf) / len(buf)
        except Exception:
            return 0.0

    # ── 全体の平均表出率取得 ──────────────────────────────────────────

    def get_overall_avg_rate(self) -> float:
        """全体の平均表出率を返す。

        FIFO内の全体表出率の平均を返す。

        Returns:
            0.0~1.0の平均表出率
        """
        try:
            if len(self._overall_buffer) == 0:
                return 0.0
            return sum(self._overall_buffer) / len(self._overall_buffer)
        except Exception:
            return 0.0

    # ── セッションサマリの出力 ────────────────────────────────────────

    def emit_session_summary(
        self, monitor: Any = None
    ) -> Optional[dict[str, Any]]:
        """セッション終了時にサマリをJSON形式でログストリームに出力する。

        第三段の累積記述と第二段の精度限界記述を統合して出力する。

        Args:
            monitor: 親のExecutionMonitorインスタンス(ログ出力用)

        Returns:
            セッションサマリの辞書。無効時はNone。
        """
        if not self._enabled:
            return None
        try:
            cumulative = self.get_cumulative_description()

            record = {
                "type": "enrichment_efficacy_session_summary",
                "timestamp": time.time(),
                **cumulative,
            }

            if monitor and hasattr(monitor, '_emit_json'):
                monitor._emit_json(record)
            else:
                text = json.dumps(record, ensure_ascii=False, default=str)
                _logger.debug(text)

            return record
        except Exception:
            # 安全弁: ログ出力失敗時の安全な無視
            return None

    # ── 読み取り専用アクセサ ──────────────────────────────────────────

    def get_item_buffers(self) -> dict[str, list[bool]]:
        """項目別照合結果バッファの読み取り専用コピーを返す。"""
        return {k: list(v) for k, v in self._item_buffers.items()}

    def get_section_buffers(self) -> dict[str, list[float]]:
        """セクション別集約バッファの読み取り専用コピーを返す。"""
        return {k: list(v) for k, v in self._section_buffers.items()}

    def get_overall_buffer(self) -> list[float]:
        """全体集約バッファの読み取り専用コピーを返す。"""
        return list(self._overall_buffer)

    # ── 内部: ログ出力 ────────────────────────────────────────────────

    def _emit_record_log(self, results: dict[str, bool]) -> None:
        """照合結果の記録をログストリームに出力する。

        記録は事実の数値のみで、評価的ラベルを含まない(安全弁3)。
        照合精度の限界を常時併記する(安全弁6)。
        """
        if not self._enabled:
            return
        try:
            manifest_count = sum(1 for v in results.values() if v)
            total = len(results)
            rate = manifest_count / total if total > 0 else 0.0

            record = {
                "type": "enrichment_efficacy_record",
                "timestamp": time.time(),
                "evaluation_count": self._evaluation_count,
                "items_evaluated": total,
                "items_manifest": manifest_count,
                "items_non_manifest": total - manifest_count,
                "overall_rate": round(rate, 4),
                # 安全弁6: 照合精度の限界を常時併記
                "precision_limitations": self.get_precision_limitations(),
            }

            text = json.dumps(record, ensure_ascii=False, default=str)
            _logger.debug(text)
        except Exception:
            # 安全弁: ログ出力失敗時の安全な無視
            pass
