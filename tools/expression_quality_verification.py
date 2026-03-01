"""
tools/expression_quality_verification.py - 代弁品質の事実記録

確定方針と代弁出力の対（ペア）を事実として記録する仕組み。
方針の内容と実際の発話テキストの対応関係を構造的に記録し、
事後的に参照可能にする。

設計書: design_expression_quality_verification.md

本機能の構造的分離:
- 出力先はPython標準ログストリーム(JSON形式)とアクセサ返却のみ
- 内部システムの状態変数を一切変更しない(READ-ONLY観測のみ)
- 判断・行動・選択に一切介入しない
- 計測値に基づく条件分岐を一切持たない
- 全内部状態はセッション境界で消失する(永続化対象外)
- save/loadの対象フィールドに一切追加しない

方針選択系との分離:
- 記録内容は方針候補の生成にもスコアリングにも入力されない

enrichment系との分離:
- 記録内容はenrichmentの構成・項目・テキストに一切反映されない

状態更新系との分離:
- 記録内容は感情・ムード・ドライブ・恐怖指数の更新に一切寄与しない

記憶系との分離:
- 記録内容はエピソード記憶・長期記憶・感情記憶に書き込まれない

帰還経路との分離:
- 記録内容は帰還経路のいずれの入力にもならない

自己知覚系との分離:
- 記録内容は自己行動知覚・意図-行動乖離認知の入力にならない

安全弁:
1. enrichment経路の構造的遮断: enrichment出力を生成する関数を持たない
2. 永続化の対象外: save/loadフィールド追加なし
3. 事実記述限定: 評価的判断・一致度スコア・品質指標を含まない
4. 蓄積量のFIFO上限: 設定によりバッファ上限を制御
5. 環境変数による完全無効化: CYRENE_MONITOR=1 で有効化
6. 感情語彙辞書の静的固定: 辞書は構成時に確定し実行時に動的変更しない
7. パターン抽出禁止: 蓄積された記録から傾向・パターン・規則性を抽出しない
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

# 独自ログ名前空間(既存モニタリング基盤と同一)
_logger = logging.getLogger("cyrene.monitor.expression_quality")


# ── 環境変数制御 ──────────────────────────────────────────────────

def _is_monitor_enabled() -> bool:
    """モニタリングが有効かどうかを実行時に判定する。

    既存の実行時観測基盤と同じ環境変数(CYRENE_MONITOR)に従う。
    安全弁5: 無効時は記録生成・蓄積・ログ出力をすべて省略する。
    インポート時ではなく呼び出し時に環境変数を確認する。
    """
    return os.environ.get("CYRENE_MONITOR", "0") == "1"


# ── 感情語彙辞書（安全弁6: 静的固定）──────────────────────────────

# 本機能専用の感情語彙辞書。既存の入力知覚処理(perception.py)の辞書とは
# 同系統だがインスタンスを共有しない独立した辞書である。
# 構成時に確定し、実行時の動的変更（追加・削除・重み付け変更）を行わない。
_EXPRESSION_EMOTION_VOCAB: dict[str, str] = {
    # --- happy ---
    "嬉しい": "happy", "楽しい": "happy", "幸せ": "happy",
    "ありがとう": "happy", "笑": "happy",
    "やったー": "happy", "よかった": "happy",
    "うれしい": "happy", "たのしい": "happy",
    "わーい": "happy", "最高": "happy",
    "ハッピー": "happy", "ウキウキ": "happy",
    "ワクワク": "happy", "るんるん": "happy",
    # --- sad ---
    "悲しい": "sad", "辛い": "sad", "寂しい": "sad",
    "かなしい": "sad", "つらい": "sad", "さみしい": "sad",
    "泣": "sad", "切ない": "sad",
    "しんどい": "sad", "落ち込": "sad",
    "ショック": "sad", "がっかり": "sad",
    # --- angry ---
    "怒": "angry", "ムカ": "angry", "イライラ": "angry",
    "腹立": "angry", "ふざけ": "angry",
    "キレ": "angry", "うざ": "angry",
    "許せ": "angry", "ひどい": "angry",
    # --- surprised ---
    "驚": "surprised", "びっくり": "surprised",
    "まさか": "surprised", "えっ": "surprised",
    "うそ": "surprised", "マジ": "surprised",
    "信じられ": "surprised",
    # --- scared ---
    "怖い": "scared", "不安": "scared",
    "こわい": "scared", "恐ろし": "scared",
    "おそろし": "scared", "ビクビク": "scared",
    "ドキドキ": "scared", "心配": "scared",
    "ゾッと": "scared", "やばい": "scared",
    # --- loving ---
    "好き": "loving", "愛してる": "loving", "大好き": "loving",
    "だいすき": "loving", "すき": "loving",
    "たまらない": "loving", "かわいい": "loving",
    "いとおしい": "loving", "キュン": "loving",
    # --- teasing ---
    "からかう": "teasing", "いじわる": "teasing",
    "冗談": "teasing", "ウソウソ": "teasing",
    "なんちゃって": "teasing", "ニヤニヤ": "teasing",
    # --- confused ---
    "困った": "confused", "わからない": "confused",
    "どうしよう": "confused", "迷う": "confused",
    "混乱": "confused", "モヤモヤ": "confused",
    # --- disappointed ---
    "残念": "disappointed", "期待はずれ": "disappointed",
    "つまらない": "disappointed",
    # --- relieved ---
    "安心": "relieved", "ほっと": "relieved",
    "助かった": "relieved",
    # --- nostalgic ---
    "懐かしい": "nostalgic", "なつかしい": "nostalgic",
    # --- embarrassed ---
    "恥ずかし": "embarrassed", "はずかし": "embarrassed",
    "照れ": "embarrassed",
    # --- frustrated ---
    "悔しい": "frustrated", "くやしい": "frustrated",
    "もどかしい": "frustrated",
    # --- anxious ---
    "焦る": "anxious", "あせる": "anxious",
    "そわそわ": "anxious",
    # --- grateful ---
    "感謝": "grateful", "ありがたい": "grateful",
    "おかげ": "grateful",
    # --- bored ---
    "退屈": "bored",
}


# ── テキスト特徴量の抽出 ──────────────────────────────────────────

def _extract_emotion_labels(text: str) -> list[str]:
    """発話テキストから感情語彙ラベルを列挙する。

    テキスト中に出現する感情関連語彙のラベルを列挙する。
    これは「発話テキストにどの感情語彙が含まれるか」の事実記述であり、
    テキストの感情を推定・判定するものではない。

    同一ラベルの重複は除去し、出現順序で返す。
    """
    seen: set[str] = set()
    labels: list[str] = []
    for keyword, label in _EXPRESSION_EMOTION_VOCAB.items():
        if keyword in text and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _has_question_mark(text: str) -> bool:
    """発話テキストに疑問符が含まれるかどうか。"""
    return "?" in text or "？" in text


def _count_sentences(text: str) -> int:
    """発話テキストの文数を数える。

    句点(。)・感嘆符(！!)・疑問符(？?)による分割数。
    空文字列の場合は0を返す。
    """
    if not text or not text.strip():
        return 0
    count = 0
    for ch in text:
        if ch in ("。", "！", "!", "？", "?"):
            count += 1
    # 区切り文字がなくてもテキストがあれば1文として数える
    return max(count, 1)


# ── 設定 ──────────────────────────────────────────────────────────


@dataclass
class ExpressionQualityConfig:
    """代弁品質記録の設定パラメータ。

    Attributes:
        max_buffer_size: 記録バッファのFIFO上限（安全弁4）。
            この件数を超えると最古の記録が自然消失する。
        recent_count: 参照提供時に返す直近N件のデフォルト値。
    """
    max_buffer_size: int = 200
    recent_count: int = 20

    def __post_init__(self) -> None:
        """設定値のバリデーション。不正な値はデフォルトに戻す。"""
        if self.max_buffer_size < 1:
            self.max_buffer_size = 200
        if self.recent_count < 1:
            self.recent_count = 20
        if self.recent_count > self.max_buffer_size:
            self.recent_count = self.max_buffer_size


# ── 記録構造体 ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpressionRecord:
    """1件の代弁コールに対する記録構造体（不変）。

    生成後に変更されない。
    """
    # ティック番号
    tick_number: int
    # 入力経路ラベル ("vision" / "text" / "internal")
    input_pathway: str
    # 記録時刻
    timestamp: float
    # 方針ラベル
    policy_label: str
    # 方針根拠
    policy_rationale: str
    # 方針側支配的感情ラベル
    policy_emotion_label: str
    # 方針側支配的感情強度
    policy_emotion_intensity: float
    # 方針側ムードvalence
    policy_mood_valence: float
    # 方針側ムードarousal
    policy_mood_arousal: float
    # enrichment文字数
    enrichment_char_count: int
    # 発話テキスト全文
    utterance_text: str
    # 発話テキスト文字数
    utterance_char_count: int
    # 発話テキスト文数
    utterance_sentence_count: int
    # 発話メタ感情ラベル
    utterance_meta_emotion: str
    # 発話メタ強度
    utterance_meta_intensity: float
    # 発話メタ行動ラベル
    utterance_meta_action: str
    # フォールバックフラグ
    is_fallback: bool
    # 発話テキスト内の感情語彙ラベルリスト
    utterance_emotion_labels: tuple[str, ...]
    # 発話テキスト内の疑問符有無
    has_question_mark: bool

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換する。"""
        return {
            "tick_number": self.tick_number,
            "input_pathway": self.input_pathway,
            "timestamp": self.timestamp,
            "policy_label": self.policy_label,
            "policy_rationale": self.policy_rationale,
            "policy_emotion_label": self.policy_emotion_label,
            "policy_emotion_intensity": self.policy_emotion_intensity,
            "policy_mood_valence": self.policy_mood_valence,
            "policy_mood_arousal": self.policy_mood_arousal,
            "enrichment_char_count": self.enrichment_char_count,
            "utterance_text": self.utterance_text,
            "utterance_char_count": self.utterance_char_count,
            "utterance_sentence_count": self.utterance_sentence_count,
            "utterance_meta_emotion": self.utterance_meta_emotion,
            "utterance_meta_intensity": self.utterance_meta_intensity,
            "utterance_meta_action": self.utterance_meta_action,
            "is_fallback": self.is_fallback,
            "utterance_emotion_labels": list(self.utterance_emotion_labels),
            "has_question_mark": self.has_question_mark,
        }


# ── ExpressionQualityVerification ────────────────────────────────


class ExpressionQualityVerification:
    """代弁品質の事実記録。

    代弁コールの前後で利用可能な確定方針と発話出力の対（ペア）を
    事実として記録し、蓄積する。

    全内部状態はインスタンス破棄時に消失する(永続化対象外)。

    本クラスの全メソッドは:
    - 内部システムの状態変数を一切変更しない
    - 記録に基づく条件分岐を一切持たない(蓄積と参照のみ)
    - 出力はログストリームへのJSON書き込みとアクセサ返却のみ
    - 品質の評価・採点・判定を行わない
    - パターン抽出・傾向分析を行わない
    """

    def __init__(
        self,
        config: Optional[ExpressionQualityConfig] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """初期化。

        Args:
            config: 設定パラメータ。Noneの場合はデフォルト設定を使用。
            enabled: 明示的な有効/無効指定。Noneの場合は環境変数で判定。
        """
        self._config = config or ExpressionQualityConfig()

        # 有効/無効判定(明示指定 > 環境変数)
        # 安全弁5: 環境変数による完全無効化
        self._enabled = enabled if enabled is not None else _is_monitor_enabled()

        # ── 記録バッファ（安全弁4: FIFO上限）──
        self._buffer: deque[ExpressionRecord] = deque(
            maxlen=self._config.max_buffer_size
        )

        # ── セッション累積 ──
        self._record_count: int = 0
        self._fallback_count: int = 0

    @property
    def enabled(self) -> bool:
        """モニタリングが有効かどうか。"""
        return self._enabled

    @property
    def record_count(self) -> int:
        """記録件数カウンタ。"""
        return self._record_count

    @property
    def fallback_count(self) -> int:
        """フォールバック発生カウンタ。"""
        return self._fallback_count

    @property
    def buffer_size(self) -> int:
        """現在のバッファ内の記録件数。"""
        return len(self._buffer)

    # ── 第1段: 対構成 ─────────────────────────────────────────────

    def record_expression(
        self,
        tick_number: int,
        input_pathway: str,
        policy_label: str,
        policy_rationale: str,
        policy_emotion_label: str,
        policy_emotion_intensity: float,
        policy_mood_valence: float,
        policy_mood_arousal: float,
        enrichment_char_count: int,
        utterance_text: str,
        utterance_meta: dict[str, Any],
        is_fallback: bool,
    ) -> Optional[ExpressionRecord]:
        """代弁コールの前後データから1件の記録を構成し蓄積する。

        代弁コール完了後に呼び出される。方針側情報と発話側情報の
        両方を引数として受け取る。本機能側が内部処理の状態変数に
        直接アクセスする経路を持たない。

        Args:
            tick_number: ティック番号
            input_pathway: 入力経路ラベル ("vision" / "text" / "internal")
            policy_label: 選択された方針ラベル
            policy_rationale: 方針の根拠文字列
            policy_emotion_label: 代弁コール時点の支配的感情ラベル
            policy_emotion_intensity: 代弁コール時点の支配的感情強度
            policy_mood_valence: 代弁コール時点のムードvalence
            policy_mood_arousal: 代弁コール時点のムードarousal
            enrichment_char_count: enrichment圧縮後の文字数
            utterance_text: 代弁出力の発話テキスト全文
            utterance_meta: 代弁出力のメタ情報辞書
                (emotion, intensity, action)
            is_fallback: フォールバック発話であったかどうか

        Returns:
            生成された記録。無効時または失敗時はNone。
        """
        if not self._enabled:
            return None
        try:
            # 入力経路ラベルの正規化
            if input_pathway not in ("vision", "text", "internal"):
                input_pathway = "unknown"

            # 発話テキストからの記述的特徴量
            emotion_labels = _extract_emotion_labels(utterance_text)
            question_mark = _has_question_mark(utterance_text)
            sentence_count = _count_sentences(utterance_text)

            # メタ情報の取得
            meta_emotion = ""
            meta_intensity = 0.0
            meta_action = ""
            if isinstance(utterance_meta, dict):
                meta_emotion = str(utterance_meta.get("emotion", ""))
                raw_intensity = utterance_meta.get("intensity", 0.0)
                meta_intensity = float(raw_intensity) if isinstance(raw_intensity, (int, float)) else 0.0
                meta_action = str(utterance_meta.get("action", ""))

            # 記録構造体の生成
            record = ExpressionRecord(
                tick_number=tick_number,
                input_pathway=input_pathway,
                timestamp=time.time(),
                policy_label=str(policy_label),
                policy_rationale=str(policy_rationale),
                policy_emotion_label=str(policy_emotion_label),
                policy_emotion_intensity=float(policy_emotion_intensity),
                policy_mood_valence=float(policy_mood_valence),
                policy_mood_arousal=float(policy_mood_arousal),
                enrichment_char_count=int(enrichment_char_count),
                utterance_text=str(utterance_text),
                utterance_char_count=len(str(utterance_text)),
                utterance_sentence_count=sentence_count,
                utterance_meta_emotion=meta_emotion,
                utterance_meta_intensity=meta_intensity,
                utterance_meta_action=meta_action,
                is_fallback=bool(is_fallback),
                utterance_emotion_labels=tuple(emotion_labels),
                has_question_mark=question_mark,
            )

            # 第2段: 蓄積
            self._buffer.append(record)

            # セッション累積の更新
            self._record_count += 1
            if is_fallback:
                self._fallback_count += 1

            # ログ出力
            self._emit_json({
                "type": "expression_quality_record",
                **record.to_dict(),
            })

            return record

        except Exception:
            # 安全弁: 記録失敗時の安全な無視
            return None

    # ── 第3段: 参照提供 ───────────────────────────────────────────

    def get_recent_records(
        self, count: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """蓄積された記録の直近N件を辞書形式で返す。

        集計・統計・傾向分析は行わない。記録の生データをそのまま返す
        （安全弁7: パターン抽出禁止）。

        Args:
            count: 返す件数。Noneの場合は設定のデフォルト値を使用。

        Returns:
            直近N件の記録リスト（新しい順）。
        """
        if not self._enabled:
            return []
        try:
            n = count if count is not None else self._config.recent_count
            n = max(1, min(n, len(self._buffer)))
            # 新しい順に返す
            records = list(self._buffer)
            recent = records[-n:]
            recent.reverse()
            return [r.to_dict() for r in recent]
        except Exception:
            return []

    # ── セッションサマリー出力 ────────────────────────────────────

    def emit_session_summary(self) -> Optional[dict[str, Any]]:
        """セッション終了時のセッション累積情報を出力する。

        Returns:
            セッションサマリーの辞書。
        """
        if not self._enabled:
            return None
        try:
            summary: dict[str, Any] = {
                "type": "expression_quality_session_summary",
                "timestamp": time.time(),
                "record_count": self._record_count,
                "fallback_count": self._fallback_count,
                "buffer_size": len(self._buffer),
            }
            self._emit_json(summary)
            return summary
        except Exception:
            return None

    # ── 読み取り専用アクセサ ──────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """現在の累積情報を読み取り専用で返す。

        外部の分析ツール(シミュレータ等)が呼び出す読み取り専用アクセサ。

        Returns:
            累積情報の辞書。
        """
        return {
            "record_count": self._record_count,
            "fallback_count": self._fallback_count,
            "buffer_size": len(self._buffer),
        }

    # ── 内部: JSON構造化ログ出力 ──────────────────────────────────

    def _emit_json(self, record: dict[str, Any]) -> None:
        """JSON構造化ログをログストリームに出力する。"""
        try:
            text = json.dumps(record, ensure_ascii=False, default=str)
            _logger.debug(text)
        except Exception:
            # 安全弁: ログ出力失敗時の安全な無視
            pass
