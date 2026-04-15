"""
psyche/multi_path_recall.py - 記憶の多経路想起（Multi-Path Recall）

記憶想起の入口を拡充し、感情連想・文脈連想・時間近接の3つの想起経路を
内部処理が参照可能な記憶候補の供給源として提供する。

設計原則 (design_multi_path_recall.md 準拠):
- 記憶の内容を変更・加工・評価しない
- 想起結果に基づいて判断・行動を確定しない
- 特定の記憶を恒常的に優先しない
- 特定の想起経路を他の経路より恒常的に優先しない
- 想起された記憶の意味・価値・正誤を判定しない
- 忘却処理の進行に直接影響を与えない（参照頻度非通知）
- 既存の外部API向け想起処理を変更・代替しない
- 3経路の候補は経路ラベル付きで等価に並列
- 出力はenrichmentの参照情報としてのみ提供

3経路:
1. 感情連想経路: 現在の感情状態と記憶の感情痕跡の近接度による候補列挙
2. 文脈連想経路: 知覚内容/トピックと記憶のトピック属性の重複による候補列挙
3. 時間近接経路: 現在時刻と記憶のタイムスタンプの時間的距離による候補列挙

安全弁:
1. 経路間等価性（経路ごとの候補上限を同一に設定）
2. 顕著性バイアス抑制（感情痕跡が弱い記憶の一定割合混入）
3. ルーミネーション防止（直近想起履歴によるスライディングウィンドウ抑制）
4. 忘却処理との分離（参照頻度非通知、INVISIBLE記憶除外）
5. 外部API想起との整合（enrichment参照のみ、上書きなし）

Usage::

    recall = create_multi_path_recall()
    result = recall.recall_all_paths(
        unified_units=units,
        binding_store=binding_store,
        forgetting_state=forgetting_state,
        emotion_snapshot=emotion_snapshot,
        context_snapshot=context_snapshot,
        temporal_snapshot=temporal_snapshot,
    )
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from . import coefficient_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RecallPathLabel(Enum):
    """想起経路ラベル。3経路を等価に識別する。"""
    EMOTIONAL = "emotional"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"


# =============================================================================
# Helpers
# =============================================================================

def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# テキスト長の安全弁上限
TEXT_MAX_LENGTH = 10000

# 語彙サイズの安全弁上限
DEFAULT_MAX_VOCAB_SIZE = 5000

# N-gram のデフォルトサイズ（bi-gram）
DEFAULT_NGRAM_N = 2


def _is_cjk(char: str) -> bool:
    """文字がCJK統合漢字・ひらがな・カタカナかどうか判定する。"""
    cp = ord(char)
    # CJK Unified Ideographs
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    # Hiragana
    if 0x3040 <= cp <= 0x309F:
        return True
    # Katakana
    if 0x30A0 <= cp <= 0x30FF:
        return True
    # CJK Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK Extension B
    if 0x20000 <= cp <= 0x2A6DF:
        return True
    # CJK Compatibility Ideographs
    if 0xF900 <= cp <= 0xFAFF:
        return True
    # Halfwidth/Fullwidth Katakana
    if 0xFF65 <= cp <= 0xFF9F:
        return True
    return False


def _tokenize(text: str, ngram_n: int = DEFAULT_NGRAM_N) -> list[str]:
    """テキストをトークンに分割する。

    Unicode文字種境界で分割し、CJK文字列にはN-gram（デフォルトbi-gram）を適用する。
    外部辞書・形態素解析器には一切依存しない。

    Args:
        text: 入力テキスト
        ngram_n: CJK文字列に適用するN-gramサイズ

    Returns:
        トークンのリスト（小文字正規化済み）
    """
    if not text:
        return []

    # 制御文字の除去（改行・タブ以外のU+0000-U+001Fは空白に置換）
    cleaned = []
    for ch in text:
        if ch in ('\n', '\t'):
            cleaned.append(' ')
        elif ord(ch) < 0x20:
            # 制御文字を空白に置換（隣接トークンの連結を防止）
            cleaned.append(' ')
        else:
            cleaned.append(ch)
    text = ''.join(cleaned)

    # テキスト長上限の適用
    if len(text) > TEXT_MAX_LENGTH:
        text = text[:TEXT_MAX_LENGTH]

    # 小文字正規化
    text = text.lower()

    tokens: list[str] = []

    # 文字種ごとにセグメントに分割
    # CJK文字のランとASCII/ラテン文字のランを区別
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if _is_cjk(ch):
            # CJKランを収集
            cjk_run = []
            while i < n and _is_cjk(text[i]):
                cjk_run.append(text[i])
                i += 1
            # N-gram分割
            cjk_str = ''.join(cjk_run)
            if len(cjk_str) < ngram_n:
                # N-gram不能→単一トークンとして扱う
                tokens.append(cjk_str)
            else:
                for j in range(len(cjk_str) - ngram_n + 1):
                    tokens.append(cjk_str[j:j + ngram_n])

        elif ch.isalnum():
            # ASCII/ラテン文字のランを収集（単語）
            # CJK文字はisalnum()=Trueだが別セグメントなので除外
            word_start = i
            while i < n and text[i].isalnum() and not _is_cjk(text[i]):
                i += 1
            tokens.append(text[word_start:i])

        else:
            # 区切り文字（空白、句読点、記号）→スキップ
            i += 1

    return tokens


def _build_tfidf_index(
    units: list[Any],
    ngram_n: int = DEFAULT_NGRAM_N,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
) -> "TfIdfIndex":
    """記憶群からTF-IDFインデックスを構築する純粋関数。

    入力のUnifiedMemoryUnitを読み取るのみで変更しない。

    Args:
        units: UnifiedMemoryUnit互換オブジェクトのリスト
        ngram_n: CJK文字列に適用するN-gramサイズ
        max_vocab_size: 語彙サイズの上限。超過時はIDF低値の語から除外

    Returns:
        構築されたTfIdfIndex
    """
    if not units:
        return TfIdfIndex()

    # Step 1: 各文書のテキストを収集しトークン化
    doc_tokens: list[tuple[str, list[str]]] = []  # (unit_id, tokens)
    for unit in units:
        unit_id = getattr(unit, "unit_id", "")
        topics = getattr(unit, "topics", []) or []
        summary = getattr(unit, "summary", "") or ""

        # トピックとsummaryを結合してトークン化
        text_parts = [' '.join(str(t) for t in topics), summary]
        combined_text = ' '.join(text_parts)
        tokens = _tokenize(combined_text, ngram_n=ngram_n)
        doc_tokens.append((unit_id, tokens))

    # Step 2: 文書頻度（DF）を計算
    doc_count = len(doc_tokens)
    df: Counter[str] = Counter()
    for _uid, tokens in doc_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1

    # Step 3: 語彙を構築（必要に応じてIDF低値から除外）
    # IDF = log(N / df_t) + 1 （スムージング付き）
    token_idf: dict[str, float] = {}
    for token, freq in df.items():
        token_idf[token] = math.log(doc_count / freq) + 1.0

    # 語彙上限の適用: IDF低値（=多くの文書に出現する非特徴的な語）から除外
    if len(token_idf) > max_vocab_size:
        # IDF値でソートし、上位max_vocab_size件を保持
        sorted_tokens = sorted(token_idf.items(), key=lambda x: x[1], reverse=True)
        token_idf = dict(sorted_tokens[:max_vocab_size])

    # 語彙辞書: token → integer ID
    vocabulary: dict[str, int] = {}
    idf: dict[int, float] = {}
    for idx, (token, idf_val) in enumerate(sorted(token_idf.items())):
        vocabulary[token] = idx
        idf[idx] = idf_val

    # Step 4: 各文書のTF-IDFベクトルを構築
    memory_vectors: dict[str, dict[int, float]] = {}
    for uid, tokens in doc_tokens:
        if not tokens:
            continue
        # TF: 正規化された語頻度
        tf_counts = Counter(tokens)
        max_tf = max(tf_counts.values())
        vec: dict[int, float] = {}
        for token, count in tf_counts.items():
            if token not in vocabulary:
                continue
            tid = vocabulary[token]
            tf = count / max_tf  # augmented TF normalization
            weight = tf * idf[tid]
            if weight > 0:
                vec[tid] = weight
        if vec:
            memory_vectors[uid] = vec

    return TfIdfIndex(
        vocabulary=vocabulary,
        idf=idf,
        memory_vectors=memory_vectors,
        doc_count=doc_count,
        built_at=time.time(),
        ngram_n=ngram_n,
    )


def _vectorize_query(
    text: str,
    vocabulary: dict[str, int],
    idf: dict[int, float],
    ngram_n: int = DEFAULT_NGRAM_N,
) -> dict[int, float]:
    """クエリテキストをTF-IDFベクトルに変換する。

    Args:
        text: クエリテキスト
        vocabulary: 語彙辞書（token → ID）
        idf: IDF辞書（ID → IDF値）
        ngram_n: N-gramサイズ

    Returns:
        疎ベクトル（token_id → TF-IDF重み）
    """
    if not text or not vocabulary:
        return {}

    tokens = _tokenize(text, ngram_n=ngram_n)
    if not tokens:
        return {}

    tf_counts = Counter(tokens)
    max_tf = max(tf_counts.values())
    vec: dict[int, float] = {}
    for token, count in tf_counts.items():
        if token not in vocabulary:
            continue
        tid = vocabulary[token]
        tf = count / max_tf
        idf_val = idf.get(tid, 0.0)
        weight = tf * idf_val
        if weight > 0:
            vec[tid] = weight
    return vec


def _cosine_similarity(v1: dict[int, float], v2: dict[int, float]) -> float:
    """2つの疎ベクトル間のcosine similarityを計算する。

    Args:
        v1: 疎ベクトル1（token_id → 重み）
        v2: 疎ベクトル2（token_id → 重み）

    Returns:
        cosine similarity値（-1.0 〜 1.0）。空ベクトルの場合は0.0
    """
    if not v1 or not v2:
        return 0.0

    # 内積: 共通キーのみ走査（疎ベクトル最適化）
    dot = 0.0
    # 小さい方をイテレートして大きい方をルックアップ
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    for key, val in v1.items():
        if key in v2:
            dot += val * v2[key]

    if dot == 0.0:
        return 0.0

    # ノルム計算
    norm1 = math.sqrt(sum(v * v for v in v1.values()))
    norm2 = math.sqrt(sum(v * v for v in v2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


# =============================================================================
# TF-IDF Index Structure
# =============================================================================

@dataclass
class TfIdfIndex:
    """TF-IDFインデックス構造。

    記憶群から構築された不変の索引。語彙辞書、IDF辞書、記憶ベクトル群を保持する。
    インデックス構築は純粋関数 _build_tfidf_index で行い、
    このデータクラスは結果の格納とシリアライズのみを担う。
    """
    # 語彙辞書: token → integer ID
    vocabulary: dict[str, int] = field(default_factory=dict)
    # IDF辞書: token_id → IDF値
    idf: dict[int, float] = field(default_factory=dict)
    # 記憶ベクトル群: unit_id → {token_id: TF-IDF weight} (疎表現)
    memory_vectors: dict[str, dict[int, float]] = field(default_factory=dict)
    # 構築時点の記憶数
    doc_count: int = 0
    # 構築タイムスタンプ
    built_at: float = 0.0
    # N-gramパラメータ
    ngram_n: int = DEFAULT_NGRAM_N

    def is_empty(self) -> bool:
        """語彙が空かどうかを返す。フォールバック判定に使用。"""
        return len(self.vocabulary) == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON互換の辞書に変換する。"""
        return {
            "vocabulary": dict(self.vocabulary),
            "idf": {str(k): v for k, v in self.idf.items()},
            "memory_vectors": {
                uid: {str(tid): w for tid, w in vec.items()}
                for uid, vec in self.memory_vectors.items()
            },
            "doc_count": self.doc_count,
            "built_at": self.built_at,
            "ngram_n": self.ngram_n,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TfIdfIndex":
        """辞書からTfIdfIndexを復元する。"""
        if not data:
            return cls()
        return cls(
            vocabulary=dict(data.get("vocabulary", {})),
            idf={int(k): v for k, v in data.get("idf", {}).items()},
            memory_vectors={
                uid: {int(tid): w for tid, w in vec.items()}
                for uid, vec in data.get("memory_vectors", {}).items()
            },
            doc_count=data.get("doc_count", 0),
            built_at=data.get("built_at", 0.0),
            ngram_n=data.get("ngram_n", DEFAULT_NGRAM_N),
        )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RecallCandidate:
    """想起候補。経路ラベル付きの単一候補。

    全候補は等価であり、重み・スコア・優先度を持たない。
    """
    candidate_id: str = field(default_factory=_gen_id)
    unit_id: str = ""
    source_id: str = ""
    summary: str = ""
    path_label: str = RecallPathLabel.EMOTIONAL.value
    recall_timestamp: float = field(default_factory=time.time)
    input_snapshot: str = ""  # 想起時の入力断面の要約

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "unit_id": self.unit_id,
            "source_id": self.source_id,
            "summary": self.summary,
            "path_label": self.path_label,
            "recall_timestamp": self.recall_timestamp,
            "input_snapshot": self.input_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecallCandidate":
        return cls(
            candidate_id=data.get("candidate_id", _gen_id()),
            unit_id=data.get("unit_id", ""),
            source_id=data.get("source_id", ""),
            summary=data.get("summary", ""),
            path_label=data.get("path_label", RecallPathLabel.EMOTIONAL.value),
            recall_timestamp=data.get("recall_timestamp", time.time()),
            input_snapshot=data.get("input_snapshot", ""),
        )


@dataclass
class PathStatistics:
    """経路別統計の断面記述。

    記述のみであり、経路の重み付けには使用しない。
    """
    emotional_count: int = 0
    contextual_count: int = 0
    temporal_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotional_count": self.emotional_count,
            "contextual_count": self.contextual_count,
            "temporal_count": self.temporal_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathStatistics":
        return cls(
            emotional_count=data.get("emotional_count", 0),
            contextual_count=data.get("contextual_count", 0),
            temporal_count=data.get("temporal_count", 0),
        )


# =============================================================================
# Input Snapshots
# =============================================================================

@dataclass
class EmotionSnapshot:
    """現在の感情断面。READ-ONLY入力。"""
    emotions: dict[str, float] = field(default_factory=dict)  # label -> intensity
    mood_valence: float = 0.0
    dominant_emotion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotions": dict(self.emotions),
            "mood_valence": self.mood_valence,
            "dominant_emotion": self.dominant_emotion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionSnapshot":
        return cls(
            emotions=dict(data.get("emotions", {})),
            mood_valence=data.get("mood_valence", 0.0),
            dominant_emotion=data.get("dominant_emotion", ""),
        )


@dataclass
class ContextSnapshot:
    """対話文脈断面。READ-ONLY入力。"""
    topics: list[str] = field(default_factory=list)
    percept_text: str = ""
    current_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topics": list(self.topics),
            "percept_text": self.percept_text,
            "current_time": self.current_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSnapshot":
        return cls(
            topics=list(data.get("topics", [])),
            percept_text=data.get("percept_text", ""),
            current_time=data.get("current_time", time.time()),
        )


@dataclass
class TemporalSnapshot:
    """時間認知断面。READ-ONLY入力。不在の場合はtick数のみを使用。"""
    snapshot: dict[str, str] = field(default_factory=dict)
    tick_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": dict(self.snapshot),
            "tick_count": self.tick_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalSnapshot":
        return cls(
            snapshot=dict(data.get("snapshot", {})),
            tick_count=data.get("tick_count", 0),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class MultiPathRecallState:
    """多経路想起の永続化可能な状態。"""
    # 直近の想起候補（毎サイクル上書き）
    current_candidates: list[RecallCandidate] = field(default_factory=list)

    # ルーミネーション防止用の直近想起履歴（unit_id → 想起回数 in window）
    recent_recall_history: list[str] = field(default_factory=list)

    # 経路別統計
    path_stats: PathStatistics = field(default_factory=PathStatistics)

    # サイクルカウンタ
    cycle_count: int = 0

    # TF-IDFインデックス（文脈連想経路用）
    tfidf_index: TfIdfIndex = field(default_factory=TfIdfIndex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_candidates": [c.to_dict() for c in self.current_candidates],
            "recent_recall_history": list(self.recent_recall_history),
            "path_stats": self.path_stats.to_dict(),
            "cycle_count": self.cycle_count,
            "tfidf_index": self.tfidf_index.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiPathRecallState":
        return cls(
            current_candidates=[
                RecallCandidate.from_dict(c)
                for c in data.get("current_candidates", [])
            ],
            recent_recall_history=list(data.get("recent_recall_history", [])),
            path_stats=PathStatistics.from_dict(data.get("path_stats", {})),
            cycle_count=data.get("cycle_count", 0),
            tfidf_index=TfIdfIndex.from_dict(data.get("tfidf_index", {})),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiPathRecallConfig:
    """設定。"""
    # 各経路の候補上限（全経路で同一値: 安全弁1）
    per_path_limit: int = 5

    # ルーミネーション防止のスライディングウィンドウサイズ（サイクル数分の履歴保持）
    rumination_window_size: int = field(default_factory=lambda: coefficient_registry.get("description_common", "window_size_30"))

    # 顕著性バイアス抑制: 感情痕跡が弱い記憶の強制混入割合 (0.0-1.0)
    weak_trace_ratio: float = 0.2

    # 感情連想の感情痕跡強度閾値（これ以下を「弱い」と見なす）
    weak_trace_threshold: float = 0.3

    # 時間近接の最大距離（秒）: これ以上古い記憶は時間近接経路では候補にならない
    temporal_max_distance: float = 86400.0  # 1日

    # enrichment に含める候補数
    enrichment_candidate_count: int = 9  # 3経路 x 3件

    # テキスト断片の最大長
    summary_snippet_length: int = 80

    # TF-IDF安全弁: 1セッション内のインデックス再構築回数上限
    # デフォルト10: 通常セッション(数十サイクル)でrebuild_threshold(20%)変動が
    # 発生する回数は10回以下。異常な高速記憶追加パターンを遮断
    max_rebuild_per_session: int = 10

    # TF-IDF安全弁: 語彙サイズ上限
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE

    # TF-IDF安全弁: 記憶数変化率の再構築閾値
    rebuild_threshold: float = 0.2

    # TF-IDF安全弁: テキスト長上限
    text_max_length: int = TEXT_MAX_LENGTH


# =============================================================================
# Processor
# =============================================================================

class MultiPathRecallProcessor:
    """記憶の多経路想起プロセッサ。

    3経路の想起候補生成 + 安全弁適用の2段構成。
    出力は参照情報としてのみ流れ、判断・行動・評価を直接引き起こさない。
    忘却処理の参照頻度に通知する経路を持たない。
    """

    def __init__(self, config: Optional[MultiPathRecallConfig] = None):
        self._config = config or MultiPathRecallConfig()
        self._state = MultiPathRecallState()
        self._rebuild_count: int = 0  # セッション内再構築カウンタ

    @property
    def state(self) -> MultiPathRecallState:
        return self._state

    @state.setter
    def state(self, value: MultiPathRecallState) -> None:
        self._state = value

    # ─── Main: 3経路の想起実行 ──────────────────────────────────

    def recall_all_paths(
        self,
        unified_units: Optional[list[Any]] = None,
        binding_store: Any = None,
        forgetting_state: Any = None,
        emotion_snapshot: Optional[EmotionSnapshot] = None,
        context_snapshot: Optional[ContextSnapshot] = None,
        temporal_snapshot: Optional[TemporalSnapshot] = None,
    ) -> list[RecallCandidate]:
        """3経路の想起を実行し、安全弁を適用した候補リストを返す。

        入力:
        - unified_units: 記憶系統統合の共通記述単位リスト (UnifiedMemoryUnit互換)
        - binding_store: 感情記憶紐付けのBindingStore
        - forgetting_state: 忘却/固定化のForgettingFixationState
        - emotion_snapshot: 現在の感情断面
        - context_snapshot: 対話文脈断面
        - temporal_snapshot: 時間認知断面

        Returns:
            3経路の想起候補リスト（経路ラベル付き、等価並列）
        """
        self._state.cycle_count += 1
        now = time.time()

        units = unified_units or []
        emo = emotion_snapshot or EmotionSnapshot()
        ctx = context_snapshot or ContextSnapshot(current_time=now)
        temp = temporal_snapshot or TemporalSnapshot()

        # 忘却段階フィルタ: INVISIBLE記憶を除外（安全弁4）
        visible_units = self._filter_invisible(units, forgetting_state)

        if not visible_units:
            self._state.current_candidates = []
            self._state.path_stats = PathStatistics()
            return []

        # 経路1: 感情連想
        emotional_candidates = self._recall_emotional(
            visible_units, binding_store, emo, now,
        )

        # 経路2: 文脈連想
        contextual_candidates = self._recall_contextual(
            visible_units, ctx, now,
        )

        # 経路3: 時間近接
        temporal_candidates = self._recall_temporal(
            visible_units, ctx, temp, now,
        )

        # 安全弁適用
        all_candidates = self._apply_safety_valves(
            emotional_candidates, contextual_candidates, temporal_candidates,
        )

        # 状態更新
        self._state.current_candidates = all_candidates
        self._state.path_stats = PathStatistics(
            emotional_count=len([c for c in all_candidates if c.path_label == RecallPathLabel.EMOTIONAL.value]),
            contextual_count=len([c for c in all_candidates if c.path_label == RecallPathLabel.CONTEXTUAL.value]),
            temporal_count=len([c for c in all_candidates if c.path_label == RecallPathLabel.TEMPORAL.value]),
        )

        # ルーミネーション防止履歴の更新
        for c in all_candidates:
            self._state.recent_recall_history.append(c.unit_id)
        # スライディングウィンドウのトリミング
        max_history = self._config.rumination_window_size * self._config.per_path_limit * 3
        if len(self._state.recent_recall_history) > max_history:
            self._state.recent_recall_history = self._state.recent_recall_history[-max_history:]

        logger.debug(
            "Multi-path recall: cycle=%d, emotional=%d, contextual=%d, temporal=%d, total=%d",
            self._state.cycle_count,
            self._state.path_stats.emotional_count,
            self._state.path_stats.contextual_count,
            self._state.path_stats.temporal_count,
            len(all_candidates),
        )

        return all_candidates

    # ─── 経路1: 感情連想 ───────────────────────────────────────

    def _recall_emotional(
        self,
        units: list[Any],
        binding_store: Any,
        emo: EmotionSnapshot,
        now: float,
    ) -> list[RecallCandidate]:
        """感情連想経路: 現在の感情状態と記憶の感情痕跡の近接度による候補列挙。

        安全弁2（顕著性バイアス抑制）も経路内で適用する。
        """
        cfg = self._config

        if not emo.emotions and abs(emo.mood_valence) < 0.01:
            return []

        scored: list[tuple[float, Any]] = []
        weak_scored: list[tuple[float, Any]] = []

        # BindingStoreから感情痕跡マップを構築
        trace_map = self._build_trace_map(binding_store)

        for unit in units:
            unit_id = getattr(unit, "unit_id", "")
            emotional_valence = getattr(unit, "emotional_valence", 0.0)
            emotional_label = getattr(unit, "emotional_label", "")

            # 感情近接度の計算
            proximity = 0.0

            # ユニット自身の感情属性との近接
            if emotional_label and emotional_label in emo.emotions:
                proximity += emo.emotions[emotional_label] * 0.5

            # 感情価の方向一致
            if abs(emo.mood_valence) > 0.01 and abs(emotional_valence) > 0.01:
                if emo.mood_valence * emotional_valence > 0:
                    proximity += abs(emo.mood_valence * emotional_valence) * 0.3

            # BindingStoreの痕跡との近接
            binding_traces = trace_map.get(unit_id, [])
            for trace_label, trace_intensity, _trace_valence in binding_traces:
                if trace_label in emo.emotions:
                    proximity += emo.emotions[trace_label] * trace_intensity * 0.2

            # 弱い痕跡 / 強い痕跡の分類
            max_trace_intensity = 0.0
            for _, trace_intensity, _ in binding_traces:
                if trace_intensity > max_trace_intensity:
                    max_trace_intensity = trace_intensity

            if max_trace_intensity <= cfg.weak_trace_threshold and abs(emotional_valence) <= cfg.weak_trace_threshold:
                weak_scored.append((proximity, unit))
            else:
                scored.append((proximity, unit))

        # 安全弁2: 顕著性バイアス抑制 - 弱い痕跡の一定割合を強制混入
        limit = cfg.per_path_limit
        weak_slots = max(1, int(limit * cfg.weak_trace_ratio))
        strong_slots = limit - weak_slots

        scored.sort(key=lambda x: x[0], reverse=True)
        weak_scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[Any] = []
        for _, unit in scored[:strong_slots]:
            selected.append(unit)
        for _, unit in weak_scored[:weak_slots]:
            selected.append(unit)

        # 強い痕跡で枠が埋まらなかった場合は弱い痕跡で補充
        if len(selected) < limit:
            remaining_strong = scored[strong_slots:]
            remaining_weak = weak_scored[weak_slots:]
            for _, unit in remaining_weak + remaining_strong:
                if len(selected) >= limit:
                    break
                if unit not in selected:
                    selected.append(unit)

        input_desc = f"emotion={emo.dominant_emotion}, mood={emo.mood_valence:.2f}"
        candidates: list[RecallCandidate] = []
        for unit in selected:
            summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
            candidates.append(RecallCandidate(
                unit_id=getattr(unit, "unit_id", ""),
                source_id=getattr(unit, "source_id", ""),
                summary=summary,
                path_label=RecallPathLabel.EMOTIONAL.value,
                recall_timestamp=now,
                input_snapshot=input_desc,
            ))

        return candidates

    def _build_trace_map(self, binding_store: Any) -> dict[str, list[tuple[str, float, float]]]:
        """BindingStoreから unit_id / memory_key → [(label, intensity, valence)] のマップを構築。"""
        trace_map: dict[str, list[tuple[str, float, float]]] = {}
        if binding_store is None:
            return trace_map

        bindings = getattr(binding_store, "bindings", ()) or ()
        for binding in bindings:
            memory_key = getattr(binding, "memory_key", "")
            if not memory_key:
                continue
            traces = getattr(binding, "traces", ()) or ()
            trace_list: list[tuple[str, float, float]] = []
            for trace in traces:
                label = getattr(trace, "emotion_label", "")
                intensity = getattr(trace, "intensity", 0.0)
                valence = getattr(trace, "valence", 0.0)
                if label:
                    trace_list.append((label, intensity, valence))
            if trace_list:
                trace_map[memory_key] = trace_list

        return trace_map

    # ─── 経路2: 文脈連想 ───────────────────────────────────────

    def _recall_contextual(
        self,
        units: list[Any],
        ctx: ContextSnapshot,
        now: float,
    ) -> list[RecallCandidate]:
        """文脈連想経路: TF-IDF + cosine similarityによるスコアリング。

        インデックスが利用不能（語彙空 or 再構築上限到達で未構築）の場合、
        旧Jaccard方式の_recall_contextual_fallbackにフォールバックする。
        """
        cfg = self._config

        if not ctx.topics and not ctx.percept_text:
            return []

        # インデックスの構築/更新を試行
        self._ensure_index(units)

        # フォールバック判定: インデックスの語彙が空
        if self._state.tfidf_index.is_empty():
            return self._recall_contextual_fallback(units, ctx, now)

        idx = self._state.tfidf_index

        # クエリベクトルの構築（percept_text + topics）
        query_parts = []
        if ctx.topics:
            query_parts.append(' '.join(str(t) for t in ctx.topics))
        if ctx.percept_text:
            query_parts.append(ctx.percept_text)
        query_text = ' '.join(query_parts)

        query_vec = _vectorize_query(query_text, idx.vocabulary, idx.idf, idx.ngram_n)
        if not query_vec:
            return self._recall_contextual_fallback(units, ctx, now)

        # 各記憶とのcosine similarityを計算
        scored: list[tuple[float, Any]] = []
        for unit in units:
            unit_id = getattr(unit, "unit_id", "")
            mem_vec = idx.memory_vectors.get(unit_id, {})
            if not mem_vec:
                continue
            sim = _cosine_similarity(query_vec, mem_vec)
            if sim > 0:
                scored.append((sim, unit))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:cfg.per_path_limit]

        input_desc = f"topics={','.join(ctx.topics[:3])}" if ctx.topics else f"text={ctx.percept_text[:30]}"

        candidates: list[RecallCandidate] = []
        for _, unit in selected:
            summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
            candidates.append(RecallCandidate(
                unit_id=getattr(unit, "unit_id", ""),
                source_id=getattr(unit, "source_id", ""),
                summary=summary,
                path_label=RecallPathLabel.CONTEXTUAL.value,
                recall_timestamp=now,
                input_snapshot=input_desc,
            ))

        return candidates

    def _recall_contextual_fallback(
        self,
        units: list[Any],
        ctx: ContextSnapshot,
        now: float,
    ) -> list[RecallCandidate]:
        """旧文脈連想方式: Jaccard的トピック重複 + テキスト部分一致。

        TF-IDFインデックスが利用不能な場合のフォールバックとして保持。
        """
        cfg = self._config

        if not ctx.topics and not ctx.percept_text:
            return []

        context_topics = {t.lower() for t in ctx.topics} if ctx.topics else set()
        text_lower = ctx.percept_text.lower() if ctx.percept_text else ""

        scored: list[tuple[float, Any]] = []

        for unit in units:
            unit_topics = getattr(unit, "topics", []) or []
            unit_topic_set = {t.lower() for t in unit_topics}

            relevance = 0.0

            # トピック重複
            if context_topics and unit_topic_set:
                overlap = len(context_topics & unit_topic_set)
                if overlap > 0:
                    relevance += min(1.0, overlap / max(1, len(context_topics))) * 0.6

            # テキスト部分一致
            if text_lower and unit_topics:
                match_count = sum(1 for t in unit_topics if t.lower() in text_lower)
                if match_count > 0:
                    relevance += min(0.4, match_count * 0.1)

            if relevance > 0:
                scored.append((relevance, unit))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = scored[:cfg.per_path_limit]
        input_desc = f"topics={','.join(ctx.topics[:3])}" if ctx.topics else f"text={ctx.percept_text[:30]}"

        candidates: list[RecallCandidate] = []
        for _, unit in selected:
            summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
            candidates.append(RecallCandidate(
                unit_id=getattr(unit, "unit_id", ""),
                source_id=getattr(unit, "source_id", ""),
                summary=summary,
                path_label=RecallPathLabel.CONTEXTUAL.value,
                recall_timestamp=now,
                input_snapshot=input_desc,
            ))

        return candidates

    # ─── インデックス管理 ─────────────────────────────────────────

    def _should_rebuild_index(self, units: list[Any]) -> bool:
        """インデックスの再構築が必要かどうかを判定する。

        再構築条件:
        - インデックスが未構築（語彙空）
        - 記憶数がrebuild_threshold以上変化
        """
        idx = self._state.tfidf_index
        if idx.is_empty():
            return True

        current_count = len(units)
        if idx.doc_count == 0:
            return True

        change_rate = abs(current_count - idx.doc_count) / idx.doc_count
        return change_rate >= self._config.rebuild_threshold

    def _ensure_index(self, units: list[Any]) -> None:
        """必要に応じてTF-IDFインデックスを構築/更新する。

        再構築上限（max_rebuild_per_session）に達した場合は構築を行わない。
        """
        if self._rebuild_count >= self._config.max_rebuild_per_session:
            return

        if not self._should_rebuild_index(units):
            return

        try:
            self._state.tfidf_index = _build_tfidf_index(
                units,
                max_vocab_size=self._config.max_vocab_size,
            )
            self._rebuild_count += 1
            logger.debug(
                "TF-IDF index rebuilt: vocab=%d, docs=%d, rebuild_count=%d",
                len(self._state.tfidf_index.vocabulary),
                self._state.tfidf_index.doc_count,
                self._rebuild_count,
            )
        except Exception:
            logger.exception("TF-IDF index build failed, will use fallback")
            # 失敗時はインデックスを空のままにしてフォールバックを使う

    # ─── 経路3: 時間近接 ───────────────────────────────────────

    def _recall_temporal(
        self,
        units: list[Any],
        ctx: ContextSnapshot,
        temp: TemporalSnapshot,
        now: float,
    ) -> list[RecallCandidate]:
        """時間近接経路: 現在時刻と記憶のタイムスタンプの時間的距離による候補列挙。"""
        cfg = self._config
        current_time = ctx.current_time if ctx.current_time > 0 else now

        scored: list[tuple[float, Any]] = []

        for unit in units:
            unit_ts = getattr(unit, "timestamp", 0.0)
            if not isinstance(unit_ts, (int, float)) or unit_ts <= 0:
                continue

            distance = abs(current_time - unit_ts)
            if distance > cfg.temporal_max_distance:
                continue

            # 近いほど高い近接度（距離の逆数的スケーリング）
            proximity = 1.0 - (distance / cfg.temporal_max_distance)
            proximity = _clamp(proximity)

            scored.append((proximity, unit))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = scored[:cfg.per_path_limit]
        input_desc = f"time={current_time:.0f}, tick={temp.tick_count}"

        candidates: list[RecallCandidate] = []
        for _, unit in selected:
            summary = getattr(unit, "summary", "")[:cfg.summary_snippet_length]
            candidates.append(RecallCandidate(
                unit_id=getattr(unit, "unit_id", ""),
                source_id=getattr(unit, "source_id", ""),
                summary=summary,
                path_label=RecallPathLabel.TEMPORAL.value,
                recall_timestamp=now,
                input_snapshot=input_desc,
            ))

        return candidates

    # ─── 安全弁適用 ────────────────────────────────────────────

    def _apply_safety_valves(
        self,
        emotional: list[RecallCandidate],
        contextual: list[RecallCandidate],
        temporal: list[RecallCandidate],
    ) -> list[RecallCandidate]:
        """安全弁を適用した候補リストを返す。

        安全弁1: 経路間等価性（各経路per_path_limit以内は保証済み）
        安全弁3: ルーミネーション防止（直近想起履歴によるスライディングウィンドウ抑制）
        """
        all_candidates = emotional + contextual + temporal

        # 安全弁3: ルーミネーション防止
        all_candidates = self._apply_rumination_suppression(all_candidates)

        return all_candidates

    def _apply_rumination_suppression(
        self, candidates: list[RecallCandidate],
    ) -> list[RecallCandidate]:
        """ルーミネーション防止: 直近想起履歴内の記憶の優先度を下げる。

        抑制は「除外」ではなく「候補としての優先度を下げる」方式。
        他に候補がない場合は再度選出される。
        """
        history = self._state.recent_recall_history
        if not history:
            return candidates

        # 直近ウィンドウ内での想起回数を集計
        window = history[-self._config.rumination_window_size * self._config.per_path_limit * 3:]
        recall_counts: dict[str, int] = {}
        for uid in window:
            recall_counts[uid] = recall_counts.get(uid, 0) + 1

        # 3回以上想起された記憶を「抑制対象」とする
        suppression_threshold = 3

        non_suppressed: list[RecallCandidate] = []
        suppressed: list[RecallCandidate] = []

        for c in candidates:
            if recall_counts.get(c.unit_id, 0) >= suppression_threshold:
                suppressed.append(c)
            else:
                non_suppressed.append(c)

        # 非抑制候補を優先し、抑制候補を末尾に追加（完全除外はしない）
        return non_suppressed + suppressed

    # ─── 忘却処理との分離（安全弁4）──────────────────────────────

    def _filter_invisible(
        self, units: list[Any], forgetting_state: Any,
    ) -> list[Any]:
        """INVISIBLE記憶を想起候補から除外する。

        忘却/固定化の段階情報を参照するが、参照頻度には通知しない。
        段階情報が存在しない記憶は ACTIVE として扱う。
        """
        if forgetting_state is None:
            return list(units)

        # 忘却状態から INVISIBLE な source_id のセットを構築
        invisible_ids: set[str] = set()
        series_index = getattr(forgetting_state, "series_index", []) or []
        for rec in series_index:
            stage = getattr(rec, "forgetting_stage", "active")
            source_id = getattr(rec, "source_id", "")
            if stage == "invisible" and source_id:
                invisible_ids.add(source_id)

        if not invisible_ids:
            return list(units)

        visible: list[Any] = []
        for unit in units:
            source_id = getattr(unit, "source_id", "")
            unit_id = getattr(unit, "unit_id", "")
            if source_id in invisible_ids or unit_id in invisible_ids:
                continue
            visible.append(unit)

        return visible

    # ─── 参照情報の提供 ──────────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        経路ラベル付きの候補一覧として、他のenrichment項目と同列に記述される。
        特定の経路を強調・選別しない。

        Returns:
            enrichment用の構造化データ
        """
        st = self._state
        cfg = self._config

        candidates = st.current_candidates[:cfg.enrichment_candidate_count]

        entries: list[dict[str, Any]] = []
        for c in candidates:
            entries.append({
                "path": c.path_label,
                "summary": c.summary,
                "unit_id": c.unit_id,
            })

        summary_text = get_recall_summary(st)

        return {
            "candidate_count": len(st.current_candidates),
            "path_stats": st.path_stats.to_dict(),
            "entries": entries,
            "summary_text": summary_text,
        }

    def get_recall_candidates(self) -> list[RecallCandidate]:
        """READ-ONLY候補リストを返す。

        想起候補は参照情報としてのみ流れ、判断・行動を確定しない。
        """
        return list(self._state.current_candidates)

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        st = self._state
        return {
            "cycle_count": st.cycle_count,
            "candidate_count": len(st.current_candidates),
            "path_stats": st.path_stats.to_dict(),
            "history_length": len(st.recent_recall_history),
        }


# =============================================================================
# Summary (enrichment 用)
# =============================================================================

def get_recall_summary(state: MultiPathRecallState) -> str:
    """多経路想起状態の要約（enrichment用）。

    全経路を等価に列挙する。特定の経路を強調しない。
    評価判定・行動指示を含まない。
    """
    if not state.current_candidates and state.cycle_count == 0:
        return "多経路想起: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    stats = state.path_stats
    if stats.emotional_count > 0:
        parts.append(f"感情連想={stats.emotional_count}")
    if stats.contextual_count > 0:
        parts.append(f"文脈連想={stats.contextual_count}")
    if stats.temporal_count > 0:
        parts.append(f"時間近接={stats.temporal_count}")

    total = len(state.current_candidates)
    if total > 0:
        parts.append(f"候補合計={total}")
    else:
        parts.append("候補=0")

    return " ".join(parts) if parts else "多経路想起: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_multi_path_recall(
    config: Optional[MultiPathRecallConfig] = None,
) -> MultiPathRecallProcessor:
    """MultiPathRecallProcessor のファクトリ関数。"""
    return MultiPathRecallProcessor(config=config)
