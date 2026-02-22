"""
psyche/emotion_cooccurrence_description.py - 感情間の共起記述

毎ティックにおいて同時に閾値以上で存在した感情の組み合わせを、
事実としてFIFOに等価蓄積する構造。

設計原則 (design_emotion_cooccurrence_description.md 準拠):
- 共起の「望ましさ」「異常さ」「健全さ」を判定しない
- 「共起すべき」「共起すべきでない」という規範を定義しない
- 特定の感情ペアの出現頻度を数えない、集計しない、統計量を算出しない
- 共起パターンを抽出・命名・分類しない
- 共起の記録から「傾向」「癖」「特徴」を推論しない
- 感情処理パイプラインのパラメータ（減衰率、振幅、ムード、連動設定）を変更しない
- 感情状態そのものを変更・矯正・誘導しない
- 共起記録を判断・行動・ポリシー選択に直接接続しない
- 動機レベルの共存検出（別モジュールに存在する衝動ペア検出）と混同しない

安全弁:
1. 収束監視: 蓄積記録全体の共起ペア構成の多様性を段階値（高・中・低）で監視。
   多様性が「低」の場合、鮮度減衰中の異なる構成の記録の鮮度段階を一段復帰させる。
2. 蓄積偏り検出: 直近の連続記録における同一ペア構成の連続を検出。
   連続時に、過去の異なる構成の記録を可視状態に復帰させる。
3. enrichment出力量制限: enrichmentに出力する記録件数に上限を設ける。
4. 解釈的テキストの不付与: enrichment出力は数値列挙に限定する。
5. 頻度情報の遮断: 共起ペアの出現回数の集計・算出・出力を行わない。

経路遮断:
1. 本機能 → 感情処理パイプラインのパラメータ（減衰率、振幅、ムード、連動設定）
2. 本機能 → ポリシー候補拡張への直接断面供給
3. 本機能 → 判断バイアス計算への直接入力
4. 本機能 → 記憶忘却・固定化パラメータの変更
5. 本機能 → 動機生成への直接入力
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class FreshnessStage(Enum):
    """鮮度段階。設計書: 活性→弱化→退行→近不可視→不可視の段階遷移。"""
    ACTIVE = "active"
    WEAKENING = "weakening"
    FADING = "fading"
    NEAR_INVISIBLE = "near_invisible"
    INVISIBLE = "invisible"


class DiversityLevel(Enum):
    """収束監視の多様性段階値。設計書: 高・中・低。"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Helpers
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _stage_from_freshness(freshness: float) -> FreshnessStage:
    """鮮度値から段階を返す。"""
    if freshness >= 0.8:
        return FreshnessStage.ACTIVE
    elif freshness >= 0.6:
        return FreshnessStage.WEAKENING
    elif freshness >= 0.4:
        return FreshnessStage.FADING
    elif freshness >= 0.2:
        return FreshnessStage.NEAR_INVISIBLE
    else:
        return FreshnessStage.INVISIBLE


def _make_pair_key(a: str, b: str) -> tuple[str, str]:
    """ペアの順序を固定的な辞書順とし、順序に意味を持たせない。"""
    return (a, b) if a <= b else (b, a)


def _composition_signature(pairs: list[dict[str, Any]]) -> frozenset[tuple[str, str]]:
    """共起ペア群の構成シグネチャ。頻度情報なし、種類のみ。"""
    return frozenset(
        _make_pair_key(p["emotion_a"], p["emotion_b"])
        for p in pairs
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CooccurrencePair:
    """共起ペア。設計書: 二つの感情名と、それぞれの値で構成。"""
    emotion_a: str = ""
    emotion_b: str = ""
    value_a: float = 0.0
    value_b: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion_a": self.emotion_a,
            "emotion_b": self.emotion_b,
            "value_a": self.value_a,
            "value_b": self.value_b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CooccurrencePair":
        return cls(
            emotion_a=data.get("emotion_a", ""),
            emotion_b=data.get("emotion_b", ""),
            value_a=data.get("value_a", 0.0),
            value_b=data.get("value_b", 0.0),
        )


@dataclass
class CooccurrenceRecord:
    """共起記録。設計書:
    - 一意な識別子
    - 記録生成時刻
    - 共起ペアの集合
    - 共起ペアが存在しなかった場合はその旨のフラグ
    - 鮮度段階
    記録は生成後に内容が変化しない（不変記録）。鮮度のみ変化する。
    """
    record_id: str = ""
    timestamp: float = field(default_factory=time.time)
    tick: int = 0
    pairs: list[CooccurrencePair] = field(default_factory=list)
    no_cooccurrence: bool = False
    freshness: float = 1.0
    freshness_stage: str = FreshnessStage.ACTIVE.value

    def __post_init__(self):
        if not self.record_id:
            self.record_id = _gen_id()

    @property
    def composition_signature(self) -> frozenset[tuple[str, str]]:
        """この記録の構成シグネチャ。"""
        return frozenset(
            _make_pair_key(p.emotion_a, p.emotion_b)
            for p in self.pairs
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "tick": self.tick,
            "pairs": [p.to_dict() for p in self.pairs],
            "no_cooccurrence": self.no_cooccurrence,
            "freshness": self.freshness,
            "freshness_stage": self.freshness_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CooccurrenceRecord":
        return cls(
            record_id=data.get("record_id", ""),
            timestamp=data.get("timestamp", time.time()),
            tick=data.get("tick", 0),
            pairs=[CooccurrencePair.from_dict(p) for p in data.get("pairs", [])],
            no_cooccurrence=data.get("no_cooccurrence", False),
            freshness=data.get("freshness", 1.0),
            freshness_stage=data.get("freshness_stage", FreshnessStage.ACTIVE.value),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class CooccurrenceState:
    """内部状態。"""
    # 共起記録のFIFOバッファ。設計書: 固定容量のFIFO構造。
    records: list[CooccurrenceRecord] = field(default_factory=list)

    # カウンタ（診断情報のみ）
    cycle_count: int = 0
    total_records_created: int = 0
    total_records_decayed: int = 0
    total_records_recovered: int = 0

    # 安全弁フラグ
    diversity_level: str = DiversityLevel.HIGH.value
    accumulation_bias_warning: bool = False
    convergence_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "cycle_count": self.cycle_count,
            "total_records_created": self.total_records_created,
            "total_records_decayed": self.total_records_decayed,
            "total_records_recovered": self.total_records_recovered,
            "diversity_level": self.diversity_level,
            "accumulation_bias_warning": self.accumulation_bias_warning,
            "convergence_warning": self.convergence_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CooccurrenceState":
        return cls(
            records=[
                CooccurrenceRecord.from_dict(r)
                for r in data.get("records", [])
            ],
            cycle_count=data.get("cycle_count", 0),
            total_records_created=data.get("total_records_created", 0),
            total_records_decayed=data.get("total_records_decayed", 0),
            total_records_recovered=data.get("total_records_recovered", 0),
            diversity_level=data.get("diversity_level", DiversityLevel.HIGH.value),
            accumulation_bias_warning=data.get("accumulation_bias_warning", False),
            convergence_warning=data.get("convergence_warning", False),
        )

    def apply_session_decay(self, decay_factor: float = 0.3) -> None:
        """セッション境界での一律鮮度減衰。"""
        remove_ids: set[str] = set()
        for rec in self.records:
            rec.freshness = _clamp(rec.freshness - decay_factor)
            rec.freshness_stage = _stage_from_freshness(rec.freshness).value
            if rec.freshness < 0.1:
                remove_ids.add(rec.record_id)
        if remove_ids:
            self.records = [
                r for r in self.records if r.record_id not in remove_ids
            ]


# =============================================================================
# Result
# =============================================================================

@dataclass
class CooccurrenceResult:
    """処理結果（参照情報形式のみ）。"""
    record_count: int = 0
    visible_count: int = 0
    pair_count: int = 0
    no_cooccurrence: bool = False
    diversity_level: str = DiversityLevel.HIGH.value
    accumulation_bias_warning: bool = False
    convergence_warning: bool = False
    diversity_restored: bool = False
    cycle_count: int = 0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CooccurrenceConfig:
    """設定。"""
    # FIFOバッファの上限件数
    max_records: int = 50

    # 共起閾値（設計書: 外部から設定可能であり、固定値ではない）
    cooccurrence_threshold: float = 0.15

    # 鮮度減衰速度（処理サイクル毎）
    freshness_decay_rate: float = 0.02

    # 多様性復元時の鮮度回復量
    diversity_recovery_amount: float = 0.15

    # enrichmentに含める直近記録の件数上限（安全弁3）
    max_enrichment_records: int = 5

    # enrichment出力のサイズ上限（文字数、安全弁3）
    max_enrichment_length: int = 1500

    # 蓄積偏り検出の連続同一構成数閾値
    bias_consecutive_threshold: int = 3

    # 収束監視の低多様性閾値（種類数がこれ以下で「低」）
    low_diversity_threshold: int = 2

    # 収束監視の中多様性閾値（種類数がこれ以下で「中」）
    medium_diversity_threshold: int = 4


# =============================================================================
# Processor
# =============================================================================

class EmotionCooccurrenceDescriptionProcessor:
    """感情間の共起記述プロセッサ。

    設計書の仕組みに従い、毎ティック以下の処理を行う:
    1. 感情状態の取得: 複数感情独立管理から感情ベクトルの全値を読み取る
    2. 共起ペアの同定: 閾値以上の感情を列挙し、全ペアを構成
    3. 共起記録の生成: 当該ティックの共起ペア群を一件の記録として構成
    4. FIFO蓄積: FIFOバッファに追加、容量超過時は最古を自然消失
    5. 鮮度減衰: 時間経過に応じた段階的な鮮度減衰
    6. 受渡準備: 可視状態のものを等価に列挙

    感情処理パイプラインのパラメータを一切変更しない。
    出力は参照情報形式のみ。
    """

    def __init__(self, config: Optional[CooccurrenceConfig] = None):
        self._config = config or CooccurrenceConfig()
        self._state = CooccurrenceState()

    @property
    def state(self) -> CooccurrenceState:
        return self._state

    @state.setter
    def state(self, value: CooccurrenceState) -> None:
        self._state = value

    def tick(self, emotion_values: dict[str, float]) -> CooccurrenceResult:
        """orchestrator から呼ばれる単一エントリポイント。

        Args:
            emotion_values: 感情ベクトルの全値（READ-ONLY参照）。
                            これが唯一の主入力源。

        Returns:
            処理結果（参照情報形式のみ）。
        """
        return self.process(emotion_values)

    def process(self, emotion_values: dict[str, float]) -> CooccurrenceResult:
        """メイン処理を実行する。"""
        self._state.cycle_count += 1
        now = time.time()
        cfg = self._config
        tick = self._state.cycle_count

        # Step 1: 感情状態の取得 (READ-ONLY)
        # emotion_values は外部から渡される。書き込みは行わない。

        # Step 2: 共起ペアの同定
        pairs = self._identify_cooccurrence_pairs(emotion_values)

        # Step 3: 共起記録の生成
        record = self._create_record(pairs, tick, now)

        # Step 4: FIFO蓄積
        self._accumulate(record)

        # Step 5: 鮮度減衰
        self._apply_freshness_decay(now)

        # Step 6: 受渡準備 + 安全弁チェック
        return self._prepare_handoff(now)

    def _identify_cooccurrence_pairs(
        self, emotion_values: dict[str, float]
    ) -> list[CooccurrencePair]:
        """閾値以上の感情を列挙し、同時に閾値以上である感情の全ペアを構成する。
        設計書: ペアの順序は固定的な辞書順とし、順序に意味を持たせない。
        """
        cfg = self._config
        threshold = cfg.cooccurrence_threshold

        # 閾値以上の感情を列挙
        active_emotions: list[tuple[str, float]] = [
            (name, value)
            for name, value in emotion_values.items()
            if value >= threshold
        ]

        # 2つ未満の場合、共起ペアは存在しない
        if len(active_emotions) < 2:
            return []

        # 全ペアを構成（辞書順）
        pairs: list[CooccurrencePair] = []
        for i, (name_a, val_a) in enumerate(active_emotions):
            for name_b, val_b in active_emotions[i + 1:]:
                a, b = _make_pair_key(name_a, name_b)
                va = val_a if a == name_a else val_b
                vb = val_b if b == name_b else val_a
                pairs.append(CooccurrencePair(
                    emotion_a=a,
                    emotion_b=b,
                    value_a=va,
                    value_b=vb,
                ))

        return pairs

    def _create_record(
        self,
        pairs: list[CooccurrencePair],
        tick: int,
        now: float,
    ) -> CooccurrenceRecord:
        """当該ティックにおける共起ペア群を一件の記録として構成する。
        設計書: 共起ペアが存在しない場合は「共起なし」として記録する。
        """
        no_cooccurrence = len(pairs) == 0
        return CooccurrenceRecord(
            timestamp=now,
            tick=tick,
            pairs=pairs,
            no_cooccurrence=no_cooccurrence,
            freshness=1.0,
            freshness_stage=FreshnessStage.ACTIVE.value,
        )

    def _accumulate(self, record: CooccurrenceRecord) -> None:
        """FIFO蓄積。設計書: バッファ容量を超過した場合は最も古い記録が自然消失する。"""
        cfg = self._config
        self._state.records.append(record)
        self._state.total_records_created += 1

        # FIFO: 容量超過時は最古を押し出し（唯一の物理的消失経路）
        if len(self._state.records) > cfg.max_records:
            overflow = len(self._state.records) - cfg.max_records
            self._state.records = self._state.records[overflow:]

    def _apply_freshness_decay(self, now: float) -> None:
        """鮮度減衰。設計書: 各記録の鮮度が時間経過に応じて段階的に減衰する。
        活性→弱化→退行→近不可視→不可視の段階遷移。
        """
        cfg = self._config
        for rec in self._state.records:
            old_freshness = rec.freshness
            rec.freshness = _clamp(rec.freshness - cfg.freshness_decay_rate)
            new_stage = _stage_from_freshness(rec.freshness)
            old_stage = rec.freshness_stage

            if new_stage.value != old_stage:
                rec.freshness_stage = new_stage.value
                if new_stage == FreshnessStage.INVISIBLE:
                    self._state.total_records_decayed += 1

    def _prepare_handoff(self, now: float) -> CooccurrenceResult:
        """受渡準備。安全弁チェックを行い結果を返す。
        出力は参照情報形式のみ。
        """
        cfg = self._config
        st = self._state

        # 可視状態の記録
        visible_records = [
            r for r in st.records
            if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
        ]
        visible_count = len(visible_records)

        # 最新の記録情報
        latest = st.records[-1] if st.records else None
        pair_count = len(latest.pairs) if latest else 0
        no_cooccurrence = latest.no_cooccurrence if latest else True

        # 安全弁1: 収束監視
        diversity_level = self._monitor_diversity()
        st.diversity_level = diversity_level.value

        # 安全弁2: 蓄積偏り検出
        diversity_restored = False
        st.accumulation_bias_warning = self._check_accumulation_bias()
        if st.accumulation_bias_warning:
            diversity_restored = self._restore_diversity()

        # 収束監視で多様性が「低」の場合も復帰
        if diversity_level == DiversityLevel.LOW:
            st.convergence_warning = True
            if not diversity_restored:
                diversity_restored = self._restore_diversity()
        else:
            st.convergence_warning = False

        return CooccurrenceResult(
            record_count=len(st.records),
            visible_count=visible_count,
            pair_count=pair_count,
            no_cooccurrence=no_cooccurrence,
            diversity_level=st.diversity_level,
            accumulation_bias_warning=st.accumulation_bias_warning,
            convergence_warning=st.convergence_warning,
            diversity_restored=diversity_restored,
            cycle_count=st.cycle_count,
        )

    # ─── 安全弁1: 収束監視 ──────────────────────────────────────

    def _monitor_diversity(self) -> DiversityLevel:
        """蓄積記録全体の共起ペア構成の多様性を段階値で監視する。
        設計書: 異なるペア構成の種類数。頻度情報は扱わない（安全弁5）。
        """
        cfg = self._config
        visible_records = [
            r for r in self._state.records
            if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
            and not r.no_cooccurrence
        ]

        if not visible_records:
            return DiversityLevel.HIGH  # データ不足時はデフォルト

        # 異なる構成シグネチャの種類数（安全弁5: 回数は数えない）
        signatures: set[frozenset[tuple[str, str]]] = set()
        for rec in visible_records:
            sig = rec.composition_signature
            if sig:
                signatures.add(sig)

        distinct_count = len(signatures)

        if distinct_count <= cfg.low_diversity_threshold:
            return DiversityLevel.LOW
        elif distinct_count <= cfg.medium_diversity_threshold:
            return DiversityLevel.MEDIUM
        else:
            return DiversityLevel.HIGH

    # ─── 安全弁2: 蓄積偏り検出 ──────────────────────────────────

    def _check_accumulation_bias(self) -> bool:
        """直近の連続記録における同一ペア構成の連続を検出する。"""
        cfg = self._config
        records = self._state.records

        if len(records) < cfg.bias_consecutive_threshold:
            return False

        # 直近N件のシグネチャを比較
        recent = records[-cfg.bias_consecutive_threshold:]
        signatures = [r.composition_signature for r in recent]

        # 全て同一かチェック
        if not signatures:
            return False

        first = signatures[0]
        return all(s == first for s in signatures)

    # ─── 安全弁 復帰処理 ──────────────────────────────────────

    def _restore_diversity(self) -> bool:
        """設計書: 鮮度減衰中の異なる構成の記録の鮮度段階を一段復帰させる。
        これにより、特定の共起パターンのみが蓄積に残る状態を抑制する。
        """
        cfg = self._config
        restored = False

        # 最新記録の構成シグネチャ
        latest_sig = None
        for rec in reversed(self._state.records):
            if not rec.no_cooccurrence:
                latest_sig = rec.composition_signature
                break

        for rec in self._state.records:
            stage = _stage_from_freshness(rec.freshness)
            if stage in (FreshnessStage.FADING, FreshnessStage.NEAR_INVISIBLE):
                # 異なる構成の記録を復帰
                if latest_sig is None or rec.composition_signature != latest_sig:
                    rec.freshness = _clamp(rec.freshness + cfg.diversity_recovery_amount)
                    rec.freshness_stage = _stage_from_freshness(rec.freshness).value
                    self._state.total_records_recovered += 1
                    restored = True

        return restored

    # ─── enrichment ──────────────────────────────────────────

    def get_enrichment_data(self) -> dict[str, Any]:
        """prompt enrichment 用の構造化データを返す。

        設計書:
        - 出力は「直近N件の共起事実の等価列挙」
        - 列挙された記録間に順位・優劣・重要度の差は存在しない
        - 出力にパターン名・傾向名・分類ラベルは含まれない
        - 出力に頻度情報は含まれない（安全弁5）
        - enrichmentに出力する記録件数に上限を設ける（安全弁3）
        - 数値列挙に限定し、解釈的テキストを付与しない（安全弁4）
        """
        st = self._state
        cfg = self._config

        if not st.records:
            return {
                "summary_text": "感情共起: 待機中",
                "record_count": 0,
                "visible_count": 0,
                "entries": [],
            }

        # 可視状態の記録（安全弁: enrichment出力から不可視は除外）
        visible = [
            r for r in st.records
            if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
        ]
        visible_count = len(visible)

        # 直近N件（安全弁3: enrichment出力量制限）
        recent = visible[-cfg.max_enrichment_records:]

        entries: list[dict[str, Any]] = []
        for rec in recent:
            if rec.no_cooccurrence:
                entries.append({
                    "tick": rec.tick,
                    "no_cooccurrence": True,
                    "freshness_stage": rec.freshness_stage,
                })
            else:
                # 安全弁4: 数値列挙に限定。解釈的テキストを付与しない。
                # 安全弁5: 頻度情報を含まない。
                pair_list = [
                    {
                        "a": p.emotion_a,
                        "b": p.emotion_b,
                        "va": round(p.value_a, 3),
                        "vb": round(p.value_b, 3),
                    }
                    for p in rec.pairs
                ]
                entries.append({
                    "tick": rec.tick,
                    "pairs": pair_list,
                    "freshness_stage": rec.freshness_stage,
                })

        summary_text = get_cooccurrence_summary(st)

        return {
            "summary_text": summary_text,
            "record_count": len(st.records),
            "visible_count": visible_count,
            "entries": entries,
            "diversity_level": st.diversity_level,
            "accumulation_bias_warning": st.accumulation_bias_warning,
            "convergence_warning": st.convergence_warning,
        }

    # ─── READ-ONLYアクセサ ──────────────────────────────────────

    def get_records(self) -> list[dict[str, Any]]:
        """全記録をREAD-ONLYで返す。フィルタリング・選別・集約機能を持たない。"""
        return [r.to_dict() for r in self._state.records]

    def get_visible_records(self) -> list[dict[str, Any]]:
        """可視状態の記録をREAD-ONLYで返す。"""
        return [
            r.to_dict() for r in self._state.records
            if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
        ]

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "record_count": len(self._state.records),
            "cycle_count": self._state.cycle_count,
            "diversity_level": self._state.diversity_level,
            "accumulation_bias_warning": self._state.accumulation_bias_warning,
            "convergence_warning": self._state.convergence_warning,
        }

    # ─── Save / Load ──────────────────────────────────────────────

    def save(self) -> dict[str, Any]:
        """永続化用のデータを返す。"""
        return self._state.to_dict()

    def load(self, data: dict[str, Any]) -> None:
        """永続化データから状態を復元する。"""
        self._state = CooccurrenceState.from_dict(data)
        logger.debug(
            "Emotion cooccurrence description state loaded: records=%d",
            len(self._state.records),
        )


# =============================================================================
# Summary
# =============================================================================

def get_cooccurrence_summary(state: CooccurrenceState) -> str:
    """enrichment用の要約テキスト。
    安全弁4: 数値列挙に限定し、解釈的テキストを付与しない。
    安全弁5: 頻度情報を含まない。
    """
    if state.cycle_count == 0 and not state.records:
        return "感情共起: 待機中"

    parts: list[str] = []
    parts.append(f"cycle={state.cycle_count}")

    visible_count = sum(
        1 for r in state.records
        if _stage_from_freshness(r.freshness) != FreshnessStage.INVISIBLE
    )
    if visible_count:
        parts.append(f"記録={visible_count}")

    # 最新記録のペア数のみ（頻度情報ではない）
    if state.records:
        latest = state.records[-1]
        if latest.no_cooccurrence:
            parts.append("共起なし")
        else:
            parts.append(f"ペア={len(latest.pairs)}")

    parts.append(f"多様性={state.diversity_level}")

    if state.accumulation_bias_warning:
        parts.append("蓄積偏り")
    if state.convergence_warning:
        parts.append("収束")

    return " ".join(parts) if parts else "感情共起: 待機中"


# =============================================================================
# Factory
# =============================================================================

def create_cooccurrence_processor(
    config: Optional[CooccurrenceConfig] = None,
) -> EmotionCooccurrenceDescriptionProcessor:
    """EmotionCooccurrenceDescriptionProcessor のファクトリ関数。"""
    return EmotionCooccurrenceDescriptionProcessor(config=config)
