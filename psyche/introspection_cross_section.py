"""
psyche/introspection_cross_section.py - 内省断面間の横断的記述（内省の文脈層）

複数の内省系モジュールの出力を同時期に並べて提示する構造を提供する。
横断的記述 = 並置（juxtaposition）であり、統合（integration）ではない。

設計原則 (design_introspection_cross_section.md 準拠):
- 断面間の相関・因果・傾向を算出しない
- 断面を統合して単一の状態記述にまとめない
- 特定の断面に重み付け・注目度・重要度を付与しない（全断面等価）
- 評価的語彙を使用しない（良好/異常/正常/理想的/問題/改善/乱れ/安定）
- パターン抽出・統計処理を行わない
- 感情パイプライン（Phase 1-2）のパラメータを変更しない
- いかなるモジュールの内部状態・処理パラメータに書き込まない（READ-ONLY参照）

3段パイプライン:
1. 断面値の収集 (section value collection)
2. スナップショットの構成と蓄積 (snapshot construction and accumulation)
3. 参照情報としての受渡準備 (handoff preparation as reference information)

安全弁:
1. パターン抽出の禁止（断面間の相関・因果・傾向の算出禁止）
2. 全断面の等価性（重み付け・注目度・重要度の付与禁止）
3. 統合の禁止（単一要約・単一スコアへの集約禁止）
4. ウィンドウサイズの制限（長期蓄積の構造的制限）
5. 判断系・行動系・感情パイプラインへの書き込み経路の遮断

経路遮断:
1. 横断的記述 → 各内省系モジュールの入力
2. 横断的記述 → ポリシー選択パイプライン
3. 横断的記述 → 感情パイプライン
4. 横断的記述 → 記憶忘却/固定化パラメータ
5. 横断的記述 → 予期形成
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Constants: Section Definitions (断面定義)
# =============================================================================

# 初期対象6モジュールの断面識別名。定義順に固定、変更禁止。
SECTION_SELF_MODEL = "self_model"
SECTION_TEMPORAL_SELF_DIFFERENCE = "temporal_self_difference"
SECTION_IDENTITY_COHERENCE = "identity_coherence"
SECTION_SELF_NARRATIVE = "self_narrative"
SECTION_INTROSPECTION_CONSUMPTION = "introspection_consumption"
SECTION_META_EMOTION_COGNITION = "meta_emotion_cognition"

# 定義順序（固定、変更禁止）。列挙順序のランダム化・最適化は行わない。
SECTION_ORDER = [
    SECTION_SELF_MODEL,
    SECTION_TEMPORAL_SELF_DIFFERENCE,
    SECTION_IDENTITY_COHERENCE,
    SECTION_SELF_NARRATIVE,
    SECTION_INTROSPECTION_CONSUMPTION,
    SECTION_META_EMOTION_COGNITION,
]

# 断面の日本語ラベル（enrichment用）。全断面等価に列挙するためのラベル。
SECTION_LABELS = {
    SECTION_SELF_MODEL: "自己モデル",
    SECTION_TEMPORAL_SELF_DIFFERENCE: "時間的自己差分",
    SECTION_IDENTITY_COHERENCE: "同一性揺らぎ",
    SECTION_SELF_NARRATIVE: "自己物語",
    SECTION_INTROSPECTION_CONSUMPTION: "内省消費",
    SECTION_META_EMOTION_COGNITION: "メタ感情認知",
}

# 不在断面の記述文字列。不在を異常やエラーとして扱わない。
ABSENT_MARKER = "不在"

# 拡張候補モジュール識別名（定義のみ保持、処理対象にはしない）
EXTENSION_CANDIDATES = [
    "intent_action_gap",
    "temporal_cognition",
    "expectation_formation",
    "intrinsic_motivation",
    "continuity_strain",
    "self_image_integration",
]

# 評価的語彙の禁止リスト（要約生成時に混入しないことを保証する）
_EVALUATIVE_WORDS = frozenset([
    "良好", "異常", "正常", "理想的", "問題", "改善", "乱れ", "安定",
    "良い", "悪い", "適切", "不適切", "最適", "劣化", "向上", "悪化",
])


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CrossSectionSnapshot:
    """内省断面の横断的スナップショット。

    6断面の要約的記述を1組として保持する。
    一度構成されたら変更されない（追記のみの構造）。
    各断面の要約に重み・スコア・優先度・重要度などの評価的属性を付与しない。
    全断面は等価である。
    """
    # 6断面それぞれの要約的記述（等価に並列）
    # key: 断面識別名, value: 要約テキスト
    sections: dict[str, str] = field(default_factory=dict)

    # スナップショット構成時点のティック番号
    tick: int = 0

    # スナップショット構成時点のタイムスタンプ
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": dict(self.sections),
            "tick": self.tick,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossSectionSnapshot":
        return cls(
            sections=dict(data.get("sections", {})),
            tick=data.get("tick", 0),
            timestamp=data.get("timestamp", time.time()),
        )


# =============================================================================
# State
# =============================================================================

@dataclass
class IntrospectionCrossSectionState:
    """内省断面横断的記述の内部状態。"""

    # スナップショットのスライディングウィンドウ: 時系列順に蓄積
    snapshot_window: list[CrossSectionSnapshot] = field(default_factory=list)

    # 直前スナップショット: 1つ前の処理実行時に構成されたスナップショット
    previous_snapshot: Optional[CrossSectionSnapshot] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_window": [s.to_dict() for s in self.snapshot_window],
            "previous_snapshot": (
                self.previous_snapshot.to_dict()
                if self.previous_snapshot is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntrospectionCrossSectionState":
        snapshot_window = [
            CrossSectionSnapshot.from_dict(s)
            for s in data.get("snapshot_window", [])
        ]
        prev_data = data.get("previous_snapshot")
        previous_snapshot = (
            CrossSectionSnapshot.from_dict(prev_data)
            if prev_data is not None
            else None
        )
        return cls(
            snapshot_window=snapshot_window,
            previous_snapshot=previous_snapshot,
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IntrospectionCrossSectionConfig:
    """設定。"""

    # スライディングウィンドウの上限（安全弁4: ウィンドウサイズの制限）
    # 25件: 内省の時間的奥行きを確保しつつ、上限による自然な押し出しを維持
    max_snapshots: int = 25

    # 各断面の要約テキストの長さ上限（文字数）
    max_summary_length: int = 200

    # enrichment出力のサイズ上限（文字数）
    max_enrichment_length: int = 2000

    # enrichmentに含めるスナップショット件数の上限
    # enrichmentには直近10件のみを出力し、残りはウィンドウ内に内部参照用として保持
    max_enrichment_snapshots: int = 10


# =============================================================================
# Summary Extraction Helpers
# =============================================================================

def _truncate(text: str, max_length: int) -> str:
    """テキストを長さ上限で切り詰める。"""
    if len(text) <= max_length:
        return text
    return text[:max_length]


def _sanitize_summary(text: str) -> str:
    """要約テキストから評価的語彙を除去する。

    評価的語彙が含まれている場合、その語を「[...]」に置換する。
    これは安全弁として機能し、上流モジュールの出力に評価的語彙が
    含まれていた場合でも本機能の出力には含まれないことを保証する。
    """
    result = text
    for word in _EVALUATIVE_WORDS:
        if word in result:
            result = result.replace(word, "[...]")
    return result


def _extract_self_model_summary(output: Any, max_length: int) -> str:
    """自己モデル出力から要約的記述を抽出する。

    他の断面を参照せず、この断面のみで独立に要約を生成する。
    評価的語彙を使用しない。
    """
    if output is None:
        return ABSENT_MARKER

    if isinstance(output, str):
        if not output.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(output.strip(), max_length))

    if isinstance(output, dict):
        # 辞書形式の場合、summaryまたはdescriptionキーを優先参照
        for key in ("summary", "description", "text"):
            val = output.get(key)
            if val and isinstance(val, str) and val.strip():
                return _sanitize_summary(_truncate(val.strip(), max_length))
        # キーが見つからない場合、辞書の内容を列挙（空でない値のみ）
        parts = []
        for k, v in output.items():
            v_str = str(v).strip() if v is not None else ""
            if v_str:
                parts.append(f"{k}: {v_str}")
        text = ", ".join(parts)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    # その他の型: 文字列化
    text = str(output).strip()
    if not text:
        return ABSENT_MARKER
    return _sanitize_summary(_truncate(text, max_length))


def _extract_temporal_self_diff_summary(output: Any, max_length: int) -> str:
    """時間的自己差分出力から要約的記述を抽出する。"""
    if output is None:
        return ABSENT_MARKER

    if isinstance(output, str):
        if not output.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(output.strip(), max_length))

    if isinstance(output, dict):
        for key in ("summary", "diff_summary", "description", "text"):
            val = output.get(key)
            if val and isinstance(val, str) and val.strip():
                return _sanitize_summary(_truncate(val.strip(), max_length))
        parts = []
        for k, v in output.items():
            v_str = str(v).strip() if v is not None else ""
            if v_str:
                parts.append(f"{k}: {v_str}")
        text = ", ".join(parts)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    text = str(output).strip()
    if not text:
        return ABSENT_MARKER
    return _sanitize_summary(_truncate(text, max_length))


def _extract_identity_coherence_summary(output: Any, max_length: int) -> str:
    """同一性揺らぎ出力から要約的記述を抽出する。"""
    if output is None:
        return ABSENT_MARKER

    if isinstance(output, str):
        if not output.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(output.strip(), max_length))

    if isinstance(output, dict):
        for key in ("summary", "stage", "direction", "description", "text"):
            val = output.get(key)
            if val and isinstance(val, str) and val.strip():
                return _sanitize_summary(_truncate(val.strip(), max_length))
        parts = []
        for k, v in output.items():
            v_str = str(v).strip() if v is not None else ""
            if v_str:
                parts.append(f"{k}: {v_str}")
        text = ", ".join(parts)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    text = str(output).strip()
    if not text:
        return ABSENT_MARKER
    return _sanitize_summary(_truncate(text, max_length))


def _extract_self_narrative_summary(output: Any, max_length: int) -> str:
    """自己物語出力から要約的記述を抽出する。"""
    if output is None:
        return ABSENT_MARKER

    if isinstance(output, str):
        if not output.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(output.strip(), max_length))

    if isinstance(output, dict):
        for key in ("summary", "fragment", "text", "narrative"):
            val = output.get(key)
            if val and isinstance(val, str) and val.strip():
                return _sanitize_summary(_truncate(val.strip(), max_length))
        parts = []
        for k, v in output.items():
            v_str = str(v).strip() if v is not None else ""
            if v_str:
                parts.append(f"{k}: {v_str}")
        text = ", ".join(parts)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    text = str(output).strip()
    if not text:
        return ABSENT_MARKER
    return _sanitize_summary(_truncate(text, max_length))


def _extract_introspection_consumption_summary(output: Any, max_length: int) -> str:
    """内省消費出力から要約的記述を抽出する。"""
    if output is None:
        return ABSENT_MARKER

    if isinstance(output, str):
        if not output.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(output.strip(), max_length))

    if isinstance(output, dict):
        for key in ("summary", "consumed", "description", "text"):
            val = output.get(key)
            if val and isinstance(val, str) and val.strip():
                return _sanitize_summary(_truncate(val.strip(), max_length))
        parts = []
        for k, v in output.items():
            v_str = str(v).strip() if v is not None else ""
            if v_str:
                parts.append(f"{k}: {v_str}")
        text = ", ".join(parts)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    if isinstance(output, list):
        # 消費断片リストの場合
        if not output:
            return ABSENT_MARKER
        items = []
        for item in output:
            if isinstance(item, str):
                items.append(item.strip())
            elif isinstance(item, dict):
                for key in ("summary", "text", "fragment"):
                    val = item.get(key)
                    if val and isinstance(val, str) and val.strip():
                        items.append(val.strip())
                        break
                else:
                    items.append(str(item))
            else:
                items.append(str(item))
        text = "; ".join(items)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    text = str(output).strip()
    if not text:
        return ABSENT_MARKER
    return _sanitize_summary(_truncate(text, max_length))


def _extract_meta_emotion_summary(output: Any, max_length: int) -> str:
    """メタ感情認知出力から要約的記述を抽出する。"""
    if output is None:
        return ABSENT_MARKER

    if isinstance(output, str):
        if not output.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(output.strip(), max_length))

    if isinstance(output, dict):
        for key in ("summary", "pattern_candidates", "description", "text"):
            val = output.get(key)
            if val and isinstance(val, str) and val.strip():
                return _sanitize_summary(_truncate(val.strip(), max_length))
        parts = []
        for k, v in output.items():
            v_str = str(v).strip() if v is not None else ""
            if v_str:
                parts.append(f"{k}: {v_str}")
        text = ", ".join(parts)
        if not text.strip():
            return ABSENT_MARKER
        return _sanitize_summary(_truncate(text.strip(), max_length))

    text = str(output).strip()
    if not text:
        return ABSENT_MARKER
    return _sanitize_summary(_truncate(text, max_length))


# 断面識別名と抽出関数のマッピング（定義順に並列）
_SECTION_EXTRACTORS = {
    SECTION_SELF_MODEL: _extract_self_model_summary,
    SECTION_TEMPORAL_SELF_DIFFERENCE: _extract_temporal_self_diff_summary,
    SECTION_IDENTITY_COHERENCE: _extract_identity_coherence_summary,
    SECTION_SELF_NARRATIVE: _extract_self_narrative_summary,
    SECTION_INTROSPECTION_CONSUMPTION: _extract_introspection_consumption_summary,
    SECTION_META_EMOTION_COGNITION: _extract_meta_emotion_summary,
}


# =============================================================================
# Processor (3-stage pipeline)
# =============================================================================

class IntrospectionCrossSectionProcessor:
    """内省断面間の横断的記述プロセッサ。

    3段パイプライン:
    1. 断面値の収集 -- 6モジュールの最新出力の要約的記述を収集
    2. スナップショットの構成と蓄積 -- 6断面を1組のスナップショットとして構成、
       スライディングウィンドウに蓄積（FIFO）
    3. 参照情報としての受渡準備 -- enrichmentテキスト生成 + READ-ONLYアクセサ

    すべての処理は記述的な並置行為であり、能動的な判断・評価・統合を含まない。
    出力は参照情報としてのみ流れる。
    """

    def __init__(self, config: Optional[IntrospectionCrossSectionConfig] = None):
        self._config = config or IntrospectionCrossSectionConfig()
        self._state = IntrospectionCrossSectionState()

    @property
    def state(self) -> IntrospectionCrossSectionState:
        return self._state

    @state.setter
    def state(self, value: IntrospectionCrossSectionState) -> None:
        self._state = value

    # ─── Stage 1: 断面値の収集 ─────────────────────────────────────

    def _collect_section_summaries(
        self,
        module_outputs: dict[str, Any],
    ) -> dict[str, str]:
        """6モジュールの出力から各断面の要約的記述を収集する。

        各断面の要約は独立に生成され、他の断面を参照して生成されない。
        評価的語彙を使用しない。
        モジュールの出力が不在の断面は「不在」として記録する。

        Args:
            module_outputs: 統合管理から渡された6モジュールの出力辞書
                key: 断面識別名, value: モジュール出力（型は各モジュールにより異なる）

        Returns:
            6断面の要約的記述の辞書
        """
        max_len = self._config.max_summary_length
        summaries: dict[str, str] = {}

        for section_name in SECTION_ORDER:
            extractor = _SECTION_EXTRACTORS.get(section_name)
            if extractor is None:
                summaries[section_name] = ABSENT_MARKER
                continue

            raw_output = module_outputs.get(section_name)
            summary = extractor(raw_output, max_len)
            summaries[section_name] = summary

        return summaries

    # ─── Stage 2: スナップショットの構成と蓄積 ─────────────────────

    def _construct_and_accumulate(
        self,
        summaries: dict[str, str],
        tick: int,
        timestamp: float,
    ) -> CrossSectionSnapshot:
        """6断面の要約を1組のスナップショットとして構成し、ウィンドウに蓄積する。

        スナップショット構成に際して以下を禁止する:
        - 断面間の比較・対照・関係記述
        - 前回スナップショットとの差分算出
        - 特定の断面の値に基づくフィルタリング・選別

        Args:
            summaries: Stage 1 で収集した6断面の要約的記述
            tick: 現在のティック番号
            timestamp: 現在のタイムスタンプ

        Returns:
            構成されたスナップショット
        """
        # 直前スナップショットを更新（現在のウィンドウ末尾を直前に移動）
        if self._state.snapshot_window:
            self._state.previous_snapshot = self._state.snapshot_window[-1]

        # 新しいスナップショットを構成
        snapshot = CrossSectionSnapshot(
            sections=dict(summaries),
            tick=tick,
            timestamp=timestamp,
        )

        # スライディングウィンドウに追加
        self._state.snapshot_window.append(snapshot)

        # 上限による押し出し（安全弁4: ウィンドウサイズの制限）
        self._apply_window_pushout()

        logger.debug(
            "Cross-section snapshot constructed: tick=%d, sections=%d, window=%d",
            tick,
            len(summaries),
            len(self._state.snapshot_window),
        )

        return snapshot

    def _apply_window_pushout(self) -> None:
        """ウィンドウの上限押し出し。最古のものから押し出す（FIFO）。

        これが唯一のデータ消失経路。特定のスナップショットを
        選択的に消去する処理は存在しない。
        """
        max_size = self._config.max_snapshots
        if len(self._state.snapshot_window) > max_size:
            pushout_count = len(self._state.snapshot_window) - max_size
            self._state.snapshot_window = self._state.snapshot_window[pushout_count:]

    # ─── Stage 3: 参照情報としての受渡準備 ─────────────────────────

    def get_enrichment_text(self) -> str:
        """enrichmentへの参照テキストを生成する。

        直近の限られた件数分のスナップショットについて、各断面の要約を
        等価に列挙する。列挙順序は断面の定義順に固定し、出力の度に変更しない。
        特定の断面を強調・選別しない。
        enrichmentの出力にはサイズ上限を設ける。

        Returns:
            enrichment用のテキスト
        """
        window = self._state.snapshot_window
        if not window:
            return "内省断面横断: 待機中"

        cfg = self._config
        # 直近の限られた件数分
        target_count = min(cfg.max_enrichment_snapshots, len(window))
        recent = window[-target_count:]

        parts: list[str] = []
        for snap in recent:
            section_parts: list[str] = []
            for section_name in SECTION_ORDER:
                label = SECTION_LABELS.get(section_name, section_name)
                value = snap.sections.get(section_name, ABSENT_MARKER)
                section_parts.append(f"{label}={value}")
            tick_label = f"t{snap.tick}"
            line = f"[{tick_label}] " + " / ".join(section_parts)
            parts.append(line)

        text = "\n".join(parts)

        # サイズ上限の適用
        if len(text) > cfg.max_enrichment_length:
            text = text[:cfg.max_enrichment_length]

        return text

    def get_snapshot_window(self) -> list[dict[str, Any]]:
        """スナップショットの蓄積リストをREAD-ONLYで返す。

        他モジュールがREAD-ONLYで参照可能な構造化データ。
        フィルタリング・選別・集約機能をアクセサに持たせない。
        全スナップショット・全断面を等価に返す。

        Returns:
            スナップショットのリスト（辞書形式のコピー）
        """
        return [s.to_dict() for s in self._state.snapshot_window]

    def get_previous_snapshot(self) -> Optional[dict[str, Any]]:
        """直前スナップショットをREAD-ONLYで返す。

        前後のスナップショットが並置可能な構造を維持するためのもの。
        ただし、前後の比較・差分算出処理は本機能に含まない。

        Returns:
            直前スナップショット（辞書形式のコピー）、またはNone
        """
        if self._state.previous_snapshot is None:
            return None
        return self._state.previous_snapshot.to_dict()

    def get_latest_snapshot(self) -> Optional[dict[str, Any]]:
        """最新のスナップショットをREAD-ONLYで返す。

        Returns:
            最新スナップショット（辞書形式のコピー）、またはNone
        """
        if not self._state.snapshot_window:
            return None
        return self._state.snapshot_window[-1].to_dict()

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。"""
        return {
            "window_size": len(self._state.snapshot_window),
            "has_previous_snapshot": self._state.previous_snapshot is not None,
            "max_snapshots": self._config.max_snapshots,
        }

    # ─── Main processing entry point ──────────────────────────────

    def process(
        self,
        module_outputs: dict[str, Any],
        tick: int,
        timestamp: Optional[float] = None,
    ) -> CrossSectionSnapshot:
        """3段パイプラインの一括実行。

        統合管理からの呼び出し時に、6モジュールの最新出力を受け取り、
        断面値収集→スナップショット構成・蓄積→参照情報準備の3段を実行する。

        Args:
            module_outputs: 統合管理から渡された6モジュールの出力辞書
                key: 断面識別名, value: モジュール出力
            tick: 現在のティック番号
            timestamp: 現在のタイムスタンプ（指定なしの場合は現在時刻）

        Returns:
            構成されたスナップショット
        """
        now = timestamp if timestamp is not None else time.time()

        # Stage 1: 断面値の収集
        summaries = self._collect_section_summaries(module_outputs)

        # Stage 2: スナップショットの構成と蓄積
        snapshot = self._construct_and_accumulate(summaries, tick, now)

        logger.debug(
            "Introspection cross-section processed: tick=%d, window=%d",
            tick,
            len(self._state.snapshot_window),
        )

        return snapshot

    # ─── Save / Load ──────────────────────────────────────────────

    def save(self) -> dict[str, Any]:
        """永続化用のデータを返す。"""
        return self._state.to_dict()

    def load(self, data: dict[str, Any]) -> None:
        """永続化データから状態を復元する。"""
        self._state = IntrospectionCrossSectionState.from_dict(data)
        logger.debug(
            "Introspection cross-section state loaded: window=%d",
            len(self._state.snapshot_window),
        )


# =============================================================================
# Factory
# =============================================================================

def create_introspection_cross_section(
    config: Optional[IntrospectionCrossSectionConfig] = None,
) -> IntrospectionCrossSectionProcessor:
    """IntrospectionCrossSectionProcessor のファクトリ関数。

    デフォルト設定のインスタンスを生成する。

    Args:
        config: カスタム設定（省略時はデフォルト）

    Returns:
        IntrospectionCrossSectionProcessor のインスタンス
    """
    return IntrospectionCrossSectionProcessor(config=config)
