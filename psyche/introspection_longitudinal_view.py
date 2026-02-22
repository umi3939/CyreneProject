"""
psyche/introspection_longitudinal_view.py - 内省の時間的縦断参照

横断的記述（introspection_cross_section）の蓄積済みスナップショットを唯一の
入力源として、呼び出しの都度「横断→縦断」の視点変換を行う薄い変換層。

設計原則 (design_introspection_longitudinal_view.md 準拠):
- 独自の永続的内部状態を保持しない
- 推移の傾向・方向性・パターンを算出しない
- 時点間の差分・変化量を算出しない
- 特定の断面を選択的に優先・強調しない
- 特定の時点を選択的にフィルタリング・強調しない
- 評価的語彙を使用しない
- パターン抽出・統計処理を行わない
- 横断的記述の内部状態・蓄積データ・処理パラメータを変更しない（READ-ONLY参照のみ）
- 感情パイプラインのパラメータを変更しない
- いかなるモジュールの内部状態・処理パラメータに書き込まない
- 自ら独立したデータ蓄積を行わない

3段パイプライン:
1. スナップショットウィンドウの取得
2. 断面別の時系列並置への変換
3. 参照情報としての受渡準備

安全弁:
1. パターン抽出の禁止
2. 全断面の等価性
3. 全時点の等価性
4. 独自の状態蓄積の禁止
5. 判断系・行動系・感情パイプライン・内省系モジュール・横断的記述への書き込み経路の遮断

経路遮断:
1. 縦断参照 → 横断的記述の処理
2. 縦断参照 → 各内省系モジュールの入力
3. 縦断参照 → ポリシー選択パイプライン
4. 縦断参照 → 感情パイプライン
5. 縦断参照 → 記憶忘却/固定化パラメータ
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .introspection_cross_section import (
    SECTION_ORDER,
    SECTION_LABELS,
    ABSENT_MARKER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LongitudinalViewConfig:
    """縦断参照の設定。

    起動時の初期化で十分であり、永続化対象ではない。
    """

    # enrichmentに含める直近の時点数の上限
    max_enrichment_timepoints: int = 10

    # enrichment出力のサイズ上限（文字数）
    max_enrichment_length: int = 2000


# =============================================================================
# Data Structures (処理中にのみ一時的に保持)
# =============================================================================

@dataclass
class TimePointEntry:
    """1時点における1断面の値。

    値+ティック番号+タイムスタンプの組。
    重み・スコア・優先度・重要度などの評価的属性を持たない。
    """
    value: str
    tick: int
    timestamp: float


@dataclass
class SectionTimeline:
    """1断面の時系列並置。

    時間順に並んだ各時点のエントリのリスト。
    パターン抽出・統計処理・傾向化を行わない。
    """
    section_name: str
    entries: list[TimePointEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_name": self.section_name,
            "entries": [
                {
                    "value": e.value,
                    "tick": e.tick,
                    "timestamp": e.timestamp,
                }
                for e in self.entries
            ],
        }


@dataclass
class LongitudinalView:
    """全断面の縦断データ。

    断面識別名をキーとし、SectionTimelineを値とする辞書。
    呼び出し元への返却のために一時的に構成されるが、
    呼び出し間で保持されない。
    全断面は等価である。
    """
    timelines: dict[str, SectionTimeline] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timelines": {
                name: tl.to_dict()
                for name, tl in self.timelines.items()
            },
        }


# =============================================================================
# Pipeline Functions (3-stage)
# =============================================================================

def _stage1_get_snapshots(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """第1段: スナップショットウィンドウの取得。

    横断的記述のREAD-ONLYアクセサを通じて取得されたスナップショットの
    辞書形式のリストをそのまま受け取る。

    ウィンドウが空の場合（横断的記述がまだ一度も処理されていない場合）は、
    空のリストを返す。これを異常やエラーとして扱わない。

    Args:
        snapshots: 横断的記述のget_snapshot_window()が返したリスト

    Returns:
        スナップショットの辞書リスト（そのまま通過）
    """
    if not snapshots:
        return []
    return snapshots


def _stage2_transform_to_longitudinal(
    snapshots: list[dict[str, Any]],
) -> dict[str, SectionTimeline]:
    """第2段: 断面別の時系列並置への変換。

    取得したスナップショットウィンドウに対して、断面識別名ごとに、
    全時点の値を時間順に取り出して並置する。

    変換に際して以下を禁止する:
    - 時点間の差分・変化量の算出
    - 値に基づくフィルタリング・選別
    - 断面間の比較・対照・統合
    - 特定の断面の値に基づく他の断面のフィルタリング
    - 並び替え（時間順以外の順序変更）

    不在の断面値は縦断データにおいてもそのまま「不在」として保持する。
    不在を除外・補間しない。

    Args:
        snapshots: 第1段で取得したスナップショットリスト（時系列順）

    Returns:
        断面識別名をキーとしたSectionTimelineの辞書
    """
    timelines: dict[str, SectionTimeline] = {}

    # 全断面について等価にタイムラインを初期化（定義順に固定）
    for section_name in SECTION_ORDER:
        timelines[section_name] = SectionTimeline(section_name=section_name)

    # 各スナップショットから断面値を時間順に取り出して並置
    for snap in snapshots:
        tick = snap.get("tick", 0)
        timestamp = snap.get("timestamp", 0.0)
        sections = snap.get("sections", {})

        for section_name in SECTION_ORDER:
            # 不在の断面値はそのまま ABSENT_MARKER として保持
            value = sections.get(section_name, ABSENT_MARKER)
            entry = TimePointEntry(
                value=value,
                tick=tick,
                timestamp=timestamp,
            )
            timelines[section_name].entries.append(entry)

    return timelines


def _stage3_prepare_handoff(
    timelines: dict[str, SectionTimeline],
) -> LongitudinalView:
    """第3段: 参照情報としての受渡準備。

    変換された断面別の縦断データを、他のモジュールが参照可能な形で整えて提供する。

    Args:
        timelines: 第2段で変換された断面別タイムライン

    Returns:
        LongitudinalView（全断面の縦断データを保持）
    """
    return LongitudinalView(timelines=timelines)


# =============================================================================
# Enrichment Generation
# =============================================================================

def _generate_enrichment_text(
    view: LongitudinalView,
    config: LongitudinalViewConfig,
) -> str:
    """enrichmentへの参照テキストを生成する。

    全断面について、直近の限られた件数分の縦断データを等価に列挙する。
    各断面の列挙は断面の定義順（横断的記述の断面定義順）に固定し、
    出力の度に変更しない。特定の断面を強調・選別しない。
    enrichmentの出力にはサイズ上限を設ける。

    横断的記述のenrichment行と重複しないよう、
    本機能のenrichmentは「断面別の時系列並置」という視点に限定する。

    Args:
        view: 縦断データ
        config: 設定

    Returns:
        enrichment用のテキスト
    """
    if not view.timelines:
        return "内省縦断: 待機中"

    # 全断面のうち、いずれかにエントリが存在するか確認
    has_any_entries = any(
        len(tl.entries) > 0
        for tl in view.timelines.values()
    )
    if not has_any_entries:
        return "内省縦断: 待機中"

    max_tp = config.max_enrichment_timepoints
    parts: list[str] = []

    # 断面の定義順に固定して列挙（全断面等価）
    for section_name in SECTION_ORDER:
        tl = view.timelines.get(section_name)
        if tl is None:
            continue

        label = SECTION_LABELS.get(section_name, section_name)
        entries = tl.entries

        # 直近の限られた件数分
        target_entries = entries[-max_tp:] if len(entries) > max_tp else entries

        # 各時点の値を等価に列挙（時間順）
        value_parts: list[str] = []
        for entry in target_entries:
            value_parts.append(f"t{entry.tick}:{entry.value}")

        line = f"{label}: " + " / ".join(value_parts)
        parts.append(line)

    text = "\n".join(parts)

    # サイズ上限の適用
    if len(text) > config.max_enrichment_length:
        text = text[:config.max_enrichment_length]

    return text


# =============================================================================
# Public API (stateless — 呼び出しの都度変換して返す)
# =============================================================================

def get_enrichment_data(
    snapshots: list[dict[str, Any]],
    config: Optional[LongitudinalViewConfig] = None,
) -> dict[str, Any]:
    """enrichment用のデータを生成する。

    横断的記述のスナップショットウィンドウを入力として受け取り、
    断面別の時系列並置に変換してenrichment用テキストを返す。

    本関数は独自の状態を保持しない。呼び出しの都度、
    入力スナップショットから変換結果を生成して返す。

    Args:
        snapshots: 横断的記述のget_snapshot_window()が返したリスト
        config: 設定（省略時はデフォルト）

    Returns:
        enrichmentデータの辞書:
            - "summary_text": enrichment用テキスト
            - "section_count": 断面数
            - "timepoint_count": 時点数
    """
    cfg = config or LongitudinalViewConfig()

    # 3段パイプライン実行
    window = _stage1_get_snapshots(snapshots)
    if not window:
        return {
            "summary_text": "内省縦断: 待機中",
            "section_count": 0,
            "timepoint_count": 0,
        }

    timelines = _stage2_transform_to_longitudinal(window)
    view = _stage3_prepare_handoff(timelines)

    text = _generate_enrichment_text(view, cfg)

    return {
        "summary_text": text,
        "section_count": len(view.timelines),
        "timepoint_count": len(window),
    }


def get_longitudinal_view(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """断面別の縦断データをREAD-ONLYで返す。

    横断的記述のスナップショットウィンドウを入力として受け取り、
    断面別の時系列並置に変換して辞書形式で返す。

    断面を指定しない場合は全断面の全時点データを等価に返す。
    取得処理自体にフィルタリング・選別・集約機能を持たない。

    本関数は独自の状態を保持しない。呼び出しの都度、
    入力スナップショットから変換結果を生成して返す。

    Args:
        snapshots: 横断的記述のget_snapshot_window()が返したリスト

    Returns:
        LongitudinalViewの辞書形式
    """
    window = _stage1_get_snapshots(snapshots)
    if not window:
        return LongitudinalView().to_dict()

    timelines = _stage2_transform_to_longitudinal(window)
    view = _stage3_prepare_handoff(timelines)
    return view.to_dict()


def get_section_timeline(
    snapshots: list[dict[str, Any]],
    section_name: str,
) -> dict[str, Any]:
    """断面指定による個別取得。

    指定された断面の全時点データを等価に返す。
    取得処理自体にフィルタリング・選別・集約機能を持たない。

    どの断面を指定しても、同一の処理が等価に適用される。

    本関数は独自の状態を保持しない。呼び出しの都度、
    入力スナップショットから変換結果を生成して返す。

    Args:
        snapshots: 横断的記述のget_snapshot_window()が返したリスト
        section_name: 断面識別名

    Returns:
        SectionTimelineの辞書形式。指定された断面が存在しない場合は空のタイムライン。
    """
    window = _stage1_get_snapshots(snapshots)
    if not window:
        return SectionTimeline(section_name=section_name).to_dict()

    timelines = _stage2_transform_to_longitudinal(window)
    tl = timelines.get(section_name)
    if tl is None:
        # 横断的記述の断面定義に存在しない断面名の場合、空のタイムラインを返す
        return SectionTimeline(section_name=section_name).to_dict()

    return tl.to_dict()


# =============================================================================
# Factory (互換性のためにインスタンス形式も提供)
# =============================================================================

class IntrospectionLongitudinalViewProcessor:
    """内省の時間的縦断参照プロセッサ。

    独自の永続的内部状態を保持しない薄い変換層。
    横断的記述のスナップショットウィンドウを唯一の入力源として、
    呼び出しの都度「横断→縦断」の視点変換を行う。

    3段パイプライン:
    1. スナップショットウィンドウの取得
    2. 断面別の時系列並置への変換
    3. 参照情報としての受渡準備

    すべての処理は視点変換的な並置行為であり、
    能動的な判断・評価・分析を含まない。
    出力は参照情報としてのみ流れる。
    """

    def __init__(self, config: Optional[LongitudinalViewConfig] = None):
        self._config = config or LongitudinalViewConfig()

    def process(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """3段パイプラインの一括実行。

        横断的記述のスナップショットウィンドウを入力として受け取り、
        縦断的な視点変換を行って結果を返す。

        独自の状態を保持しない。変換結果は呼び出し元への返却のために
        一時的に構成されるが、呼び出し間で保持されない。

        Args:
            snapshots: 横断的記述のget_snapshot_window()が返したリスト

        Returns:
            LongitudinalViewの辞書形式
        """
        return get_longitudinal_view(snapshots)

    def get_enrichment_data(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """enrichment用のデータを生成する。

        Args:
            snapshots: 横断的記述のget_snapshot_window()が返したリスト

        Returns:
            enrichmentデータの辞書
        """
        return get_enrichment_data(snapshots, self._config)

    def get_longitudinal_view(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """全断面の縦断データをREAD-ONLYで返す。

        Args:
            snapshots: 横断的記述のget_snapshot_window()が返したリスト

        Returns:
            LongitudinalViewの辞書形式
        """
        return get_longitudinal_view(snapshots)

    def get_section_timeline(
        self,
        snapshots: list[dict[str, Any]],
        section_name: str,
    ) -> dict[str, Any]:
        """断面指定による個別取得。

        Args:
            snapshots: 横断的記述のget_snapshot_window()が返したリスト
            section_name: 断面識別名

        Returns:
            SectionTimelineの辞書形式
        """
        return get_section_timeline(snapshots, section_name)

    def get_summary(self) -> dict[str, Any]:
        """モジュールサマリを返す。

        独自の永続的状態を持たないため、設定情報のみを返す。
        """
        return {
            "has_state": False,
            "max_enrichment_timepoints": self._config.max_enrichment_timepoints,
            "max_enrichment_length": self._config.max_enrichment_length,
        }


def create_introspection_longitudinal_view(
    config: Optional[LongitudinalViewConfig] = None,
) -> IntrospectionLongitudinalViewProcessor:
    """IntrospectionLongitudinalViewProcessor のファクトリ関数。

    Args:
        config: カスタム設定（省略時はデフォルト）

    Returns:
        IntrospectionLongitudinalViewProcessor のインスタンス
    """
    return IntrospectionLongitudinalViewProcessor(config=config)
