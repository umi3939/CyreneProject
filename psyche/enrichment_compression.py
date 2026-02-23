"""
psyche/enrichment_compression.py - enrichment-to-prompt パイプライン効率化

enrichmentテキストの組み立て段階における圧縮ヘルパー関数群。
各モジュールのenrichment出力メソッドには一切手を加えず、
orchestrator側の組み立てロジックのみで圧縮を実現する。

設計書: design_enrichment_compression.md

処理は3段構成:
  第1段: 項目別変動度の算出（二値判定のみ）
  第2段: 記述粒度の選択（全文記述 / 短縮形記述）
  第3段: フォーマット圧縮（セクションヘッダ簡潔化、空セクション短縮、フッター簡潔化）

安全弁:
  1. キャッシュ不在時のフォールバック（全項目全文記述）
  2. 圧縮率の下限監視（ログ警告のみ、処理停止なし）
  3. 短縮形の固定性（「(安定)」固定文字列、状態非依存）
  4. セクション消失の防止（全項目短縮形でもセクション名は残る）
  5. 個別項目の圧縮無効化経路（除外リスト、静的定義）
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── 安全弁5: 圧縮除外リスト（静的定義、実行時変更不可） ──
# 常に全文記述すべき項目のラベル一覧
# 基本感情・ムード等の常時参照される項目を除外リストに登録
ALWAYS_FULL_LABELS: frozenset[str] = frozenset({
    "感情",
    "ムード",
    "ドライブ",
    "支配的感情",
})

# ── セクションヘッダの簡潔化マッピング ──
SECTION_HEADER_MAP: dict[str, str] = {
    "【心理状態（内面）】": "[内面]",
    "【自己認識】": "[自己]",
    "【動機・目標】": "[動機]",
    "【記憶・内省】": "[記憶]",
    "【判断傾向】": "[判断]",
}

# ── フッター簡潔化 ──
ORIGINAL_FOOTER = (
    "この内面状態を自然に反映した反応をしてください。"
    "機械的に読み上げないこと。"
)
COMPRESSED_FOOTER = "内面を自然に反映。機械的読み上げ禁止。"

# ── 短縮形の固定接尾辞（安全弁3: 固定文字列、状態非依存） ──
STABLE_SUFFIX = "(安定)"


def detect_item_changed(
    label: str,
    current_text: str,
    prev_cache: dict[str, str],
) -> bool:
    """第1段: 項目別変動度の算出（二値判定）。

    前回テキストと今回テキストの差異を複数の断面で判定する:
    - テキスト長の変化
    - テキスト内容の変化（同一か否か）

    いずれか1つでも変化があれば True（変動あり）、
    全て無変化であれば False（変動なし）。

    キャッシュに前回テキストが存在しない場合は True（変動あり扱い）。
    除外リストに含まれる項目は常に True。

    Args:
        label: 項目の接頭辞ラベル（「感情連動」等）
        current_text: 今回のenrichment項目テキスト
        prev_cache: 前回enrichmentテキストキャッシュ（ラベル→テキスト）

    Returns:
        True=変動あり（全文記述）、False=変動なし（短縮形記述）
    """
    # 安全弁5: 除外リストの項目は常に全文記述
    if label in ALWAYS_FULL_LABELS:
        return True

    # 安全弁1: キャッシュ不在時は変動あり扱い
    if label not in prev_cache:
        return True

    prev_text = prev_cache[label]

    # 断面1: テキスト長の変化
    if len(current_text) != len(prev_text):
        return True

    # 断面2: テキスト内容の変化（同一か否か）
    if current_text != prev_text:
        return True

    return False


def apply_item_granularity(
    label: str,
    current_text: str,
    changed: bool,
) -> str:
    """第2段: 記述粒度の選択。

    変動ありの場合: モジュール出力テキストをそのまま使用（全文記述）。
    変動なしの場合: 短縮形に機械的に置換（ラベル + 「(安定)」）。

    Args:
        label: 項目の接頭辞ラベル
        current_text: 今回のenrichment項目テキスト
        changed: 第1段の変動判定結果（True=変動あり）

    Returns:
        全文テキストまたは短縮形テキスト
    """
    if changed:
        return current_text
    # 短縮形: ラベルに「(安定)」を付与した固定形式
    return f"{label}: {STABLE_SUFFIX}"


def compress_section(
    header: str,
    item_lines: list[str],
) -> str:
    """第3段: セクション単位のフォーマット圧縮。

    - セクションヘッダを簡潔化
    - セクション内の全項目が短縮形（「(安定)」含む）の場合、
      セクション全体を「[セクション名: 安定]」の1行に圧縮
    - セクション消失の防止（安全弁4）

    Args:
        header: 元のセクションヘッダ文字列（例: 「【心理状態（内面）】」）
        item_lines: セクション内の各項目テキスト行のリスト

    Returns:
        圧縮済みセクションテキスト
    """
    # ヘッダ簡潔化
    short_header = SECTION_HEADER_MAP.get(header, header)

    if not item_lines:
        # 安全弁4: セクション消失の防止
        return f"{short_header} {STABLE_SUFFIX}"

    # 全項目が短縮形かチェック
    all_stable = all(STABLE_SUFFIX in line for line in item_lines)

    if all_stable:
        # セクション全体を1行に圧縮
        return f"{short_header} {STABLE_SUFFIX}"

    # ヘッダ + 各項目行を結合
    return "\n".join([short_header] + item_lines)


def compress_footer(footer: str) -> str:
    """フッターの簡潔化。

    Args:
        footer: 元のフッターテキスト

    Returns:
        簡潔化されたフッターテキスト
    """
    if footer == ORIGINAL_FOOTER:
        return COMPRESSED_FOOTER
    return footer


def compute_compression_ratio(
    original_text: str,
    compressed_text: str,
) -> float:
    """圧縮率を算出する（圧縮後文字数 / 圧縮前文字数）。

    ログ出力用のみ。enrichmentテキストやモジュール内部状態には含めない。
    圧縮前テキストが空の場合は 1.0 を返す。

    Args:
        original_text: 圧縮前の全文展開テキスト
        compressed_text: 圧縮後のテキスト

    Returns:
        圧縮率（0.0〜1.0+）
    """
    if len(original_text) == 0:
        return 1.0
    return len(compressed_text) / len(original_text)


# ── 安全弁2: 圧縮率の下限監視閾値 ──
COMPRESSION_RATIO_WARNING_THRESHOLD = 0.3


def log_compression_ratio(ratio: float) -> None:
    """圧縮率をログに出力する。

    極端に低い場合（閾値未満）は警告を出力する。
    ただし圧縮処理自体を停止・変更はしない。

    Args:
        ratio: 圧縮率（compute_compression_ratio の戻り値）
    """
    if ratio < COMPRESSION_RATIO_WARNING_THRESHOLD:
        logger.warning(
            "Enrichment compression ratio %.2f is below threshold %.2f",
            ratio,
            COMPRESSION_RATIO_WARNING_THRESHOLD,
        )
    else:
        logger.debug("Enrichment compression ratio: %.2f", ratio)


def build_compressed_enrichment(
    sections_data: list[dict],
    prev_cache: dict[str, str],
    footer: str,
) -> tuple[str, dict[str, str], float]:
    """enrichmentテキスト全体の圧縮パイプラインを実行する。

    第1段（変動度算出）→ 第2段（粒度選択）→ 第3段（フォーマット圧縮）
    を順に適用し、圧縮済みテキストと更新キャッシュを返す。

    Args:
        sections_data: セクション定義のリスト。各要素は:
            {
                "header": str,          # セクションヘッダ（例: 「【心理状態（内面）】」）
                "items": list[tuple[str, str]],  # (ラベル, テキスト) のリスト
            }
        prev_cache: 前回enrichmentテキストキャッシュ（ラベル→テキスト）
        footer: フッターテキスト

    Returns:
        tuple of:
            - compressed_text: 圧縮済みenrichmentテキスト
            - new_cache: 今回のテキストで更新されたキャッシュ
            - compression_ratio: 圧縮率
    """
    new_cache: dict[str, str] = {}
    compressed_sections: list[str] = []
    original_sections: list[str] = []

    for section in sections_data:
        header = section["header"]
        items = section["items"]

        if not items:
            # 項目なしのセクションはスキップ
            continue

        original_lines: list[str] = []
        compressed_lines: list[str] = []

        for label, text in items:
            # キャッシュ更新（全項目の今回テキストを保存）
            new_cache[label] = text

            # 元テキスト行（圧縮率計算用）
            original_lines.append(text)

            # 第1段: 変動度算出
            changed = detect_item_changed(label, text, prev_cache)

            # 第2段: 粒度選択
            compressed_line = apply_item_granularity(label, text, changed)
            compressed_lines.append(compressed_line)

        # 元テキスト（圧縮率計算用）
        original_section = "\n".join([header] + original_lines)
        original_sections.append(original_section)

        # 第3段: セクション圧縮
        compressed_section = compress_section(header, compressed_lines)
        compressed_sections.append(compressed_section)

    # フッター圧縮
    original_footer = footer
    compressed_footer_text = compress_footer(footer)

    # 全体テキスト組み立て
    original_text = "\n\n".join(original_sections + [original_footer])
    compressed_text = "\n\n".join(compressed_sections + [compressed_footer_text])

    # 圧縮率算出
    ratio = compute_compression_ratio(original_text, compressed_text)

    # 安全弁2: ログ出力（処理への介入なし）
    log_compression_ratio(ratio)

    return compressed_text, new_cache, ratio
