"""
psyche/enrichment_compression.py - enrichment-to-prompt パイプライン効率化

enrichmentテキストの組み立て段階における圧縮ヘルパー関数群。
各モジュールのenrichment出力メソッドには一切手を加えず、
orchestrator側の組み立てロジックのみで圧縮を実現する。

設計書: design_enrichment_compression.md, design_startup_session_quality.md

処理は3段構成 + 起動品質ヘルパー:
  第1段: 項目別変動度の算出（二値判定のみ）
  第2段: 記述粒度の選択（全文記述 / 短縮形記述）
  第3段: フォーマット圧縮（セクションヘッダ簡潔化、空セクション短縮、フッター簡潔化）
  起動品質A: 空状態記述の統一（「(未蓄積)」への置換）
  起動品質B: セッション境界の鮮度注釈（経過時間段階値+経過ティック）

安全弁:
  1. キャッシュ不在時のフォールバック（全項目全文記述）
  2. 圧縮率の下限監視（ログ警告のみ、処理停止なし）
  3. 短縮形の固定性（「(安定)」固定文字列、状態非依存）
  4. セクション消失の防止（全項目短縮形でもセクション名は残る）
  5. 個別項目の圧縮無効化経路（除外リスト、静的定義）
  6. 空状態表記の単一性（「(未蓄積)」一種類のみ）
  7. 鮮度注釈の無条件消失（経過ティック数のみで判定）
  8. 初回起動への鮮度注釈非適用
  9. 永続化フィールド非追加
  10. enrichment以外への非露出
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


# =============================================================================
# 起動品質A: 空状態記述の統一
# 設計書: design_startup_session_quality.md §2.1, §3.1
# =============================================================================

# 空状態の統一表記（安全弁6: 一種類のみ、段階表記なし）
EMPTY_STATE_MARKER = "(未蓄積)"

# 既知の空状態パターン（各モジュールの実際の空状態出力を走査して収集）
_KNOWN_EMPTY_PATTERNS: frozenset[str] = frozenset({
    "",
    "(なし)",
    "(空)",
    "(蓄積前)",
})


def is_empty_state_text(text: str) -> bool:
    """enrichment項目テキストが空状態を示すかを判定する。

    判定基準（設計書 §3.1）:
    - テキストが空文字列
    - ホワイトスペースのみ
    - 既知の空表現パターンに一致

    Args:
        text: enrichment項目のテキスト

    Returns:
        True if the text represents an empty/unaccumulated state
    """
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in _KNOWN_EMPTY_PATTERNS:
        return True
    return False


def normalize_empty_state(label: str, text: str) -> str:
    """空状態テキストを統一表記に置換する。

    空状態と判定された項目に対して、統一的な短縮表記「(未蓄積)」を適用する。
    空状態でない項目には一切の変更を加えない。

    Args:
        label: 項目のラベル（例: 「感情連動」）
        text: enrichment項目のテキスト

    Returns:
        元テキスト（空状態でない場合）、または「{label}: (未蓄積)」
    """
    if is_empty_state_text(text):
        return f"{label}: {EMPTY_STATE_MARKER}"
    return text


def normalize_section_items(
    items: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """セクション内の全項目に空状態統一を適用する。

    各項目の(ラベル, テキスト)ペアに対してnormalize_empty_stateを適用し、
    空状態の場合に統一表記に置換する。

    Args:
        items: (ラベル, テキスト) のリスト

    Returns:
        正規化済みの (ラベル, テキスト) リスト
    """
    result: list[tuple[str, str]] = []
    for label, text in items:
        normalized_text = normalize_empty_state(label, text)
        result.append((label, normalized_text))
    return result


# =============================================================================
# 起動品質B: セッション境界の鮮度注釈
# 設計書: design_startup_session_quality.md §2.2, §3.2
# =============================================================================

# 過渡期ティック閾値
# スライディングウィンドウ型蓄積の最大ウィンドウサイズに準じる。
# 現在の最大: drive_variation_description, internal_contradiction_description = 50
FRESHNESS_TRANSITION_TICKS = 50


def classify_elapsed_time(elapsed_seconds: float) -> str:
    """セッション間経過時間を段階値で記述する。

    設計書 §3.2: 「数分前」「数時間前」「数日前」等の段階値。
    規範的表現（「復帰中」「不完全」等）は使用しない。

    Args:
        elapsed_seconds: 前セッションからの経過秒数

    Returns:
        段階値テキスト
    """
    if elapsed_seconds < 0:
        return "不明"
    if elapsed_seconds < 60:
        return "数秒前"
    if elapsed_seconds < 3600:
        minutes = int(elapsed_seconds / 60)
        if minutes < 5:
            return "数分前"
        elif minutes < 30:
            return f"約{minutes}分前"
        else:
            return "約30分以上前"
    if elapsed_seconds < 86400:
        hours = int(elapsed_seconds / 3600)
        if hours == 1:
            return "約1時間前"
        else:
            return f"約{hours}時間前"
    days = int(elapsed_seconds / 86400)
    if days == 1:
        return "約1日前"
    elif days < 7:
        return f"約{days}日前"
    else:
        return f"{days}日以上前"


def build_freshness_annotation(
    session_gap_seconds: Optional[float],
    session_resume_tick: Optional[int],
    current_tick: int,
) -> Optional[str]:
    """セッション再開時の鮮度注釈を生成する。

    設計書 §3.2:
    - 初回起動時（session_gap_seconds=None）は鮮度注釈を返さない（安全弁8）
    - セッション再開後、一定ティック数が経過すると鮮度注釈は消失する（安全弁7）
    - 鮮度注釈は事実の記述であり、状態の良し悪しを含意しない

    Args:
        session_gap_seconds: セッション間経過時間（秒）。初回起動時はNone
        session_resume_tick: セッション再開時のティック数。初回起動時はNone
        current_tick: 現在のティック数

    Returns:
        鮮度注釈テキスト（1行）、または None（注釈不要の場合）
    """
    # 安全弁8: 初回起動への鮮度注釈非適用
    if session_gap_seconds is None or session_resume_tick is None:
        return None

    # 安全弁7: 鮮度注釈の無条件消失（ティック数の単純比較）
    elapsed_ticks = current_tick - session_resume_tick
    if elapsed_ticks >= FRESHNESS_TRANSITION_TICKS:
        return None

    # 鮮度注釈の文面構成
    time_label = classify_elapsed_time(session_gap_seconds)
    return (
        f"[セッション再開: 前回終了から{time_label}, "
        f"再開後{elapsed_ticks}ティック経過]"
    )


def prepend_freshness_annotation(
    enrichment_text: str,
    session_gap_seconds: Optional[float],
    session_resume_tick: Optional[int],
    current_tick: int,
) -> str:
    """enrichmentテキストの冒頭に鮮度注釈を付与する。

    設計書 §4: 鮮度注釈はenrichmentの最初のセクションの前に配置し、
    セクション構造の内部には介入しない。

    Args:
        enrichment_text: 圧縮済みenrichmentテキスト
        session_gap_seconds: セッション間経過時間（秒）
        session_resume_tick: セッション再開時のティック数
        current_tick: 現在のティック数

    Returns:
        鮮度注釈付きenrichmentテキスト（注釈不要の場合は元テキストそのまま）
    """
    annotation = build_freshness_annotation(
        session_gap_seconds, session_resume_tick, current_tick,
    )
    if annotation is None:
        return enrichment_text
    return f"{annotation}\n\n{enrichment_text}"
