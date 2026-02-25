"""
psyche/save_load_warmup.py - Save/Load復帰時のpsyche状態ウォームアップ

design_save_load_warmup.md に基づく実装:
  永続化復元完了後に、蓄積的内部状態から再導出可能な中間キャッシュを
  充填する。独自の内部状態を保持せず、1回だけ実行して痕跡を残さない。

方式: キャッシュ再導出方式（空入力予備実行を採用しない）

安全弁:
  1. 再導出不可能なキャッシュ(分類B)は一切処理しない
  2. モジュール内部更新メソッドを呼び出さない(読み取り専用アクセサ・属性のみ)
  3. 再導出失敗時は対応キャッシュをNoneのまま維持(エラーとせずログのみ)
  4. 再導出対象の宣言は実行時に変更されない(静的定義)
  5. enrichment構造を変更しない
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── 再導出方法の種別 ────────────────────────────────────────────────

class DerivationType(str, Enum):
    """再導出方法の3種別。

    R: 読み取り直接代入 — モジュールの公開属性から読み取りキャッシュに代入
    A: アクセサ経由取得 — モジュールの公開アクセサを呼び出し結果をキャッシュに代入
    S: スキップ — 再導出不可能(分類B)。何も行わない
    """
    R = "read_direct"
    A = "accessor"
    S = "skip"


# ── 再導出対象の宣言 ────────────────────────────────────────────────

@dataclass(frozen=True)
class WarmupEntry:
    """再導出対象のエントリ定義。

    Attributes:
        cache_attr: オーケストレータ上の中間キャッシュ属性名
        module_attr: オーケストレータ上のモジュール属性名(種別R/Aの場合)
        derivation: 再導出方法の種別
        source_sub_attr: モジュール属性からのサブパス(種別Rの場合)
        accessor_name: 呼び出すアクセサメソッド名(種別Aの場合)
    """
    cache_attr: str
    module_attr: str
    derivation: DerivationType
    source_sub_attr: Optional[str] = None
    accessor_name: Optional[str] = None


# ── 静的宣言: 再導出対象リスト ───────────────────────────────────────
#
# 分類A: 蓄積的内部状態から再導出可能なキャッシュ
#   種別A = モジュールの公開アクセサ経由で取得(get_last_store/get_last_state等)
#
# 分類B: 再導出不可能(直前ティックの入力依存)
#   種別S = スキップ
#
# 注: 分類Aの対象は既に FIELD_DEFINITIONS でも永続化されている。
#     load_fields() で正常に復元された場合は再導出で同一値が上書きされるのみ。
#     load_fields() で復元に失敗した場合に、モジュールの蓄積状態から再導出する
#     安全弁として機能する。
#
# 注: _last_self_view, _last_diff_summary, _last_strain, _last_trace,
#     _last_coupling は FIELD_DEFINITIONS で直接永続化されているが、
#     対応するモジュールに公開アクセサが存在しないため本宣言に含めない。
#     これらはload_fields()のみで復元される。

WARMUP_ENTRIES: tuple[WarmupEntry, ...] = (
    # ── 分類A: アクセサ経由で再導出可能 ──────────────────────────────

    # 自己モデル系列 (種別A: アクセサ経由)
    WarmupEntry(
        cache_attr="_last_self_image",
        module_attr="_self_image_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_image",
    ),
    WarmupEntry(
        cache_attr="_last_coherence",
        module_attr="_coherence_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_state",
    ),
    WarmupEntry(
        cache_attr="_last_narrative",
        module_attr="_narrative_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_state",
    ),
    WarmupEntry(
        cache_attr="_last_consumption",
        module_attr="_consumption_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_store",
    ),
    WarmupEntry(
        cache_attr="_last_expectations",
        module_attr="_expectation_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_store",
    ),
    WarmupEntry(
        cache_attr="_last_motives",
        module_attr="_motivation_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_store",
    ),

    # 記憶系列 (種別A: アクセサ経由)
    WarmupEntry(
        cache_attr="_last_episodes",
        module_attr="_episodic_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_store",
    ),
    WarmupEntry(
        cache_attr="_last_bindings",
        module_attr="_binding_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_store",
    ),

    # 他者モデル (種別A: アクセサ経由)
    WarmupEntry(
        cache_attr="_last_other_model",
        module_attr="_other_model_sys",
        derivation=DerivationType.A,
        accessor_name="get_last_store",
    ),

    # ── 分類B: 再導出不可能(種別S: スキップ) ─────────────────────────
    # 直前ティックの入力に依存し、永続化フィールドから再計算できない。
    # 空の初期値を維持し、次のティック進行で自然に充填される。
    WarmupEntry(cache_attr="_last_percept", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_recalled_memories", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_feed_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_integration_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_text_handoff", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_activation_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_vo_validation", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_forgetting_fixation", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_action_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_selected_policy_label", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_selected_policy_axis", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_emotion_for_action_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_dialogue_learning", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_meta_emotion", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_contradiction_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_backdrop_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_drive_variation_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_cooccurrence_result", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_boundary_accumulation", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_decision_bias", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_tone_mod", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_sensitivity_bias", module_attr="", derivation=DerivationType.S),
    WarmupEntry(cache_attr="_last_has_silence", module_attr="", derivation=DerivationType.S),
)


def execute_warmup(orchestrator: Any) -> dict[str, str]:
    """永続化復元完了後のキャッシュ再導出を実行する。

    WARMUP_ENTRIES の静的宣言に従い、各エントリの再導出方法で
    中間キャッシュ変数を充填する。

    本関数は永続化復元の後処理として1回だけ呼び出される。
    完了後に痕跡を残さない(独自の内部状態を保持しない)。

    Args:
        orchestrator: PsycheOrchestrator インスタンス

    Returns:
        再導出結果の辞書 {cache_attr: status}
        status は "derived", "skipped", "failed", "empty_source" のいずれか
    """
    results: dict[str, str] = {}

    for entry in WARMUP_ENTRIES:
        if entry.derivation == DerivationType.S:
            # 種別S: スキップ — 何も行わない
            results[entry.cache_attr] = "skipped"
            continue

        if entry.derivation == DerivationType.R:
            # 種別R: 読み取り直接代入
            try:
                module = getattr(orchestrator, entry.module_attr, None)
                if module is None:
                    results[entry.cache_attr] = "failed"
                    logger.debug(
                        "Warmup: module %s not found for cache %s",
                        entry.module_attr, entry.cache_attr,
                    )
                    continue

                if entry.source_sub_attr:
                    source_value = getattr(module, entry.source_sub_attr, None)
                else:
                    source_value = module

                if source_value is None:
                    # モジュールの蓄積状態が空 — キャッシュもNoneのまま維持
                    results[entry.cache_attr] = "empty_source"
                    logger.debug(
                        "Warmup: source %s.%s is None for cache %s",
                        entry.module_attr, entry.source_sub_attr or "",
                        entry.cache_attr,
                    )
                    continue

                setattr(orchestrator, entry.cache_attr, source_value)
                results[entry.cache_attr] = "derived"

            except Exception as e:
                # 安全弁3: 再導出失敗時の無操作
                results[entry.cache_attr] = "failed"
                logger.debug(
                    "Warmup: failed to derive cache %s from %s: %s",
                    entry.cache_attr, entry.module_attr, e,
                )
                continue

        elif entry.derivation == DerivationType.A:
            # 種別A: アクセサ経由取得
            try:
                module = getattr(orchestrator, entry.module_attr, None)
                if module is None:
                    results[entry.cache_attr] = "failed"
                    logger.debug(
                        "Warmup: module %s not found for cache %s",
                        entry.module_attr, entry.cache_attr,
                    )
                    continue

                accessor = getattr(module, entry.accessor_name, None)
                if accessor is None or not callable(accessor):
                    results[entry.cache_attr] = "failed"
                    logger.debug(
                        "Warmup: accessor %s.%s not found for cache %s",
                        entry.module_attr, entry.accessor_name,
                        entry.cache_attr,
                    )
                    continue

                result_value = accessor()
                if result_value is None:
                    results[entry.cache_attr] = "empty_source"
                    logger.debug(
                        "Warmup: accessor %s.%s returned None for cache %s",
                        entry.module_attr, entry.accessor_name,
                        entry.cache_attr,
                    )
                    continue

                setattr(orchestrator, entry.cache_attr, result_value)
                results[entry.cache_attr] = "derived"

            except Exception as e:
                # 安全弁3: 再導出失敗時の無操作
                results[entry.cache_attr] = "failed"
                logger.debug(
                    "Warmup: failed to derive cache %s via accessor %s.%s: %s",
                    entry.cache_attr, entry.module_attr,
                    entry.accessor_name, e,
                )
                continue

    # ログ出力: 再導出結果の集計
    derived_count = sum(1 for v in results.values() if v == "derived")
    skipped_count = sum(1 for v in results.values() if v == "skipped")
    failed_count = sum(1 for v in results.values() if v == "failed")
    empty_count = sum(1 for v in results.values() if v == "empty_source")

    logger.info(
        "Warmup completed: derived=%d, skipped=%d, empty_source=%d, failed=%d",
        derived_count, skipped_count, empty_count, failed_count,
    )

    return results


def get_warmup_entries() -> tuple[WarmupEntry, ...]:
    """静的宣言のエントリリストを返す(テスト・検証用)。"""
    return WARMUP_ENTRIES


def get_classification_a_entries() -> list[WarmupEntry]:
    """分類A(再導出可能)のエントリのみを返す(テスト・検証用)。"""
    return [e for e in WARMUP_ENTRIES if e.derivation != DerivationType.S]


def get_classification_b_entries() -> list[WarmupEntry]:
    """分類B(再導出不可能、スキップ)のエントリのみを返す(テスト・検証用)。"""
    return [e for e in WARMUP_ENTRIES if e.derivation == DerivationType.S]
