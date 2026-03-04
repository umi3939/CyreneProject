"""
psyche/save_load_warmup.py - Save/Load復帰時のpsyche状態ウォームアップ + 整合性検証

design_save_load_warmup.md に基づく実装:
  永続化復元完了後に、蓄積的内部状態から再導出可能な中間キャッシュを
  充填する。独自の内部状態を保持せず、1回だけ実行して痕跡を残さない。

方式: キャッシュ再導出方式（空入力予備実行を採用しない）

安全弁(ウォームアップ):
  1. 再導出不可能なキャッシュ(分類B)は一切処理しない
  2. モジュール内部更新メソッドを呼び出さない(読み取り専用アクセサ・属性のみ)
  3. 再導出失敗時は対応キャッシュをNoneのまま維持(エラーとせずログのみ)
  4. 再導出対象の宣言は実行時に変更されない(静的定義)
  5. enrichment構造を変更しない

design_session_recovery_check.md に基づく拡張:
  復元完了 → ウォームアップ完了後に、復元されたフィールド間の数値的整合性を
  受動的に走査し、不整合を警告として記録する。

安全弁(整合性検証):
  1. 修復禁止: 不整合検出に基づく状態の修復・補正・正規化を一切行わない
  2. 進行阻止禁止: 不整合検出時もティック進行を阻止・遅延しない
  3. 規範的基準の排除: 数値的矛盾の検出に限定（「正常な状態範囲」を定義しない）
  4. 状態無書き込み: オーケストレータの属性に対する書き込みを行わない
  5. 1回実行・痕跡なし: 復元後の1回のみ実行し、固有状態を残さない
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


# ══════════════════════════════════════════════════════════════════════════════
# セッション復帰時の状態整合性検証
# design_session_recovery_check.md に基づく実装
# ══════════════════════════════════════════════════════════════════════════════


class CheckType(str, Enum):
    """整合性検証の種別。

    A: ティック番号の数値的矛盾検出
    B: 窓内レコードの時間的整合性
    C: 鮮度減衰の前提照合
    D: ウォームアップ結果の照合
    E: フィールド間の数値的前提照合
    """
    A = "tick_consistency"
    B = "window_temporal"
    C = "freshness_premise"
    D = "warmup_cross_check"
    E = "cross_field_premise"


@dataclass(frozen=True)
class ConsistencyCheckEntry:
    """整合性検証エントリの宣言。

    Attributes:
        check_type: 検証種別 (A/B/C/D/E)
        module_attr: オーケストレータ上のモジュール属性名
        state_sub_attr: モジュールの内部状態サブ属性名 (e.g. "state", "_state")
        records_field: 蓄積構造のフィールド名 (e.g. "elapsed_records", "snapshot_window")
        tick_field: レコード内のティック番号フィールド名 (e.g. "tick")
        window_size: 窓サイズの宣言値 (検証種別Bの場合)
        description: 人間可読な説明
        persistence_key: 永続化辞書内のキー名 (検証種別D/Eで使用)
    """
    check_type: CheckType
    module_attr: str = ""
    state_sub_attr: str = ""
    records_field: str = ""
    tick_field: str = "tick"
    window_size: int = 0
    description: str = ""
    persistence_key: str = ""


# ── 静的宣言: 整合性検証エントリテーブル ─────────────────────────────
#
# 検証種別A: ティック番号超過検出
#   蓄積構造内のレコードのティック番号が復元ティック番号を超えていないことを照合
#
# 検証種別B: 窓内レコードの単調非減少・逸脱検出
#   窓内のティック番号順序が単調非減少であることを照合
#
# 検証種別C: 鮮度減衰の前提照合
#   減衰基準のティック番号が復元ティック番号と整合しているかを照合
#
# 検証種別D: ウォームアップ結果の照合
#   ウォームアップの失敗と永続化フィールドの関係を照合
#
# 検証種別E: フィールド間の数値的前提照合
#   複数フィールドが共有する数値的前提を照合

CONSISTENCY_CHECK_ENTRIES: tuple[ConsistencyCheckEntry, ...] = (
    # ── 検証種別A: ティック番号超過検出 ──────────────────────────────
    # temporal_cognition: elapsed_records[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_temporal_cognition",
        state_sub_attr="state",
        records_field="elapsed_records",
        tick_field="tick",
        description="時間認知の経過記録ティック番号",
    ),
    # introspection_cross_section: snapshot_window[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_introspection_cross_section",
        state_sub_attr="state",
        records_field="snapshot_window",
        tick_field="tick",
        description="内省断面スナップショットのティック番号",
    ),
    # self_action_perception: records[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_self_action_recorder",
        state_sub_attr="state",
        records_field="records",
        tick_field="tick",
        description="自己行動知覚記録のティック番号",
    ),
    # intent_action_gap: records[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_intent_action_gap_recorder",
        state_sub_attr="state",
        records_field="records",
        tick_field="tick",
        description="意図行動乖離記録のティック番号",
    ),
    # emotional_backdrop: sliding_window[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_emotional_backdrop_processor",
        state_sub_attr="state",
        records_field="sliding_window",
        tick_field="tick",
        description="感情基調スライディングウィンドウのティック番号",
    ),
    # drive_variation: sliding_window[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_drive_variation_processor",
        state_sub_attr="state",
        records_field="sliding_window",
        tick_field="tick",
        description="駆動変動スライディングウィンドウのティック番号",
    ),
    # selection_attribution: records[].tick
    ConsistencyCheckEntry(
        check_type=CheckType.A,
        module_attr="_selection_attribution_recorder",
        state_sub_attr="state",
        records_field="records",
        tick_field="tick",
        description="選択帰属記録のティック番号",
    ),

    # ── 検証種別B: 窓内レコードの時間的整合性 ─────────────────────────
    # temporal_cognition: elapsed_records の単調非減少
    ConsistencyCheckEntry(
        check_type=CheckType.B,
        module_attr="_temporal_cognition",
        state_sub_attr="state",
        records_field="elapsed_records",
        tick_field="tick",
        window_size=100,
        description="時間認知の経過記録窓",
    ),
    # introspection_cross_section: snapshot_window の単調非減少
    ConsistencyCheckEntry(
        check_type=CheckType.B,
        module_attr="_introspection_cross_section",
        state_sub_attr="state",
        records_field="snapshot_window",
        tick_field="tick",
        window_size=25,
        description="内省断面スナップショット窓",
    ),
    # emotional_backdrop: sliding_window の単調非減少
    ConsistencyCheckEntry(
        check_type=CheckType.B,
        module_attr="_emotional_backdrop_processor",
        state_sub_attr="state",
        records_field="sliding_window",
        tick_field="tick",
        window_size=30,
        description="感情基調スライディングウィンドウ窓",
    ),
    # drive_variation: sliding_window の単調非減少
    ConsistencyCheckEntry(
        check_type=CheckType.B,
        module_attr="_drive_variation_processor",
        state_sub_attr="state",
        records_field="sliding_window",
        tick_field="tick",
        window_size=50,
        description="駆動変動スライディングウィンドウ窓",
    ),

    # ── 検証種別C: 鮮度減衰の前提照合 ─────────────────────────────────
    # temporal_cognition: pathway_last_used_tick の各値がティック番号以内
    ConsistencyCheckEntry(
        check_type=CheckType.C,
        module_attr="_temporal_cognition",
        state_sub_attr="state",
        records_field="pathway_last_used_tick",
        tick_field="",
        description="時間認知の入力経路別最終使用ティック",
    ),

    # ── 検証種別D: ウォームアップ結果の照合 ──────────────────────────────
    ConsistencyCheckEntry(
        check_type=CheckType.D,
        description="ウォームアップ再導出結果の照合",
    ),

    # ── 検証種別E: フィールド間の数値的前提照合 ─────────────────────────
    # version フィールドの一致確認
    ConsistencyCheckEntry(
        check_type=CheckType.E,
        description="永続化バージョン番号の整合性",
        persistence_key="version",
    ),
)


def _resolve_module_state(orchestrator: Any, entry: ConsistencyCheckEntry) -> Any:
    """エントリで指定されたモジュールの内部状態を読み取り専用で取得する。

    状態が取得できない場合は None を返す。
    """
    module = getattr(orchestrator, entry.module_attr, None)
    if module is None:
        return None
    if entry.state_sub_attr:
        state = getattr(module, entry.state_sub_attr, None)
        return state
    return module


def _get_records_list(state: Any, records_field: str) -> Optional[list]:
    """状態オブジェクトから指定フィールドのリストを取得する。"""
    records = getattr(state, records_field, None)
    if records is None:
        return None
    if isinstance(records, list):
        return records
    return None


def _get_tick_value(record: Any, tick_field: str) -> Optional[int]:
    """レコードからティック番号を取得する。"""
    if isinstance(record, dict):
        val = record.get(tick_field)
    else:
        val = getattr(record, tick_field, None)
    if isinstance(val, (int, float)):
        return int(val)
    return None


@dataclass
class ConsistencyFinding:
    """整合性検証で検出された事実。

    内部状態への書き込みを引き起こさず、ログ出力のみに使用される。
    """
    check_type: CheckType
    field_path: str
    fact: str

    def to_dict(self) -> dict[str, str]:
        return {
            "check_type": self.check_type.value,
            "field_path": self.field_path,
            "fact": self.fact,
        }


@dataclass
class ConsistencyCheckResult:
    """整合性検証の結果。

    状態として蓄積されず、検証完了後にログ出力のみに使用される。
    """
    restored_tick: int
    total_fields_checked: int
    total_patterns_applied: int
    findings: list[ConsistencyFinding]
    summary: dict[str, int]

    @property
    def total_findings(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "restored_tick": self.restored_tick,
            "total_fields_checked": self.total_fields_checked,
            "total_patterns_applied": self.total_patterns_applied,
            "findings": [f.to_dict() for f in self.findings],
            "summary": dict(self.summary),
            "total_findings": self.total_findings,
        }


def _check_type_a(
    orchestrator: Any,
    entry: ConsistencyCheckEntry,
    current_tick: int,
) -> list[ConsistencyFinding]:
    """検証種別A: ティック番号の数値的矛盾検出。

    蓄積構造内のレコードに含まれるティック番号が:
    - 復元されたティック番号を超過していないこと
    - 負値でないこと
    を照合する。
    """
    findings: list[ConsistencyFinding] = []
    state = _resolve_module_state(orchestrator, entry)
    if state is None:
        return findings

    records = _get_records_list(state, entry.records_field)
    if records is None:
        return findings

    field_path = f"{entry.module_attr}.{entry.state_sub_attr}.{entry.records_field}"

    for idx, rec in enumerate(records):
        tick_val = _get_tick_value(rec, entry.tick_field)
        if tick_val is None:
            continue

        if tick_val < 0:
            findings.append(ConsistencyFinding(
                check_type=CheckType.A,
                field_path=f"{field_path}[{idx}].{entry.tick_field}",
                fact=(
                    f"{entry.description}: "
                    f"レコード[{idx}]のティック番号({tick_val})が負値"
                ),
            ))

        if tick_val > current_tick:
            findings.append(ConsistencyFinding(
                check_type=CheckType.A,
                field_path=f"{field_path}[{idx}].{entry.tick_field}",
                fact=(
                    f"{entry.description}: "
                    f"レコード[{idx}]のティック番号({tick_val})が"
                    f"復元ティック番号({current_tick})を超過"
                ),
            ))

    return findings


def _check_type_b(
    orchestrator: Any,
    entry: ConsistencyCheckEntry,
    current_tick: int,
) -> list[ConsistencyFinding]:
    """検証種別B: 窓内レコードの時間的整合性。

    窓内レコード群が時間的に単調非減少であることを照合する。
    窓内の最古レコードのティック番号と現在ティック番号の差を記録する。
    """
    findings: list[ConsistencyFinding] = []
    state = _resolve_module_state(orchestrator, entry)
    if state is None:
        return findings

    records = _get_records_list(state, entry.records_field)
    if records is None or len(records) == 0:
        return findings

    field_path = f"{entry.module_attr}.{entry.state_sub_attr}.{entry.records_field}"

    # 単調非減少の確認
    prev_tick: Optional[int] = None
    for idx, rec in enumerate(records):
        tick_val = _get_tick_value(rec, entry.tick_field)
        if tick_val is None:
            continue

        if prev_tick is not None and tick_val < prev_tick:
            findings.append(ConsistencyFinding(
                check_type=CheckType.B,
                field_path=f"{field_path}[{idx}].{entry.tick_field}",
                fact=(
                    f"{entry.description}: "
                    f"レコード[{idx}]のティック番号({tick_val})が"
                    f"前のレコードのティック番号({prev_tick})より小さい"
                    f"（単調非減少に違反）"
                ),
            ))
        prev_tick = tick_val

    # 最古レコードとの差分が窓サイズ仮定を大幅に逸脱しているか
    first_tick = _get_tick_value(records[0], entry.tick_field)
    if first_tick is not None and entry.window_size > 0:
        gap = current_tick - first_tick
        # 窓サイズの3倍以上の差がある場合に記録（大幅逸脱の閾値）
        if gap > entry.window_size * 3:
            findings.append(ConsistencyFinding(
                check_type=CheckType.B,
                field_path=f"{field_path}[0].{entry.tick_field}",
                fact=(
                    f"{entry.description}: "
                    f"最古レコードのティック番号({first_tick})と"
                    f"復元ティック番号({current_tick})の差({gap})が"
                    f"窓サイズ宣言値({entry.window_size})の3倍を超過"
                ),
            ))

    return findings


def _check_type_c(
    orchestrator: Any,
    entry: ConsistencyCheckEntry,
    current_tick: int,
) -> list[ConsistencyFinding]:
    """検証種別C: 鮮度減衰の前提照合。

    鮮度減衰を持つモジュールの蓄積レコードにおいて、
    減衰基準値が復元後のティック番号と数値的に整合しているかを照合する。
    """
    findings: list[ConsistencyFinding] = []
    state = _resolve_module_state(orchestrator, entry)
    if state is None:
        return findings

    field_path = f"{entry.module_attr}.{entry.state_sub_attr}.{entry.records_field}"

    # pathway_last_used_tick は dict[str, int] 構造
    tick_map = getattr(state, entry.records_field, None)
    if tick_map is None:
        return findings

    if isinstance(tick_map, dict):
        for pathway_key, tick_val in tick_map.items():
            if not isinstance(tick_val, (int, float)):
                continue
            tick_int = int(tick_val)

            if tick_int < 0:
                findings.append(ConsistencyFinding(
                    check_type=CheckType.C,
                    field_path=f"{field_path}[{pathway_key}]",
                    fact=(
                        f"{entry.description}: "
                        f"経路'{pathway_key}'のティック値({tick_int})が負値"
                    ),
                ))

            if tick_int > current_tick:
                findings.append(ConsistencyFinding(
                    check_type=CheckType.C,
                    field_path=f"{field_path}[{pathway_key}]",
                    fact=(
                        f"{entry.description}: "
                        f"経路'{pathway_key}'のティック値({tick_int})が"
                        f"復元ティック番号({current_tick})を超過"
                    ),
                ))

    return findings


def _check_type_d(
    warmup_results: dict[str, str],
) -> list[ConsistencyFinding]:
    """検証種別D: ウォームアップ結果の照合。

    ウォームアップ機構が「失敗」と報告したエントリの一覧を収集する。
    """
    findings: list[ConsistencyFinding] = []

    failed_entries = [
        k for k, v in warmup_results.items() if v == "failed"
    ]

    for cache_attr in failed_entries:
        findings.append(ConsistencyFinding(
            check_type=CheckType.D,
            field_path=cache_attr,
            fact=(
                f"ウォームアップ再導出失敗: "
                f"キャッシュ'{cache_attr}'の再導出が失敗した"
            ),
        ))

    return findings


def _check_type_e(
    orchestrator: Any,
    current_tick: int,
) -> list[ConsistencyFinding]:
    """検証種別E: フィールド間の数値的前提照合。

    復元された複数フィールドが共有する数値的前提を照合する。
    """
    findings: list[ConsistencyFinding] = []

    # tick_count の非負値確認
    if current_tick < 0:
        findings.append(ConsistencyFinding(
            check_type=CheckType.E,
            field_path="_tick_count",
            fact=f"復元されたティック番号({current_tick})が負値",
        ))

    # session_resume_tick が存在する場合、tick_count 以下であることを照合
    resume_tick = getattr(orchestrator, "_session_resume_tick", None)
    if resume_tick is not None and isinstance(resume_tick, (int, float)):
        if int(resume_tick) > current_tick:
            findings.append(ConsistencyFinding(
                check_type=CheckType.E,
                field_path="_session_resume_tick",
                fact=(
                    f"セッション復帰ティック({int(resume_tick)})が"
                    f"現在ティック番号({current_tick})を超過"
                ),
            ))

    return findings


def execute_session_recovery_check(
    orchestrator: Any,
    warmup_results: Optional[dict[str, str]] = None,
) -> ConsistencyCheckResult:
    """セッション復帰時の状態整合性検証を実行する。

    CONSISTENCY_CHECK_ENTRIES の静的宣言に従い、復元されたフィールド間の
    数値的整合性を受動的に走査する。

    本関数は永続化復元 → ウォームアップ完了後に1回だけ呼び出される。
    内部状態を保持せず、オーケストレータの属性に書き込みを行わない。
    検証結果はログへの警告出力のみを行う。

    安全弁:
      1. 修復禁止: 不整合検出に基づく状態の修復・補正・正規化を一切行わない
      2. 進行阻止禁止: 不整合検出時もティック進行を阻止・遅延しない
      3. 規範的基準の排除: 数値的矛盾の検出に限定
      4. 状態無書き込み: オーケストレータの属性に対する書き込みを行わない
      5. 1回実行・痕跡なし: 実行後に固有の状態を残さない

    Args:
        orchestrator: PsycheOrchestrator インスタンス (読み取り専用)
        warmup_results: execute_warmup() の結果辞書 (検証種別Dで使用)

    Returns:
        ConsistencyCheckResult: 検証結果
    """
    current_tick: int = getattr(orchestrator, "_tick_count", 0)
    all_findings: list[ConsistencyFinding] = []
    fields_checked = 0
    patterns_applied = 0

    for entry in CONSISTENCY_CHECK_ENTRIES:
        patterns_applied += 1
        try:
            if entry.check_type == CheckType.A:
                fields_checked += 1
                findings = _check_type_a(orchestrator, entry, current_tick)
                all_findings.extend(findings)

            elif entry.check_type == CheckType.B:
                fields_checked += 1
                findings = _check_type_b(orchestrator, entry, current_tick)
                all_findings.extend(findings)

            elif entry.check_type == CheckType.C:
                fields_checked += 1
                findings = _check_type_c(orchestrator, entry, current_tick)
                all_findings.extend(findings)

            elif entry.check_type == CheckType.D:
                if warmup_results is not None:
                    findings = _check_type_d(warmup_results)
                    all_findings.extend(findings)

            elif entry.check_type == CheckType.E:
                fields_checked += 1
                findings = _check_type_e(orchestrator, current_tick)
                all_findings.extend(findings)

        except Exception as e:
            logger.debug(
                "Recovery check: error processing entry '%s': %s",
                entry.description, e,
            )
            continue

    # 検証種別ごとの集計
    summary: dict[str, int] = {}
    for ct in CheckType:
        count = sum(1 for f in all_findings if f.check_type == ct)
        summary[ct.value] = count

    # ログ出力
    total_findings = len(all_findings)
    if total_findings > 0:
        logger.warning(
            "Session recovery check: %d inconsistencies detected "
            "(tick=%d, fields=%d, patterns=%d)",
            total_findings, current_tick, fields_checked, patterns_applied,
        )
        for f in all_findings:
            logger.warning(
                "  [%s] %s: %s", f.check_type.value, f.field_path, f.fact,
            )
    else:
        logger.info(
            "Session recovery check: no inconsistencies "
            "(tick=%d, fields=%d, patterns=%d)",
            current_tick, fields_checked, patterns_applied,
        )

    return ConsistencyCheckResult(
        restored_tick=current_tick,
        total_fields_checked=fields_checked,
        total_patterns_applied=patterns_applied,
        findings=all_findings,
        summary=summary,
    )


def get_consistency_check_entries() -> tuple[ConsistencyCheckEntry, ...]:
    """整合性検証エントリの静的宣言テーブルを返す (テスト・検証用)。"""
    return CONSISTENCY_CHECK_ENTRIES


# ══════════════════════════════════════════════════════════════════════════════
# セッション間差分記述
# design_session_difference.md に基づく実装
# ══════════════════════════════════════════════════════════════════════════════


def _compute_field_distance(prev_value: Any, curr_value: Any) -> float:
    """2つのフィールド値間の数値的距離を算出する。

    フィールド値の構造によって算出方法が異なる:
    - スカラー数値（整数・浮動小数点）: 差の絶対値
    - 辞書型: 内部数値フィールドの差の絶対値の合計（非数値は無視）
    - リスト型: レコード件数の差の絶対値のみ
    - その他: 0（距離算出不可能）

    Args:
        prev_value: 前回スナップショットの値
        curr_value: 現在の値

    Returns:
        数値的距離（非負のスカラー値）
    """
    # 両方がスカラー数値の場合
    if isinstance(prev_value, (int, float)) and isinstance(curr_value, (int, float)):
        return abs(float(curr_value) - float(prev_value))

    # 両方が辞書型の場合
    if isinstance(prev_value, dict) and isinstance(curr_value, dict):
        total = 0.0
        all_keys = set(prev_value.keys()) | set(curr_value.keys())
        for key in all_keys:
            pv = prev_value.get(key)
            cv = curr_value.get(key)
            if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
                total += abs(float(cv) - float(pv))
            elif isinstance(pv, dict) and isinstance(cv, dict):
                # 再帰的に辞書内の数値フィールドを走査
                total += _compute_field_distance(pv, cv)
        return total

    # 両方がリスト型の場合
    if isinstance(prev_value, list) and isinstance(curr_value, list):
        return abs(float(len(curr_value)) - float(len(prev_value)))

    # 型が異なるか、距離算出不可能な型の場合
    return 0.0


def compute_session_difference_scalar(
    prev_snapshot: dict[str, Any],
    current_dict: dict[str, Any],
) -> float:
    """前回スナップショットと現在の保存辞書間の数値的距離の総和を算出する。

    設計書の仕様:
    - 全永続化フィールドのフィールドごとの数値的距離を算出
    - 各フィールドの距離をスカラー値に集約（総和）
    - 片方にのみ存在するフィールドの距離は0とする

    メタデータフィールド（version, save_timestamp, tick_count, session_diff_scalar）
    は差分算出の対象外とする。

    Args:
        prev_snapshot: 前回復元時の辞書データ
        current_dict: 現在の保存辞書データ

    Returns:
        セッション間差分のスカラー要約値（非負）
    """
    # 差分算出対象外のメタデータキー
    exclude_keys = {"version", "save_timestamp", "tick_count", "session_diff_scalar"}

    total_distance = 0.0
    all_keys = (set(prev_snapshot.keys()) | set(current_dict.keys())) - exclude_keys

    for key in all_keys:
        prev_val = prev_snapshot.get(key)
        curr_val = current_dict.get(key)

        if prev_val is None or curr_val is None:
            # 片方にのみ存在するフィールド: 距離0
            continue

        total_distance += _compute_field_distance(prev_val, curr_val)

    return total_distance


# ── セッション間差分の段階値変換 ──────────────────────────────────────
#
# 安全弁3: スカラー値から段階値への変換は固定の区間分割であり、
# 内部状態に依存しない。区間の境界値は静的に定義される。

# 段階値の区間分割定義（境界値, テキスト）
# 設計書: 「ほぼ変化なし」/「微小な変化」/「中程度の変化」/「大きな変化」
_SESSION_DIFF_THRESHOLDS: list[tuple[float, str]] = [
    (0.1, "ほぼ変化なし"),
    (5.0, "微小な変化"),
    (50.0, "中程度の変化"),
]
_SESSION_DIFF_LARGE_LABEL = "大きな変化"

# 差分値不在時の空状態テキスト
SESSION_DIFF_EMPTY_LABEL = "(不明)"


def classify_session_difference(scalar: Optional[float]) -> str:
    """セッション間差分スカラー値を段階値テキストに変換する。

    安全弁3: 固定の区間分割による機械的変換。
    安全弁4: 評価的表現を含まない事実記述のみ。
    安全弁5: フィールド内訳を含まない。

    Args:
        scalar: セッション間差分のスカラー値。Noneの場合は空状態

    Returns:
        段階値テキスト
    """
    if scalar is None:
        return SESSION_DIFF_EMPTY_LABEL

    for threshold, label in _SESSION_DIFF_THRESHOLDS:
        if scalar <= threshold:
            return label

    return _SESSION_DIFF_LARGE_LABEL


def build_session_diff_enrichment_text(scalar: Optional[float]) -> str:
    """セッション間差分のenrichment項目テキストを生成する。

    安全弁1: enrichment経由の間接参照のみ。
    安全弁4: 評価的表現を含まない事実記述のみ。
    安全弁5: フィールド内訳を含まない（スカラー要約のみ）。

    Args:
        scalar: セッション間差分のスカラー値。Noneの場合は空状態

    Returns:
        enrichment項目テキスト
    """
    label = classify_session_difference(scalar)
    return f"セッション間状態変化: {label}"
