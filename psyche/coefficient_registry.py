"""
psyche/coefficient_registry.py - 散在する定数の外部ファイル集約と起動時読み込み

外部係数ファイル（data/coefficients.json）を起動時に一度だけ読み込み、
各モジュールに読み取り専用で定数値を提供する。

設計原則 (design_coefficient_file.md 準拠):
- 起動時の一度の読み込みのみ（ティック実行中の再読み込み禁止）
- 読み取り専用（書き込み経路なし）
- ファイル不在時は全デフォルト値使用
- 個別定数欠落時はその定数のみデフォルト値
- 導出ロジックの外部化禁止（定数値のみ）
- enrichmentに非露出

セッション間変動履歴 (design_coefficient_history.md 準拠):
- load()完了後に全係数値のスナップショットを取得
- 前回記録との差分を算出し履歴ファイルに追記
- FIFO制限で古いエントリを自動削除
- 履歴データはpsycheに帰還しない（開発者向け記録のみ）
- enrichmentに非露出
- save/loadフィールドに非追加

安全弁:
  1. 読み取り専用の強制（書き込みメソッドなし）
  2. フォールバックの保証（デフォルト値 = ハードコード値と同一）
  3. ティック中の再読み込み禁止（起動時のみ）
  4. enrichmentへの非露出
  5. 導出ロジックの非外部化
  6. FIFO上限（履歴エントリ数制限）
  7. 読み書き失敗の安全な無視
  8. psycheへの非帰還
  9. 永続化機構との非接続
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Default values (identical to current hardcoded values)
# =============================================================================

_DEFAULTS: dict[str, Any] = {
    # ── Category A: Drive dynamics band constants ──
    "drive_dynamics": {
        "section_band": {
            "emotion_drive_coupling": {"social": 0.06, "curiosity": 0.06, "expression": 0.06},
            "drive_interaction":      {"social": 0.03, "curiosity": 0.03, "expression": 0.03},
            "goal_hierarchy":         {"social": 0.05, "curiosity": 0.05, "expression": 0.05},
            "time_passage":           {"social": 0.06, "curiosity": 0.06, "expression": 0.06},
            "arousal_drive":          {"social": 0.04, "curiosity": 0.04, "expression": 0.04},
            "behavioral_diversity":   {"social": 0.02, "curiosity": 0.02, "expression": 0.02},
            "internal_contradiction": {"social": 0.02, "curiosity": 0.02, "expression": 0.02},
            "result_diversity_return": {"social": 0.03, "curiosity": 0.03, "expression": 0.03},
        },
        "total_change_limit": 0.15,
    },

    # ── Category B: Mood autonomy band constants ──
    "mood_autonomy": {
        "mood_band": {
            "emotion":  {"valence": 0.12, "arousal": 0.10},
            "drive":    {"valence": 0.05, "arousal": 0.04},
            "goal":     {"valence": 0.03, "arousal": 0.02},
            "fear":     {"valence": 0.00, "arousal": 0.06},
        },
        "tracking_speed_min": 0.03,
        "tracking_speed_max": 0.25,
        "mood_delta_limit": 0.15,
    },

    # ── Category C: Policy selection band constants ──
    "policy_selection": {
        "score_section_band": 1.5,
    },

    # ── Category D: Value orientation constants ──
    "value_orientation": {
        "base_learning_rate": 0.01,
        "confidence_damping": 0.5,
        "confidence_growth_rate": 0.005,
        "confidence_decay_rate": 0.001,
        "max_bias_strength": 0.15,
        "min_dimension_threshold": 0.1,
        "confidence_bias_amplifier": 0.5,
        "neutral_decay_rate": 0.0001,
    },

    # ── Category E: Fluctuation constants ──
    "fluctuation": {
        "amplitude_cap": 0.12,
        "amplitude_floor": 0.005,
    },

    # ── Category F: Experience intensity band expansion constants ──
    "experience_intensity": {
        "bandwidth_max_multiplier": 4.0,
        "bandwidth_max_delta_per_dim": 0.08,
        "bandwidth_cooldown_ticks": 3,
        "cooldown_min_ticks": 2,
        "drive_limit_multiplier_max": 1.3,
        "score_band_addition_max": 0.5,
        # ── Cumulative safety valve constants ──
        "cumulative_limit_ratio": 2.5,
        "consecutive_firing_threshold": 3,
        "consecutive_firing_decay_base": 0.85,
        "consecutive_firing_min_factor": 0.3,
        "firing_window_size": 10,
    },

    # ── Category G: Emotion processing constants ──
    "emotion_processing": {
        "decay_rate": 0.95,
        "stimulus_base_delta": 0.2,
        "valence_positive": {"joy": 0.15, "love": 0.05, "fun": 0.05},
        "valence_negative": {"sorrow": 0.10, "anger": 0.05, "fear": 0.05},
    },

    # ── Category H: Perception processing constants ──
    "perception": {
        "bias_bandwidth": 0.04,
        "bias_coefficient": 0.1,
    },

    # ── Category I: Memory emotion return constants ──
    "memory_emotion_return": {
        "per_candidate_max_delta": 0.03,
        "total_max_delta": 0.15,
        "rumination_threshold": 2,
        "rumination_decay_factor": 0.5,
        "low_arousal_threshold": 0.2,
        "low_arousal_scale": 0.3,
        "convergence_scale": 0.5,
        "direction_freshness_decay": 0.8,
        "tracking_speed_modulation_ratio_cap": 0.10,
        "tracking_speed_modulation_scale": 0.02,
    },

    # ── Category J: Other hypothesis emotion return constants ──
    "other_hypothesis_emotion_return": {
        "per_candidate_max_delta": 0.02,
        "total_max_delta": 0.07,
        "rumination_threshold": 2,
        "rumination_decay_factor": 0.5,
        "low_arousal_threshold": 0.2,
        "low_arousal_scale": 0.3,
        "convergence_scale": 0.5,
        "combined_max_delta": 0.15,
    },

    # ── Category K: Description layer common constants ──
    # 記述層モジュール群の共通パターン定数（等価変換・値変更なし）
    "description_common": {
        # FIFO蓄積上限 — 複数モジュールに共通するFIFOバッファの最大保持件数
        "fifo_limit_30": 30,     # stabilization / behavioral_diversity / forgetting_recall_balance / reference_frequency / attention_distribution / input_pathway_balance(snapshot)
        "fifo_limit_50": 50,     # selection_attribution / self_action_perception / intent_action_gap / emotion_cooccurrence / perceptual_context(summaries)
        "fifo_limit_100": 100,   # interaction_accumulation(pairs) / temporal_cognition(elapsed) / responsibility_temporal_trace(snapshots)
        "fifo_limit_200": 200,   # expectation_lifecycle / goal_hierarchy_propagation / input_pathway_balance(usage_facts) / other_boundary(total) / situational_self_presentation(total) / hypothesis_observation(total)
        # スライディングウィンドウサイズ — 複数モジュールに共通する時間窓
        "window_size_25": 25,    # introspection_cross_section
        "window_size_30": 30,    # emotional_backdrop_cognition / multi_path_recall(rumination) / spontaneous_recall(rumination)
        "window_size_50": 50,    # internal_contradiction / drive_variation / input_pathway_balance(sliding)
        # 鮮度減衰速度 — 複数モジュールに共通する鮮度減衰レート
        "freshness_decay_rate_002": 0.02,  # emotion_cooccurrence / drive_variation / emotional_backdrop / other_boundary / situational_self_presentation / hypothesis_observation / goal_hierarchy / expectation_lifecycle
    },
}


# =============================================================================
# Registry (module-level singleton)
# =============================================================================

_registry: dict[str, Any] | None = None
_loaded: bool = False


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults.

    For each key in defaults:
    - If the key exists in overrides and both values are dicts, recurse.
    - If the key exists in overrides and the override value is not a dict
      (or the default value is not a dict), use the override value.
    - If the key does not exist in overrides, keep the default value.

    Keys in overrides that do not exist in defaults are ignored.
    """
    result = {}
    for key, default_val in defaults.items():
        if key in overrides:
            override_val = overrides[key]
            if isinstance(default_val, dict) and isinstance(override_val, dict):
                result[key] = _deep_merge(default_val, override_val)
            else:
                result[key] = override_val
        else:
            # Deep copy dicts to prevent mutation
            if isinstance(default_val, dict):
                result[key] = _deep_copy_dict(default_val)
            else:
                result[key] = default_val
    return result


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dict structure (dicts and primitives only)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        else:
            result[k] = v
    return result


def _resolve_coefficient_file_path() -> str:
    """Resolve the path to the coefficients.json file.

    Looks for data/coefficients.json relative to the project root
    (parent of the psyche/ directory).
    """
    psyche_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(psyche_dir)
    return os.path.join(project_root, "data", "coefficients.json")


def load(file_path: str | None = None) -> None:
    """Load coefficients from the external JSON file.

    This function should be called once at system startup, before
    any module initialization.

    If the file does not exist or cannot be read, all defaults are used.
    If individual constants are missing from the file, their defaults
    are used.

    Args:
        file_path: Optional path to the coefficients file. If None,
                   the default path (data/coefficients.json) is used.
    """
    global _registry, _loaded

    path = file_path or _resolve_coefficient_file_path()

    if not os.path.isfile(path):
        logger.info(
            "Coefficient file not found at %s; using all defaults.", path
        )
        _registry = _deep_copy_dict(_DEFAULTS)
        _loaded = True
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning(
            "Failed to read coefficient file %s: %s; using all defaults.",
            path, e,
        )
        _registry = _deep_copy_dict(_DEFAULTS)
        _loaded = True
        return

    if not isinstance(file_data, dict):
        logger.warning(
            "Coefficient file content is not a dict; using all defaults."
        )
        _registry = _deep_copy_dict(_DEFAULTS)
        _loaded = True
        return

    # Merge file values into defaults (missing values fall back to defaults)
    _registry = _deep_merge(_DEFAULTS, file_data)
    _loaded = True
    logger.info("Coefficients loaded from %s.", path)


def get(category: str, key: str | None = None) -> Any:
    """Get coefficient value(s) by category and optional key.

    If the registry has not been loaded yet, it is initialized with
    all default values (equivalent to file-not-found behavior).

    Args:
        category: Top-level category name (e.g., "drive_dynamics").
        key: Optional key within the category. If None, returns the
             entire category dict.

    Returns:
        The coefficient value. For nested dicts, returns a deep copy
        to prevent external mutation.

    Raises:
        KeyError: If the category or key does not exist in the defaults.
    """
    global _registry, _loaded

    if not _loaded:
        # Auto-initialize with defaults if not explicitly loaded
        _registry = _deep_copy_dict(_DEFAULTS)
        _loaded = True

    if category not in _registry:
        raise KeyError(f"Unknown coefficient category: {category}")

    cat_data = _registry[category]

    if key is None:
        # Return deep copy of entire category
        if isinstance(cat_data, dict):
            return _deep_copy_dict(cat_data)
        return cat_data

    if not isinstance(cat_data, dict):
        raise KeyError(
            f"Category '{category}' is not a dict; cannot access key '{key}'"
        )

    if key not in cat_data:
        raise KeyError(
            f"Unknown coefficient key '{key}' in category '{category}'"
        )

    val = cat_data[key]
    if isinstance(val, dict):
        return _deep_copy_dict(val)
    return val


def get_defaults() -> dict[str, Any]:
    """Return a deep copy of all default values.

    This is useful for testing and verification purposes.
    """
    return _deep_copy_dict(_DEFAULTS)


def reset() -> None:
    """Reset the registry to unloaded state.

    This is intended for testing purposes only.
    """
    global _registry, _loaded
    _registry = None
    _loaded = False


# =============================================================================
# Session history recording (design_coefficient_history.md)
# =============================================================================

_HISTORY_FIFO_LIMIT: int = 100


def _resolve_history_file_path() -> str:
    """Resolve the path to the coefficient_history.json file.

    Located at data/coefficient_history.json, separate from coefficients.json.
    """
    psyche_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(psyche_dir)
    return os.path.join(project_root, "data", "coefficient_history.json")


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys.

    Example: {"a": {"b": 1}} -> {"a.b": 1}
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, full_key))
        else:
            result[full_key] = v
    return result


def _compute_changes(
    previous_snapshot: dict[str, Any],
    current_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute the list of changed coefficients between two snapshots.

    Both snapshots are flat dicts (dot-separated keys).
    Returns a list of dicts with keys: "key", "old_value", "new_value".
    """
    changes: list[dict[str, Any]] = []
    all_keys = sorted(set(previous_snapshot.keys()) | set(current_snapshot.keys()))
    for key in all_keys:
        old_val = previous_snapshot.get(key)
        new_val = current_snapshot.get(key)
        if old_val != new_val:
            change: dict[str, Any] = {"key": key}
            if old_val is not None:
                change["old_value"] = old_val
            if new_val is not None:
                change["new_value"] = new_val
            changes.append(change)
    return changes


def _load_history(history_path: str) -> list[dict[str, Any]]:
    """Load existing history from file. Returns empty list on any failure."""
    if not os.path.isfile(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("History file content is not a list; starting fresh.")
        return []
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("Failed to read history file %s: %s; starting fresh.", history_path, e)
        return []


def _save_history(history_path: str, entries: list[dict[str, Any]]) -> None:
    """Save history entries to file. Logs warning on failure."""
    try:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Failed to write history file %s: %s", history_path, e)


def record_history(history_path: str | None = None) -> dict[str, Any] | None:
    """Record a history entry comparing current coefficients with the previous session.

    This function should be called once after load() completes.
    It takes a snapshot of the current registry values, compares with the most
    recent history entry (if any), and appends a new entry to the history file.

    The history file is separate from coefficients.json and has no effect on
    psyche processing. It is a developer-facing record only.

    Args:
        history_path: Optional path to the history file. If None,
                      the default path (data/coefficient_history.json) is used.

    Returns:
        The newly created history entry dict, or None if the registry
        has not been loaded yet or if recording failed.
    """
    if not _loaded or _registry is None:
        logger.warning("Cannot record history: registry not loaded.")
        return None

    path = history_path or _resolve_history_file_path()

    # Take snapshot of current values (flattened)
    current_flat = _flatten_dict(_registry)

    # Load existing history
    history = _load_history(path)

    # Compare with most recent entry's snapshot
    changes: list[dict[str, Any]] = []
    if history:
        last_entry = history[-1]
        previous_flat = last_entry.get("snapshot", {})
        changes = _compute_changes(previous_flat, current_flat)

    # Build new entry
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changes": changes,
        "change_count": len(changes),
        "snapshot": current_flat,
    }

    # Append and apply FIFO
    history.append(entry)
    if len(history) > _HISTORY_FIFO_LIMIT:
        history = history[-_HISTORY_FIFO_LIMIT:]

    # Save
    _save_history(path, history)

    if changes:
        logger.info(
            "Coefficient history recorded: %d change(s) detected.",
            len(changes),
        )
    else:
        if len(history) == 1:
            logger.info("Coefficient history recorded: first session (no previous data).")
        else:
            logger.info("Coefficient history recorded: no changes from previous session.")

    return entry


def get_history(history_path: str | None = None) -> list[dict[str, Any]]:
    """Read and return the current history entries.

    This is intended for developer inspection and testing only.
    The returned data must never be used as input to psyche processing.

    Args:
        history_path: Optional path to the history file. If None,
                      the default path is used.

    Returns:
        A list of history entry dicts. Empty list if no history exists.
    """
    path = history_path or _resolve_history_file_path()
    return _load_history(path)
