"""
psyche/responsibility_manager.py - 責任状態の永続化管理

内部データ構造や保存場所を他の機能から隠蔽する。
外部からは責任モジュールの公開APIを通じてのみアクセス可能。

Usage::

    from psyche.responsibility_manager import ResponsibilityManager

    mgr = ResponsibilityManager()
    state = mgr.get_state("user_id")
    state = mgr.record_decision("user_id", policy, context)
    state = mgr.evaluate_outcome("user_id", decision_id, outcome)
    influence = mgr.get_influence("user_id")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .responsibility import (
    ResponsibilityState,
    ResponsibilityInfluence,
    record_decision as _record_decision,
    evaluate_outcome as _evaluate_outcome,
    apply_decay as _apply_decay,
    get_influence as _get_influence,
    create_default_state,
    to_dict,
    from_dict,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RESPONSIBILITY_FILE = DATA_DIR / "responsibility.json"


class ResponsibilityManager:
    """責任状態の永続化マネージャ

    ファイルベースで責任状態を保存・読み込みする。
    内部構造は隠蔽され、外部からはAPIを通じてのみアクセス可能。
    """

    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or RESPONSIBILITY_FILE
        self._data: dict[str, dict] = self._load()
        self._save_suppressed: bool = False
        logger.info("ResponsibilityManager loaded from %s", self.filepath)

    def _load(self) -> dict[str, dict]:
        """ファイルから読み込む。欠損時は空辞書を返す。"""
        if not self.filepath.exists():
            return {}
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load responsibility data: %s", e)
            return {}

    def _save(self):
        """ファイルに保存する（アトミック書き込み）。抑制中はスキップ。"""
        if self._save_suppressed:
            return
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.filepath.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.filepath)

    def _get_raw_state(self, user_id: str) -> ResponsibilityState:
        """内部用: 生の状態を取得（減衰適用前）"""
        if user_id not in self._data:
            self._data[user_id] = to_dict(create_default_state())
            self._save()
        return from_dict(self._data[user_id])

    def _apply_time_decay(self, state: ResponsibilityState) -> ResponsibilityState:
        """時間経過による減衰を適用する"""
        try:
            last = datetime.fromisoformat(state.last_updated)
            now = datetime.now()
            hours_elapsed = (now - last).total_seconds() / 3600.0
            if hours_elapsed > 0.1:  # 6分以上経過していれば減衰
                return _apply_decay(state, hours_elapsed)
        except (ValueError, KeyError):
            pass
        return state

    # ── Public API ─────────────────────────────────────────────

    def get_state(self, user_id: str) -> ResponsibilityState:
        """責任状態を取得する（減衰適用済み）

        Args:
            user_id: ユーザーID

        Returns:
            ResponsibilityState
        """
        state = self._get_raw_state(user_id)
        state = self._apply_time_decay(state)
        self._data[user_id] = to_dict(state)
        return state

    def record_decision(
        self,
        user_id: str,
        policy: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[ResponsibilityState, str]:
        """判断を記録する

        判断（Policy）を確定した瞬間に呼び出す。

        Args:
            user_id: ユーザーID
            policy: 確定した方針
            context: 判断時のコンテキスト

        Returns:
            (新しい ResponsibilityState, decision_id)
        """
        # Suppress intermediate saves during get_state, save once at end
        self._save_suppressed = True
        state = self.get_state(user_id)
        self._save_suppressed = False
        new_state, decision_id = _record_decision(state, policy, context)
        self._data[user_id] = to_dict(new_state)
        self._save()
        logger.debug(
            "Decision recorded: user=%s, policy=%s, id=%s",
            user_id, policy.get("policy_label", "unknown"), decision_id
        )
        return new_state, decision_id

    def evaluate_outcome(
        self,
        user_id: str,
        decision_id: str,
        outcome: dict[str, Any],
    ) -> ResponsibilityState:
        """結果を観測して責任を評価する

        ユーザーの反応や関係性の変化を観測した後に呼び出す。

        Args:
            user_id: ユーザーID
            decision_id: 評価対象の決定ID
            outcome: 観測された結果

        Returns:
            新しい ResponsibilityState
        """
        # Suppress intermediate saves during get_state, save once at end
        self._save_suppressed = True
        state = self.get_state(user_id)
        self._save_suppressed = False
        new_state = _evaluate_outcome(state, decision_id, outcome)
        self._data[user_id] = to_dict(new_state)
        self._save()
        logger.debug(
            "Outcome evaluated: user=%s, decision=%s, reaction=%s",
            user_id, decision_id, outcome.get("user_reaction", "unknown")
        )
        return new_state

    def get_influence(self, user_id: str) -> ResponsibilityInfluence:
        """責任が心理状態に与える影響を取得する

        Args:
            user_id: ユーザーID

        Returns:
            ResponsibilityInfluence
        """
        state = self.get_state(user_id)
        return _get_influence(state)

    def get_summary(self, user_id: str) -> dict[str, Any]:
        """責任状態のサマリーを取得する（デバッグ・表示用）

        Args:
            user_id: ユーザーID

        Returns:
            サマリー辞書
        """
        state = self.get_state(user_id)
        influence = _get_influence(state)
        return {
            "total_weight": round(state.total_weight, 4),
            "accumulated_harm": round(state.accumulated_harm, 4),
            "accumulated_confidence": round(state.accumulated_confidence, 4),
            "pending_decisions": state.pending_decisions,
            "recent_decision_count": len(state.recent_decisions),
            "influence": {
                "fear_amplification": influence.fear_amplification,
                "caution_bias": influence.caution_bias,
                "anxiety_baseline": influence.anxiety_baseline,
                "empathy_bias": influence.empathy_bias,
            },
        }


# ── Convenience functions for calc_ style API ──────────────────

def calc_responsibility_influence(
    state: ResponsibilityState,
) -> ResponsibilityInfluence:
    """責任状態から影響を計算する（外部向けAPI）"""
    return _get_influence(state)
