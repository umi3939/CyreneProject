"""
psyche/memory_system_integration.py - Memory System Integration (LOCAL ONLY).

記憶系統統合（自己観測記憶系統と長期要約記憶系統）

異質な記憶表現を同一の参照窓で扱えるようにし、参照結果の偏りを減らす。
統合の目的は記憶の一元化ではなく、共通の記述単位への正規化と参照候補化。

この機能は記憶内容の正誤判定を行わない。
この機能は単一系統の記憶を恒常的に優先しない。
この機能は記憶の参照結果だけで応答方向を確定しない。
この機能は既存記憶の意味づけを固定しない。

入力参照:
  自己観測由来の出来事記録、長期要約由来の圧縮記録、
  現在の感情断面、対話文脈断面、時間経過断面

処理順序:
  抽象化 → 整列 → 重複調整 → 競合保持 → 受け渡し準備

Usage::

    integrator = create_integrator()
    result = integrator.integrate(
        episodes=episode_store,
        long_term_memories=recalled_memories,
        bindings=binding_store,
        context=integration_context,
    )
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Memory Source ───────────────────────────────────────────


class MemorySource(Enum):
    """記憶の出所系統。"""
    EPISODIC = "episodic"            # 自己観測由来の出来事記録
    LONG_TERM = "long_term"          # 長期要約由来の圧縮記録
    BINDING = "binding"              # 感情記憶の紐づけ
    ACTION_RESULT = "action_result"  # 行動-結果対由来の選択-観測記録


# ── Temporal Phase ──────────────────────────────────────────


class TemporalPhase(Enum):
    """記憶の時相区分。"""
    IMMEDIATE = "immediate"   # ~5分以内
    RECENT = "recent"         # ~1時間以内
    MEDIUM = "medium"         # ~1日以内
    DISTANT = "distant"       # それ以前


# ── Unified Memory Unit ────────────────────────────────────


@dataclass
class UnifiedMemoryUnit:
    """共通記述単位。可逆変換を前提とし、元系統への逆参照経路を保持。"""
    unit_id: str = ""
    source: MemorySource = MemorySource.EPISODIC
    source_id: str = ""            # 元系統での一意ID
    summary: str = ""              # 正規化された要約
    topics: list[str] = field(default_factory=list)
    temporal_phase: TemporalPhase = TemporalPhase.DISTANT
    timestamp: float = 0.0        # 元記憶のタイムスタンプ
    certainty: float = 0.5        # 確からしさ (0.0-1.0)
    relevance: float = 0.0        # 現在文脈との関連度 (0.0-1.0)
    reuse_count: int = 0          # 再利用回数
    freshness: float = 0.5        # 鮮度 (0.0-1.0)
    emotional_valence: float = 0.0  # 感情価 (-1.0-1.0)
    emotional_label: str = ""     # 主要感情ラベル
    importance: float = 0.5       # 重要度 (0.0-1.0)
    original_data: dict[str, Any] = field(default_factory=dict)  # 逆参照用

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source": self.source.value,
            "source_id": self.source_id,
            "summary": self.summary,
            "topics": list(self.topics),
            "temporal_phase": self.temporal_phase.value,
            "timestamp": self.timestamp,
            "certainty": self.certainty,
            "relevance": self.relevance,
            "reuse_count": self.reuse_count,
            "freshness": self.freshness,
            "emotional_valence": self.emotional_valence,
            "emotional_label": self.emotional_label,
            "importance": self.importance,
            "original_data": self.original_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedMemoryUnit:
        try:
            source = MemorySource(data.get("source", "episodic"))
        except ValueError:
            source = MemorySource.EPISODIC
        try:
            phase = TemporalPhase(data.get("temporal_phase", "distant"))
        except ValueError:
            phase = TemporalPhase.DISTANT
        return cls(
            unit_id=data.get("unit_id", ""),
            source=source,
            source_id=data.get("source_id", ""),
            summary=data.get("summary", ""),
            topics=data.get("topics", []),
            temporal_phase=phase,
            timestamp=data.get("timestamp", 0.0),
            certainty=data.get("certainty", 0.5),
            relevance=data.get("relevance", 0.0),
            reuse_count=data.get("reuse_count", 0),
            freshness=data.get("freshness", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
            emotional_label=data.get("emotional_label", ""),
            importance=data.get("importance", 0.5),
            original_data=data.get("original_data", {}),
        )

    def to_memory_dict(self) -> dict[str, Any]:
        """brain.py / expression.py 互換のメモリ dict 形式。"""
        return {
            "summary": self.summary,
            "keywords": list(self.topics),
            "importance": int(self.importance * 5),
            "date": str(self.timestamp),
            "_source": self.source.value,
            "_unit_id": self.unit_id,
            "_certainty": self.certainty,
            "_relevance": self.relevance,
            "_emotional_valence": self.emotional_valence,
            "_integrated": True,
        }


# ── Duplicate Entry ─────────────────────────────────────────


@dataclass
class DuplicateEntry:
    """重複対応表エントリ。同一事象の複数視点を並立保持。"""
    group_id: str = ""
    unit_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    similarity: float = 0.0
    topic_overlap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "unit_ids": list(self.unit_ids),
            "sources": list(self.sources),
            "similarity": self.similarity,
            "topic_overlap": self.topic_overlap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DuplicateEntry:
        return cls(
            group_id=data.get("group_id", ""),
            unit_ids=data.get("unit_ids", []),
            sources=data.get("sources", []),
            similarity=data.get("similarity", 0.0),
            topic_overlap=data.get("topic_overlap", 0.0),
        )


# ── Conflict Entry ──────────────────────────────────────────


@dataclass
class ConflictEntry:
    """競合対応表エントリ。矛盾を解消せず併存させる。"""
    conflict_id: str = ""
    unit_id_a: str = ""
    unit_id_b: str = ""
    conflict_type: str = ""   # valence_mismatch, importance_gap, topic_contradiction
    severity: float = 0.0     # 0.0-1.0
    turn_hidden: int = 0      # 不可視化されたターン数
    visible: bool = True       # 参照時の可視状態

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "unit_id_a": self.unit_id_a,
            "unit_id_b": self.unit_id_b,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "turn_hidden": self.turn_hidden,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConflictEntry:
        return cls(
            conflict_id=data.get("conflict_id", ""),
            unit_id_a=data.get("unit_id_a", ""),
            unit_id_b=data.get("unit_id_b", ""),
            conflict_type=data.get("conflict_type", ""),
            severity=data.get("severity", 0.0),
            turn_hidden=data.get("turn_hidden", 0),
            visible=data.get("visible", True),
        )


# ── Reference History Entry ─────────────────────────────────


@dataclass
class ReferenceHistoryEntry:
    """参照履歴エントリ。累積と希薄化を併置。"""
    unit_id: str = ""
    turn: int = 0
    relevance_at_ref: float = 0.0
    decay_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "turn": self.turn,
            "relevance_at_ref": self.relevance_at_ref,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceHistoryEntry:
        return cls(
            unit_id=data.get("unit_id", ""),
            turn=data.get("turn", 0),
            relevance_at_ref=data.get("relevance_at_ref", 0.0),
            decay_factor=data.get("decay_factor", 1.0),
        )


# ── Integration Context ────────────────────────────────────


@dataclass
class IntegrationContext:
    """統合に必要な文脈情報。"""
    # 感情断面
    emotions: dict[str, float] = field(default_factory=dict)
    mood_valence: float = 0.0
    # 対話文脈断面
    percept_topics: list[str] = field(default_factory=list)
    percept_text: str = ""
    percept_intent: str = "unknown"
    # 時間経過断面
    current_time: float = field(default_factory=time.time)
    tick_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotions": dict(self.emotions),
            "mood_valence": self.mood_valence,
            "percept_topics": list(self.percept_topics),
            "percept_text": self.percept_text,
            "percept_intent": self.percept_intent,
            "current_time": self.current_time,
            "tick_count": self.tick_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationContext:
        return cls(
            emotions=data.get("emotions", {}),
            mood_valence=data.get("mood_valence", 0.0),
            percept_topics=data.get("percept_topics", []),
            percept_text=data.get("percept_text", ""),
            percept_intent=data.get("percept_intent", "unknown"),
            current_time=data.get("current_time", time.time()),
            tick_count=data.get("tick_count", 0),
        )


# ── Integration Config ──────────────────────────────────────


@dataclass
class IntegrationConfig:
    """統合設定パラメータ。"""
    max_unified_units: int = 30
    max_output_candidates: int = 10
    max_duplicates: int = 20
    max_conflicts: int = 20
    max_reference_history: int = 100
    reference_decay_rate: float = 0.05
    topic_similarity_threshold: float = 0.3   # 重複検出閾値
    conflict_valence_threshold: float = 0.5   # 感情価矛盾閾値
    conflict_importance_threshold: float = 0.4  # 重要度差閾値
    conflict_hidden_restore_turns: int = 8     # 不可視化復元閾値
    single_source_cap: float = 0.6             # 単一出所上限比率
    recency_suppression_count: int = 3         # 直近再採用抑制
    episodic_certainty: float = 0.7            # 自己観測由来の基本確からしさ
    long_term_certainty: float = 0.6           # 長期要約由来の基本確からしさ
    binding_certainty: float = 0.5             # 感情結合由来の基本確からしさ
    temporal_immediate: float = 300.0          # 5分 (秒)
    temporal_recent: float = 3600.0            # 1時間
    temporal_medium: float = 86400.0           # 1日

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_unified_units": self.max_unified_units,
            "max_output_candidates": self.max_output_candidates,
            "max_duplicates": self.max_duplicates,
            "max_conflicts": self.max_conflicts,
            "max_reference_history": self.max_reference_history,
            "reference_decay_rate": self.reference_decay_rate,
            "topic_similarity_threshold": self.topic_similarity_threshold,
            "conflict_valence_threshold": self.conflict_valence_threshold,
            "conflict_importance_threshold": self.conflict_importance_threshold,
            "conflict_hidden_restore_turns": self.conflict_hidden_restore_turns,
            "single_source_cap": self.single_source_cap,
            "recency_suppression_count": self.recency_suppression_count,
            "episodic_certainty": self.episodic_certainty,
            "long_term_certainty": self.long_term_certainty,
            "binding_certainty": self.binding_certainty,
            "temporal_immediate": self.temporal_immediate,
            "temporal_recent": self.temporal_recent,
            "temporal_medium": self.temporal_medium,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationConfig:
        return cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})


# ── Integration State ───────────────────────────────────────


@dataclass
class IntegrationState:
    """記憶統合の永続化可能な状態。"""
    config: IntegrationConfig = field(default_factory=IntegrationConfig)
    duplicate_table: list[DuplicateEntry] = field(default_factory=list)
    conflict_table: list[ConflictEntry] = field(default_factory=list)
    reference_history: list[ReferenceHistoryEntry] = field(default_factory=list)
    reuse_history: dict[str, int] = field(default_factory=dict)  # unit_id → count
    turn_count: int = 0
    total_integrations: int = 0
    convergence_warnings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "duplicate_table": [d.to_dict() for d in self.duplicate_table],
            "conflict_table": [c.to_dict() for c in self.conflict_table],
            "reference_history": [r.to_dict() for r in self.reference_history],
            "reuse_history": dict(self.reuse_history),
            "turn_count": self.turn_count,
            "total_integrations": self.total_integrations,
            "convergence_warnings": self.convergence_warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationState:
        config_data = data.get("config", {})
        config = IntegrationConfig.from_dict(config_data) if isinstance(config_data, dict) else IntegrationConfig()
        return cls(
            config=config,
            duplicate_table=[DuplicateEntry.from_dict(d) for d in data.get("duplicate_table", [])],
            conflict_table=[ConflictEntry.from_dict(c) for c in data.get("conflict_table", [])],
            reference_history=[ReferenceHistoryEntry.from_dict(r) for r in data.get("reference_history", [])],
            reuse_history=data.get("reuse_history", {}),
            turn_count=data.get("turn_count", 0),
            total_integrations=data.get("total_integrations", 0),
            convergence_warnings=data.get("convergence_warnings", 0),
        )


# ── Integration Result ──────────────────────────────────────


@dataclass
class IntegrationResult:
    """統合出力。候補集合と付随情報。"""
    candidates: list[UnifiedMemoryUnit] = field(default_factory=list)
    duplicate_groups: list[DuplicateEntry] = field(default_factory=list)
    active_conflicts: list[ConflictEntry] = field(default_factory=list)
    source_distribution: dict[str, int] = field(default_factory=dict)
    convergence_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "duplicate_groups": [d.to_dict() for d in self.duplicate_groups],
            "active_conflicts": [c.to_dict() for c in self.active_conflicts],
            "source_distribution": dict(self.source_distribution),
            "convergence_warning": self.convergence_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationResult:
        return cls(
            candidates=[UnifiedMemoryUnit.from_dict(c) for c in data.get("candidates", [])],
            duplicate_groups=[DuplicateEntry.from_dict(d) for d in data.get("duplicate_groups", [])],
            active_conflicts=[ConflictEntry.from_dict(c) for c in data.get("active_conflicts", [])],
            source_distribution=data.get("source_distribution", {}),
            convergence_warning=data.get("convergence_warning", False),
        )

    def to_memory_list(self) -> list[dict[str, Any]]:
        """brain.py 互換の memory リスト形式に変換。"""
        return [c.to_memory_dict() for c in self.candidates]


# ── Normalization: Episodic → Unified ───────────────────────

def _make_unit_id(source: str, source_id: str) -> str:
    """一意IDを生成する。"""
    raw = f"{source}:{source_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _determine_temporal_phase(
    timestamp: float,
    current_time: float,
    config: IntegrationConfig,
) -> TemporalPhase:
    """タイムスタンプから時相を決定する。"""
    elapsed = current_time - timestamp
    if elapsed < 0:
        elapsed = 0
    if elapsed <= config.temporal_immediate:
        return TemporalPhase.IMMEDIATE
    elif elapsed <= config.temporal_recent:
        return TemporalPhase.RECENT
    elif elapsed <= config.temporal_medium:
        return TemporalPhase.MEDIUM
    else:
        return TemporalPhase.DISTANT


def normalize_episodic(
    episodes: Any,
    context: IntegrationContext,
    config: IntegrationConfig,
) -> list[UnifiedMemoryUnit]:
    """自己観測由来の出来事記録 → 共通記述単位。"""
    units: list[UnifiedMemoryUnit] = []

    entries = getattr(episodes, 'entries', None) or getattr(episodes, 'episodes', ())
    if not entries:
        return units

    for entry in entries:
        episode_id = getattr(entry, 'episode_id', '') or ''
        summary = getattr(entry, 'summary', '') or ''
        if not summary:
            continue

        topics = list(getattr(entry, 'topics', ()) or ())
        ts = getattr(entry, 'timestamp', 0.0) or 0.0
        vividness = getattr(entry, 'vividness', 0.5)
        ref_count = getattr(entry, 'reference_count', 0)

        # 重要度の正規化 (enum → float)
        importance_raw = getattr(entry, 'importance', None)
        importance_map = {"trivial": 0.1, "minor": 0.3, "moderate": 0.5, "notable": 0.7, "significant": 0.9}
        if importance_raw is not None:
            imp_str = importance_raw.value if hasattr(importance_raw, 'value') else str(importance_raw)
            importance = importance_map.get(imp_str, 0.5)
        else:
            importance = 0.5

        # 感情情報の抽出
        emotional_valence = 0.0
        emotional_label = ""
        companion = getattr(entry, 'emotional_companion', None)
        if companion is not None:
            emotional_valence = getattr(companion, 'valence', 0.0)
            emotional_label = getattr(companion, 'primary_emotion', '')

        phase = _determine_temporal_phase(ts, context.current_time, config)
        unit_id = _make_unit_id("episodic", episode_id)

        # 関連度: トピック一致 + 感情一致
        relevance = _compute_relevance(topics, emotional_valence, context)

        units.append(UnifiedMemoryUnit(
            unit_id=unit_id,
            source=MemorySource.EPISODIC,
            source_id=episode_id,
            summary=summary,
            topics=topics,
            temporal_phase=phase,
            timestamp=ts,
            certainty=config.episodic_certainty,
            relevance=relevance,
            reuse_count=ref_count,
            freshness=vividness if isinstance(vividness, (int, float)) else 0.5,
            emotional_valence=emotional_valence,
            emotional_label=emotional_label,
            importance=importance,
            original_data={"source": "episodic", "episode_id": episode_id},
        ))

    return units


# ── Normalization: Long-Term → Unified ──────────────────────

def normalize_long_term(
    memories: Optional[list[dict[str, Any]]],
    context: IntegrationContext,
    config: IntegrationConfig,
) -> list[UnifiedMemoryUnit]:
    """長期要約由来の圧縮記録 → 共通記述単位。"""
    units: list[UnifiedMemoryUnit] = []
    if not memories:
        return units

    for mem in memories:
        mem_id = str(mem.get("id", ""))
        summary = mem.get("summary", "")
        if not summary:
            continue

        keywords = mem.get("keywords", [])
        importance_raw = mem.get("importance", 3)
        importance = importance_raw / 5.0 if isinstance(importance_raw, (int, float)) else 0.6
        date_str = mem.get("date", "")
        last_recalled = mem.get("last_recalled", None)

        # タイムスタンプ推定
        ts = 0.0
        if date_str:
            try:
                ts = float(date_str)
            except (ValueError, TypeError):
                import datetime
                try:
                    dt = datetime.datetime.fromisoformat(str(date_str))
                    ts = dt.timestamp()
                except (ValueError, TypeError):
                    ts = 0.0

        phase = _determine_temporal_phase(ts, context.current_time, config)
        unit_id = _make_unit_id("long_term", mem_id)

        # 感情価の推定 (キーワードベース)
        emotional_valence = _estimate_valence_from_keywords(keywords)

        # 関連度
        relevance = _compute_relevance(keywords, emotional_valence, context)

        # 鮮度: 最終参照時刻ベース
        freshness = 0.5
        if last_recalled:
            try:
                last_ts = float(last_recalled) if isinstance(last_recalled, (int, float)) else 0.0
                elapsed = context.current_time - last_ts
                freshness = max(0.1, 1.0 - elapsed / (config.temporal_medium * 7))
            except (ValueError, TypeError):
                freshness = 0.4

        units.append(UnifiedMemoryUnit(
            unit_id=unit_id,
            source=MemorySource.LONG_TERM,
            source_id=mem_id,
            summary=summary,
            topics=keywords,
            temporal_phase=phase,
            timestamp=ts,
            certainty=config.long_term_certainty,
            relevance=relevance,
            freshness=freshness,
            emotional_valence=emotional_valence,
            importance=importance,
            original_data={"source": "long_term", "id": mem_id},
        ))

    return units


# ── Normalization: Binding → Unified ────────────────────────

def normalize_bindings(
    bindings: Any,
    context: IntegrationContext,
    config: IntegrationConfig,
) -> list[UnifiedMemoryUnit]:
    """感情記憶の紐づけ → 共通記述単位。"""
    units: list[UnifiedMemoryUnit] = []

    binding_list = getattr(bindings, 'bindings', ()) or ()
    if not binding_list:
        return units

    for binding in binding_list:
        binding_id = getattr(binding, 'binding_id', '') or ''
        summary = getattr(binding, 'memory_summary', '') or ''
        if not summary:
            continue

        freshness = getattr(binding, 'freshness', 0.5)
        ref_count = getattr(binding, 'reference_count', 0)

        # 感情トレースから感情情報を抽出
        traces = getattr(binding, 'traces', ()) or ()
        emotional_valence = 0.0
        emotional_label = ""
        if traces:
            # 最も強いトレース
            best_trace = max(traces, key=lambda t: getattr(t, 'intensity', 0.0) * getattr(t, 'freshness', 0.0))
            emotional_valence = getattr(best_trace, 'valence', 0.0)
            emotional_label = getattr(best_trace, 'emotion_label', '')

        ts_str = getattr(binding, 'creation_timestamp', '') or ''
        ts = 0.0
        if ts_str:
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(str(ts_str))
                ts = dt.timestamp()
            except (ValueError, TypeError):
                try:
                    ts = float(ts_str)
                except (ValueError, TypeError):
                    pass

        phase = _determine_temporal_phase(ts, context.current_time, config)
        unit_id = _make_unit_id("binding", binding_id)

        relevance = _compute_relevance([], emotional_valence, context)

        units.append(UnifiedMemoryUnit(
            unit_id=unit_id,
            source=MemorySource.BINDING,
            source_id=binding_id,
            summary=summary,
            topics=[],
            temporal_phase=phase,
            timestamp=ts,
            certainty=config.binding_certainty,
            relevance=relevance,
            reuse_count=ref_count,
            freshness=freshness if isinstance(freshness, (int, float)) else 0.5,
            emotional_valence=emotional_valence,
            emotional_label=emotional_label,
            importance=0.4,  # binding は補助的
            original_data={"source": "binding", "binding_id": binding_id},
        ))

    return units


def normalize_action_results(
    action_result_pairs: Optional[list[dict[str, Any]]],
    context: IntegrationContext,
    config: IntegrationConfig,
) -> list[UnifiedMemoryUnit]:
    """行動-結果対 → 共通記述単位。

    行動-結果対はエピソード記憶とは異なる抽象度の情報を保持する。
    エピソード記憶は「何が起きたか」の出来事単位の記録であり、
    行動-結果対は「何を選択した後にどのような変化が隣接したか」の選択-観測対である。
    二重記録の消去は行わず、同一経験の異なる視点として並立保持する。
    """
    units: list[UnifiedMemoryUnit] = []
    if not action_result_pairs:
        return units

    for pair_data in action_result_pairs:
        pair_id = pair_data.get("pair_id", "")
        if not pair_id:
            continue

        # 行動記述からサマリを構成
        action = pair_data.get("action", {})
        result = pair_data.get("result", {})
        policy_label = action.get("policy_label", "")
        context_str = action.get("selection_context", "")
        summary = f"action:{policy_label}"
        if context_str:
            summary += f" context:{context_str[:50]}"

        # 結果の断面記述からトピックを抽出
        topics: list[str] = []
        if policy_label:
            topics.append(policy_label)
        sections = result.get("sections", [])
        for sec in sections[:3]:
            sec_name = sec.get("section", "")
            if sec_name:
                topics.append(sec_name)

        # 感情差分（結果にemotion_diffがあれば）
        emotion_diff = result.get("emotion_diff", {})
        emotional_valence = 0.0
        emotional_label = ""
        if emotion_diff:
            vals = list(emotion_diff.values())
            if vals:
                emotional_valence = sum(vals) / len(vals)
                max_key = max(emotion_diff, key=lambda k: abs(emotion_diff[k]))
                emotional_label = max_key

        freshness = pair_data.get("freshness", 0.5)
        ts = pair_data.get("creation_time", 0.0)
        ref_count = pair_data.get("reference_count", 0)

        phase = _determine_temporal_phase(ts, context.current_time, config)
        unit_id = _make_unit_id("action_result", pair_id)

        relevance = _compute_relevance(topics, emotional_valence, context)

        units.append(UnifiedMemoryUnit(
            unit_id=unit_id,
            source=MemorySource.ACTION_RESULT,
            source_id=pair_id,
            summary=summary,
            topics=topics,
            temporal_phase=phase,
            timestamp=ts,
            certainty=0.5,  # 行動-結果対は時系列隣接であり因果確定ではない
            relevance=relevance,
            reuse_count=ref_count,
            freshness=freshness if isinstance(freshness, (int, float)) else 0.5,
            emotional_valence=emotional_valence,
            emotional_label=emotional_label,
            importance=0.35,  # 行動-結果対は参照補助的
            original_data={"source": "action_result", "pair_id": pair_id},
        ))

    return units


# ── Relevance Computation ───────────────────────────────────

def _compute_relevance(
    topics: list[str],
    emotional_valence: float,
    context: IntegrationContext,
) -> float:
    """現在の文脈との関連度を計算する。"""
    relevance = 0.0

    # トピック一致
    if topics and context.percept_topics:
        topic_set = {t.lower() for t in topics}
        context_set = {t.lower() for t in context.percept_topics}
        overlap = len(topic_set & context_set)
        if overlap > 0:
            relevance += min(1.0, overlap / max(1, len(context_set))) * 0.5

    # テキスト部分一致
    if context.percept_text and topics:
        text_lower = context.percept_text.lower()
        match_count = sum(1 for t in topics if t.lower() in text_lower)
        if match_count > 0:
            relevance += min(0.3, match_count * 0.1)

    # 感情一致 (mood congruence)
    if abs(context.mood_valence) > 0.1 and abs(emotional_valence) > 0.1:
        congruence = context.mood_valence * emotional_valence
        if congruence > 0:
            relevance += congruence * 0.2

    return min(1.0, relevance)


def _estimate_valence_from_keywords(keywords: list[str]) -> float:
    """キーワードから感情価を推定する。"""
    positive = {"楽しい", "嬉しい", "好き", "幸せ", "笑", "fun", "happy", "love", "joy", "good"}
    negative = {"悲しい", "怒り", "辛い", "嫌", "怖い", "sad", "angry", "fear", "bad", "hate"}
    score = 0.0
    for kw in keywords:
        kw_lower = kw.lower()
        if any(p in kw_lower for p in positive):
            score += 0.3
        if any(n in kw_lower for n in negative):
            score -= 0.3
    return max(-1.0, min(1.0, score))


# ── Topic Overlap Computation ───────────────────────────────

def _compute_topic_overlap(topics_a: list[str], topics_b: list[str]) -> float:
    """2つのトピックリスト間の重複率を計算する。"""
    if not topics_a or not topics_b:
        return 0.0
    set_a = {t.lower() for t in topics_a}
    set_b = {t.lower() for t in topics_b}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ── Duplicate Detection ────────────────────────────────────

def detect_duplicates(
    units: list[UnifiedMemoryUnit],
    config: IntegrationConfig,
) -> list[DuplicateEntry]:
    """同一事象の複数視点を検出する。統合消去ではなく並立保持。"""
    duplicates: list[DuplicateEntry] = []
    seen: set[str] = set()

    for i, a in enumerate(units):
        for j, b in enumerate(units):
            if j <= i:
                continue
            if a.source == b.source:
                continue  # 同一系統は重複扱いしない

            pair_key = f"{a.unit_id}:{b.unit_id}"
            if pair_key in seen:
                continue

            overlap = _compute_topic_overlap(a.topics, b.topics)
            # サマリーの部分一致チェック
            summary_sim = _compute_summary_similarity(a.summary, b.summary)
            combined = max(overlap, summary_sim)

            if combined >= config.topic_similarity_threshold:
                seen.add(pair_key)
                group_id = _make_unit_id("dup", f"{a.unit_id}-{b.unit_id}")
                duplicates.append(DuplicateEntry(
                    group_id=group_id,
                    unit_ids=[a.unit_id, b.unit_id],
                    sources=[a.source.value, b.source.value],
                    similarity=combined,
                    topic_overlap=overlap,
                ))

    return duplicates[:config.max_duplicates]


def _compute_summary_similarity(a: str, b: str) -> float:
    """2つの要約テキストの類似度を計算する。"""
    if not a or not b:
        return 0.0
    # 単語ベースの Jaccard 類似度
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


# ── Conflict Detection ─────────────────────────────────────

def detect_conflicts(
    units: list[UnifiedMemoryUnit],
    config: IntegrationConfig,
) -> list[ConflictEntry]:
    """矛盾を検出する。解消せず併存させ、揺らぎ情報として保持。"""
    conflicts: list[ConflictEntry] = []

    for i, a in enumerate(units):
        for j, b in enumerate(units):
            if j <= i:
                continue

            conflict_type = ""
            severity = 0.0

            # 感情価矛盾: 同一トピックだが感情方向が逆
            if _compute_topic_overlap(a.topics, b.topics) > 0.2:
                valence_diff = abs(a.emotional_valence - b.emotional_valence)
                if valence_diff >= config.conflict_valence_threshold:
                    conflict_type = "valence_mismatch"
                    severity = valence_diff

            # 重要度ギャップ: 同系統でないのに重要度が大きく異なる
            if not conflict_type and a.source != b.source:
                importance_diff = abs(a.importance - b.importance)
                if importance_diff >= config.conflict_importance_threshold:
                    overlap = _compute_topic_overlap(a.topics, b.topics)
                    if overlap > 0.15:
                        conflict_type = "importance_gap"
                        severity = importance_diff

            if conflict_type:
                conflict_id = _make_unit_id("conf", f"{a.unit_id}-{b.unit_id}")
                conflicts.append(ConflictEntry(
                    conflict_id=conflict_id,
                    unit_id_a=a.unit_id,
                    unit_id_b=b.unit_id,
                    conflict_type=conflict_type,
                    severity=severity,
                    visible=True,
                ))

    return conflicts[:config.max_conflicts]


# ── Ranking ─────────────────────────────────────────────────

def _rank_units(
    units: list[UnifiedMemoryUnit],
    reuse_history: dict[str, int],
    reference_history: list[ReferenceHistoryEntry],
    config: IntegrationConfig,
) -> list[UnifiedMemoryUnit]:
    """候補をスコアリングし並び替える。"""
    # 直近参照頻度マップ（再採用抑制用）
    recent_ref_count: dict[str, int] = {}
    for entry in reference_history[-20:]:
        recent_ref_count[entry.unit_id] = recent_ref_count.get(entry.unit_id, 0) + 1

    scored: list[tuple[float, UnifiedMemoryUnit]] = []
    for unit in units:
        score = 0.0
        # 関連度: 最重要
        score += unit.relevance * 3.0
        # 鮮度
        score += unit.freshness * 1.5
        # 重要度
        score += unit.importance * 1.0
        # 確からしさ
        score += unit.certainty * 0.5
        # 時相ボーナス
        phase_bonus = {
            TemporalPhase.IMMEDIATE: 1.0,
            TemporalPhase.RECENT: 0.5,
            TemporalPhase.MEDIUM: 0.2,
            TemporalPhase.DISTANT: 0.0,
        }
        score += phase_bonus.get(unit.temporal_phase, 0.0)

        # 直近再採用抑制
        recent = recent_ref_count.get(unit.unit_id, 0)
        if recent >= config.recency_suppression_count:
            score *= 0.5

        scored.append((score, unit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in scored]


# ── Source Diversity ────────────────────────────────────────

def _ensure_source_diversity(
    ranked: list[UnifiedMemoryUnit],
    config: IntegrationConfig,
) -> tuple[list[UnifiedMemoryUnit], bool]:
    """出所横断の混在提示を維持し、単一視点への収束を防ぐ。

    Returns:
        (diversified_list, convergence_warning)
    """
    if not ranked:
        return ranked, False

    max_count = config.max_output_candidates
    cap = config.single_source_cap

    selected: list[UnifiedMemoryUnit] = []
    source_counts: dict[str, int] = {}

    for unit in ranked:
        source_key = unit.source.value
        current_count = source_counts.get(source_key, 0)

        # 単一出所上限チェック
        total = len(selected)
        if total > 0 and current_count / total >= cap and total >= 3:
            continue  # この出所はこれ以上追加しない

        selected.append(unit)
        source_counts[source_key] = current_count + 1

        if len(selected) >= max_count:
            break

    # 収束警告チェック
    convergence = False
    if selected:
        max_source_ratio = max(source_counts.values()) / len(selected)
        if max_source_ratio > cap and len(selected) >= 3:
            convergence = True
            # 代替視点を補充
            remaining = [u for u in ranked if u not in selected]
            minority_sources = {s for s, c in source_counts.items() if c <= 1}
            for u in remaining:
                if u.source.value not in minority_sources:
                    continue
                selected.append(u)
                if len(selected) >= max_count + 2:
                    break

    return selected, convergence


# ── Conflict Health Check ───────────────────────────────────

def check_conflict_health(state: IntegrationState) -> None:
    """競合情報が長期にわたり不可視化された場合は再有効化する。"""
    for conflict in state.conflict_table:
        if not conflict.visible:
            conflict.turn_hidden += 1
            if conflict.turn_hidden >= state.config.conflict_hidden_restore_turns:
                conflict.visible = True
                conflict.turn_hidden = 0
                logger.debug(
                    "Conflict restored: %s (%s vs %s)",
                    conflict.conflict_id, conflict.unit_id_a, conflict.unit_id_b,
                )


# ── Main Integrator Class ──────────────────────────────────


class MemorySystemIntegrator:
    """記憶系統統合の主クラス。

    各記憶系統へのアクセスは読み取り中心。
    統合処理から元記憶への直接改変経路を持たない。
    出力は候補集合と付随情報に限定し、確定命令形式で渡さない。
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self._state = IntegrationState(config=config or IntegrationConfig())

    @property
    def state(self) -> IntegrationState:
        return self._state

    def integrate(
        self,
        episodes: Any = None,
        long_term_memories: Optional[list[dict[str, Any]]] = None,
        bindings: Any = None,
        context: Optional[IntegrationContext] = None,
        action_result_pairs: Optional[list[dict[str, Any]]] = None,
    ) -> IntegrationResult:
        """記憶を統合して参照候補を生成する。

        統合出力を同ターン内で再入力しない。
        直近参照結果の再採用を抑制し、未参照候補の再提示経路を保持する。

        Args:
            episodes: EpisodeStore (自己観測由来)
            long_term_memories: MemoryManager recall結果 (長期要約由来)
            bindings: BindingStore (感情記憶結合)
            context: 現在の文脈情報
            action_result_pairs: 行動-結果対の辞書リスト（選択-観測記録）

        Returns:
            IntegrationResult: 統合済み参照候補
        """
        self._state.turn_count += 1
        ctx = context or IntegrationContext()
        config = self._state.config

        # 1. 抽象化: 各系統 → 共通記述単位
        episodic_units = normalize_episodic(episodes, ctx, config)
        long_term_units = normalize_long_term(long_term_memories, ctx, config)
        binding_units = normalize_bindings(bindings, ctx, config)
        ar_units = normalize_action_results(action_result_pairs, ctx, config)

        all_units = episodic_units + long_term_units + binding_units + ar_units

        if not all_units:
            return IntegrationResult()

        # 2. 整列: 時相・関連度でソート（準備）
        all_units.sort(key=lambda u: (u.relevance, u.freshness), reverse=True)

        # 上限適用
        all_units = all_units[:config.max_unified_units]

        # 3. 重複調整: 同一事象の複数視点を並立保持
        duplicates = detect_duplicates(all_units, config)
        self._state.duplicate_table = duplicates

        # 4. 競合保持: 矛盾を解消せず併存
        conflicts = detect_conflicts(all_units, config)
        # 既存の不可視競合とマージ
        existing_ids = {c.conflict_id for c in self._state.conflict_table}
        for new_conflict in conflicts:
            if new_conflict.conflict_id not in existing_ids:
                self._state.conflict_table.append(new_conflict)
        # 上限適用
        self._state.conflict_table = self._state.conflict_table[-config.max_conflicts:]

        # 競合ヘルスチェック
        check_conflict_health(self._state)

        # 5. ランキング
        ranked = _rank_units(
            all_units,
            self._state.reuse_history,
            self._state.reference_history,
            config,
        )

        # 6. 出所多様性の確保
        selected, convergence_warning = _ensure_source_diversity(ranked, config)

        if convergence_warning:
            self._state.convergence_warnings += 1

        # 7. 参照履歴の更新
        self._update_reference_history(selected)

        # 8. 再利用履歴の更新
        for unit in selected:
            self._state.reuse_history[unit.unit_id] = \
                self._state.reuse_history.get(unit.unit_id, 0) + 1

        # 9. 出所分布の集計
        source_dist: dict[str, int] = {}
        for unit in selected:
            source_dist[unit.source.value] = source_dist.get(unit.source.value, 0) + 1

        # 10. 可視な競合のみ抽出
        visible_conflicts = [c for c in self._state.conflict_table if c.visible]

        self._state.total_integrations += len(selected)

        logger.debug(
            "Memory integration: %d candidates (ep=%d, lt=%d, bind=%d), "
            "duplicates=%d, conflicts=%d, convergence=%s",
            len(selected),
            len(episodic_units), len(long_term_units), len(binding_units),
            len(duplicates), len(visible_conflicts), convergence_warning,
        )

        return IntegrationResult(
            candidates=selected,
            duplicate_groups=duplicates,
            active_conflicts=visible_conflicts,
            source_distribution=source_dist,
            convergence_warning=convergence_warning,
        )

    def _update_reference_history(self, selected: list[UnifiedMemoryUnit]) -> None:
        """参照履歴を更新する。累積と希薄化を併置。"""
        config = self._state.config

        # 既存履歴の希薄化
        for entry in self._state.reference_history:
            entry.decay_factor *= (1.0 - config.reference_decay_rate)

        # 新規参照の追加
        for unit in selected:
            self._state.reference_history.append(ReferenceHistoryEntry(
                unit_id=unit.unit_id,
                turn=self._state.turn_count,
                relevance_at_ref=unit.relevance,
            ))

        # 上限適用
        if len(self._state.reference_history) > config.max_reference_history:
            self._state.reference_history = self._state.reference_history[-config.max_reference_history:]


# ── Factory ─────────────────────────────────────────────────

def create_integrator(
    config: Optional[IntegrationConfig] = None,
) -> MemorySystemIntegrator:
    """MemorySystemIntegrator のファクトリ関数。"""
    return MemorySystemIntegrator(config=config)


def create_config(**kwargs: Any) -> IntegrationConfig:
    """IntegrationConfig のファクトリ関数。"""
    return IntegrationConfig(**kwargs)


# ── Summary for Enrichment ──────────────────────────────────

def get_integration_summary(
    integrator: MemorySystemIntegrator,
) -> dict[str, Any]:
    """統合状態のサマリーを返す。"""
    state = integrator.state
    return {
        "turn_count": state.turn_count,
        "total_integrations": state.total_integrations,
        "duplicate_count": len(state.duplicate_table),
        "conflict_count": len([c for c in state.conflict_table if c.visible]),
        "convergence_warnings": state.convergence_warnings,
        "unique_reused": len(state.reuse_history),
    }


def get_integration_summary_text(
    integrator: MemorySystemIntegrator,
) -> str:
    """Prompt enrichment 用のテキストサマリー。"""
    summary = get_integration_summary(integrator)
    if summary["total_integrations"] == 0:
        return ""

    parts = []
    parts.append(f"統合記憶={summary['total_integrations']}件参照")
    if summary["duplicate_count"] > 0:
        parts.append(f"重複視点={summary['duplicate_count']}組")
    if summary["conflict_count"] > 0:
        parts.append(f"競合={summary['conflict_count']}件")
    if summary["convergence_warnings"] > 0:
        parts.append(f"収束警告={summary['convergence_warnings']}回")

    return ", ".join(parts)
