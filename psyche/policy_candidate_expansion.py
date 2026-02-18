"""
psyche/policy_candidate_expansion.py - Policy Candidate Expansion (LOCAL ONLY).

ポリシー候補拡張（内面反映経路の増設）

現在の候補集合（thought.py の6固定ポリシー）に対して、
内部状態の8断面の交差に基づく追加候補を生成する。

候補は常に複数の競合単位として扱い、単一方向の確定を避ける。
候補の優先は都度の状態差分に依存し、恒常的な優先順を持たない。
この機能は候補情報の形成に限定され、採択、正誤判定、価値判定、行動確定を行わない。
特定の表現様式や対話姿勢を恒久化しない。

入力参照断面:
  感情断面、記憶断面、傾向断面、責任断面、
  対話断面、自己観測断面、他者推定断面、目的断面

候補軸:
  接近、保留、探索、転換、維持、修復、境界調整、確認、委譲、内省反映

Usage::

    expander = create_expander()
    expanded = expander.expand_candidates(
        base_candidates=candidates,
        psyche_state=state,
        percept=percept,
        cross_section_inputs=inputs,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Cross-Section (入力断面) ─────────────────────────────────


class CrossSection(Enum):
    """入力参照の8断面。"""
    EMOTION = "emotion"              # 感情断面
    MEMORY = "memory"                # 記憶断面
    TENDENCY = "tendency"            # 傾向断面
    RESPONSIBILITY = "responsibility"  # 責任断面
    DIALOGUE = "dialogue"            # 対話断面
    SELF_OBSERVATION = "self_observation"  # 自己観測断面
    OTHER_ESTIMATION = "other_estimation"  # 他者推定断面
    PURPOSE = "purpose"              # 目的断面


# ── Policy Axis (候補軸) ────────────────────────────────────


class PolicyAxis(Enum):
    """候補の機能カテゴリ軸。固定列挙ではなく活性状態が変化する可変集合。"""
    APPROACH = "approach"            # 接近
    HOLD = "hold"                    # 保留
    EXPLORE = "explore"              # 探索
    SHIFT = "shift"                  # 転換
    MAINTAIN = "maintain"            # 維持
    REPAIR = "repair"                # 修復
    BOUNDARY = "boundary"            # 境界調整
    CONFIRM = "confirm"              # 確認
    DELEGATE = "delegate"            # 委譲
    REFLECT = "reflect"              # 内省反映


# ── Axis labels & fallback texts ────────────────────────────

_AXIS_LABELS: dict[PolicyAxis, str] = {
    PolicyAxis.APPROACH: "距離を詰める",
    PolicyAxis.HOLD: "様子を見る",
    PolicyAxis.EXPLORE: "別の角度から探る",
    PolicyAxis.SHIFT: "話の方向を変える",
    PolicyAxis.MAINTAIN: "今の調子を続ける",
    PolicyAxis.REPAIR: "関係を修復する",
    PolicyAxis.BOUNDARY: "境界を調整する",
    PolicyAxis.CONFIRM: "確認を取る",
    PolicyAxis.DELEGATE: "相手に委ねる",
    PolicyAxis.REFLECT: "内面を振り返る",
}

_AXIS_RATIONALES: dict[PolicyAxis, str] = {
    PolicyAxis.APPROACH: "内面の傾きが接近を示唆している",
    PolicyAxis.HOLD: "判断を保留し状況を見守る方が適切",
    PolicyAxis.EXPLORE: "好奇心や未知の要素が探索を促している",
    PolicyAxis.SHIFT: "現在の流れに違和感があり転換が浮上している",
    PolicyAxis.MAINTAIN: "安定した状態を維持する方向が自然",
    PolicyAxis.REPAIR: "関係や状態の修復が内面から求められている",
    PolicyAxis.BOUNDARY: "距離感の調整が必要と感じられている",
    PolicyAxis.CONFIRM: "不確かさが確認の必要性を示している",
    PolicyAxis.DELEGATE: "自分から手放すことが適切と感じている",
    PolicyAxis.REFLECT: "内省的な振り返りが浮上している",
}

_AXIS_FALLBACK_TEXT: dict[PolicyAxis, str] = {
    PolicyAxis.APPROACH: "...もう少し近づいてみようかな",
    PolicyAxis.HOLD: "...ちょっと待って、考えさせて",
    PolicyAxis.EXPLORE: "...ねえ、こういう見方はどう？",
    PolicyAxis.SHIFT: "...あ、そうだ、ちょっと別の話なんだけど",
    PolicyAxis.MAINTAIN: "...うん、このままでいいと思う",
    PolicyAxis.REPAIR: "...さっきのこと、ちゃんと言い直したいな",
    PolicyAxis.BOUNDARY: "...あたしはあたしのペースでいくね",
    PolicyAxis.CONFIRM: "...ちょっと確認したいんだけど、いい？",
    PolicyAxis.DELEGATE: "...あなたに任せるね",
    PolicyAxis.REFLECT: "...なんか、自分のこと考えちゃった",
}

_AXIS_DRIVE_CHANGES: dict[PolicyAxis, dict[str, float]] = {
    PolicyAxis.APPROACH: {"social": -0.08, "curiosity": -0.02, "expression": -0.03},
    PolicyAxis.HOLD: {"social": -0.02, "curiosity": -0.03, "expression": -0.01},
    PolicyAxis.EXPLORE: {"social": -0.03, "curiosity": -0.09, "expression": -0.02},
    PolicyAxis.SHIFT: {"social": -0.02, "curiosity": -0.07, "expression": -0.03},
    PolicyAxis.MAINTAIN: {"social": -0.03, "curiosity": -0.02, "expression": -0.02},
    PolicyAxis.REPAIR: {"social": -0.09, "curiosity": -0.01, "expression": -0.04},
    PolicyAxis.BOUNDARY: {"social": -0.04, "curiosity": -0.02, "expression": -0.05},
    PolicyAxis.CONFIRM: {"social": -0.05, "curiosity": -0.06, "expression": -0.02},
    PolicyAxis.DELEGATE: {"social": -0.06, "curiosity": -0.02, "expression": -0.01},
    PolicyAxis.REFLECT: {"social": -0.02, "curiosity": -0.04, "expression": -0.08},
}


# ── Input Fragment (特徴断片) ───────────────────────────────


@dataclass
class InputFragment:
    """各断面から抽出された特徴断片。

    共通の中間表現として断面を横断的に比較可能にする。
    """
    section: CrossSection
    key: str
    value: float  # -1.0 ~ 1.0
    confidence: float = 1.0  # 0.0 ~ 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section.value,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputFragment:
        try:
            section = CrossSection(data.get("section", "emotion"))
        except ValueError:
            section = CrossSection.EMOTION
        return cls(
            section=section,
            key=data.get("key", ""),
            value=data.get("value", 0.0),
            confidence=data.get("confidence", 1.0),
        )


# ── Expanded Policy Candidate ──────────────────────────────


@dataclass
class ExpandedCandidate:
    """拡張されたポリシー候補。

    発生根拠、成立条件、競合関係、抑制要因、再浮上要因を保持。
    """
    axis: PolicyAxis
    origin_sections: list[CrossSection]
    score: float = 0.0
    conditions: dict[str, float] = field(default_factory=dict)
    competing_axes: list[str] = field(default_factory=list)
    suppression_factors: list[str] = field(default_factory=list)
    resurface_factors: list[str] = field(default_factory=list)
    is_primary: bool = True
    created_at: float = field(default_factory=time.time)

    def to_policy_dict(self) -> dict[str, Any]:
        """thought.py 互換の候補dictに変換する。"""
        return {
            "policy_label": _AXIS_LABELS.get(self.axis, self.axis.value),
            "rationale": _AXIS_RATIONALES.get(self.axis, "内面状態の交差から生成"),
            "expected_drive_change": dict(
                _AXIS_DRIVE_CHANGES.get(self.axis, {"social": -0.03, "curiosity": -0.03, "expression": -0.03})
            ),
            "text": _AXIS_FALLBACK_TEXT.get(self.axis, "..."),
            "_score": self.score,
            "_axis": self.axis.value,
            "_origin_sections": [s.value for s in self.origin_sections],
            "_conditions": dict(self.conditions),
            "_competing": list(self.competing_axes),
            "_suppression_factors": list(self.suppression_factors),
            "_resurface_factors": list(self.resurface_factors),
            "_is_primary": self.is_primary,
            "_expanded": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "origin_sections": [s.value for s in self.origin_sections],
            "score": self.score,
            "conditions": dict(self.conditions),
            "competing_axes": list(self.competing_axes),
            "suppression_factors": list(self.suppression_factors),
            "resurface_factors": list(self.resurface_factors),
            "is_primary": self.is_primary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpandedCandidate:
        try:
            axis = PolicyAxis(data.get("axis", "approach"))
        except ValueError:
            axis = PolicyAxis.APPROACH
        sections = []
        for s in data.get("origin_sections", []):
            try:
                sections.append(CrossSection(s))
            except ValueError:
                pass
        return cls(
            axis=axis,
            origin_sections=sections,
            score=data.get("score", 0.0),
            conditions=data.get("conditions", {}),
            competing_axes=data.get("competing_axes", []),
            suppression_factors=data.get("suppression_factors", []),
            resurface_factors=data.get("resurface_factors", []),
            is_primary=data.get("is_primary", True),
            created_at=data.get("created_at", time.time()),
        )


# ── Candidate History Entry ─────────────────────────────────


@dataclass
class HistoryEntry:
    """候補履歴エントリ。残存と希薄化を併置。"""
    axis: str
    origin_sections: list[str]
    score: float
    turn: int
    decay_factor: float = 1.0  # 希薄化係数

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "origin_sections": list(self.origin_sections),
            "score": self.score,
            "turn": self.turn,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        return cls(
            axis=data.get("axis", ""),
            origin_sections=data.get("origin_sections", []),
            score=data.get("score", 0.0),
            turn=data.get("turn", 0),
            decay_factor=data.get("decay_factor", 1.0),
        )


# ── Suppression Entry ───────────────────────────────────────


@dataclass
class SuppressionEntry:
    """抑制履歴エントリ。可逆状態。"""
    axis: str
    reason: str
    strength: float  # 0.0 ~ 1.0
    turn_created: int
    turn_count: int = 0  # 連続抑制ターン数
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "reason": self.reason,
            "strength": self.strength,
            "turn_created": self.turn_created,
            "turn_count": self.turn_count,
            "released": self.released,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuppressionEntry:
        return cls(
            axis=data.get("axis", ""),
            reason=data.get("reason", ""),
            strength=data.get("strength", 0.0),
            turn_created=data.get("turn_created", 0),
            turn_count=data.get("turn_count", 0),
            released=data.get("released", False),
        )


# ── Competition Entry ───────────────────────────────────────


@dataclass
class CompetitionEntry:
    """競合履歴エントリ。未採択系列の保持。"""
    selected_axis: str
    unselected_axes: list[str]
    turn: int
    score_gap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_axis": self.selected_axis,
            "unselected_axes": list(self.unselected_axes),
            "turn": self.turn,
            "score_gap": self.score_gap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompetitionEntry:
        return cls(
            selected_axis=data.get("selected_axis", ""),
            unselected_axes=data.get("unselected_axes", []),
            turn=data.get("turn", 0),
            score_gap=data.get("score_gap", 0.0),
        )


# ── Configuration ───────────────────────────────────────────


@dataclass
class ExpansionConfig:
    """候補拡張の設定パラメータ。"""
    max_expanded_candidates: int = 5
    min_crossing_sections: int = 2  # 候補成立に必要な最低断面交差数
    history_max_entries: int = 50
    history_decay_rate: float = 0.05
    suppression_max_entries: int = 20
    suppression_chronic_threshold: int = 10  # 恒常化検知閾値
    suppression_release_threshold: float = 0.3  # 緩和閾値
    competition_max_entries: int = 30
    competition_reinject_boost: float = 0.15  # 未採択系列再注入ブースト
    fragment_history_max: int = 100
    single_section_dominance_cap: float = 0.6  # 単一断面支配上限
    linearization_warning_threshold: int = 1  # 単線化警告（競合系列数下限）

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_expanded_candidates": self.max_expanded_candidates,
            "min_crossing_sections": self.min_crossing_sections,
            "history_max_entries": self.history_max_entries,
            "history_decay_rate": self.history_decay_rate,
            "suppression_max_entries": self.suppression_max_entries,
            "suppression_chronic_threshold": self.suppression_chronic_threshold,
            "suppression_release_threshold": self.suppression_release_threshold,
            "competition_max_entries": self.competition_max_entries,
            "competition_reinject_boost": self.competition_reinject_boost,
            "fragment_history_max": self.fragment_history_max,
            "single_section_dominance_cap": self.single_section_dominance_cap,
            "linearization_warning_threshold": self.linearization_warning_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpansionConfig:
        return cls(
            max_expanded_candidates=data.get("max_expanded_candidates", 5),
            min_crossing_sections=data.get("min_crossing_sections", 2),
            history_max_entries=data.get("history_max_entries", 50),
            history_decay_rate=data.get("history_decay_rate", 0.05),
            suppression_max_entries=data.get("suppression_max_entries", 20),
            suppression_chronic_threshold=data.get("suppression_chronic_threshold", 10),
            suppression_release_threshold=data.get("suppression_release_threshold", 0.3),
            competition_max_entries=data.get("competition_max_entries", 30),
            competition_reinject_boost=data.get("competition_reinject_boost", 0.15),
            fragment_history_max=data.get("fragment_history_max", 100),
            single_section_dominance_cap=data.get("single_section_dominance_cap", 0.6),
            linearization_warning_threshold=data.get("linearization_warning_threshold", 1),
        )


# ── Expansion State ─────────────────────────────────────────


@dataclass
class ExpansionState:
    """ポリシー候補拡張の永続化可能な状態。"""
    config: ExpansionConfig = field(default_factory=ExpansionConfig)
    axis_activations: dict[str, float] = field(default_factory=dict)
    candidate_history: list[HistoryEntry] = field(default_factory=list)
    suppression_history: list[SuppressionEntry] = field(default_factory=list)
    competition_history: list[CompetitionEntry] = field(default_factory=list)
    fragment_history: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    total_expansions: int = 0
    linearization_warnings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "axis_activations": dict(self.axis_activations),
            "candidate_history": [h.to_dict() for h in self.candidate_history],
            "suppression_history": [s.to_dict() for s in self.suppression_history],
            "competition_history": [c.to_dict() for c in self.competition_history],
            "fragment_history": list(self.fragment_history),
            "turn_count": self.turn_count,
            "total_expansions": self.total_expansions,
            "linearization_warnings": self.linearization_warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpansionState:
        config_data = data.get("config", {})
        config = ExpansionConfig.from_dict(config_data) if isinstance(config_data, dict) else ExpansionConfig()
        return cls(
            config=config,
            axis_activations=data.get("axis_activations", {}),
            candidate_history=[
                HistoryEntry.from_dict(h) for h in data.get("candidate_history", [])
            ],
            suppression_history=[
                SuppressionEntry.from_dict(s) for s in data.get("suppression_history", [])
            ],
            competition_history=[
                CompetitionEntry.from_dict(c) for c in data.get("competition_history", [])
            ],
            fragment_history=data.get("fragment_history", []),
            turn_count=data.get("turn_count", 0),
            total_expansions=data.get("total_expansions", 0),
            linearization_warnings=data.get("linearization_warnings", 0),
        )


# ── Cross-Section Input Bundle ──────────────────────────────


@dataclass
class CrossSectionInputs:
    """orchestrator から供給される各断面の参照データ。入力は参照専用。"""
    # 感情断面
    emotions: dict[str, float] = field(default_factory=dict)
    mood_valence: float = 0.0
    mood_arousal: float = 0.3
    # 記憶断面
    recalled_count: int = 0
    has_emotional_bindings: bool = False
    episode_count: int = 0
    # 傾向断面
    tendency_count: int = 0
    dominant_tendency: str = ""
    tendency_strength: float = 0.0
    # 責任断面
    caution_bias: float = 0.0
    empathy_bias: float = 0.0
    dispersion_active: bool = False
    # 対話断面
    percept_intent: str = "unknown"
    percept_valence: float = 0.0
    percept_text_length: int = 0
    # 自己観測断面
    self_image_stability: float = 0.5
    coherence_level: float = 0.5
    strain_level: float = 0.0
    narrative_coherence: float = 0.5
    # 他者推定断面
    other_model_count: int = 0
    other_boundary_clarity: float = 0.5
    # 目的断面
    has_active_goal: bool = False
    goal_strength: float = 0.0
    motive_count: int = 0
    expectation_count: int = 0
    vector_count: int = 0
    # 行動-結果経験断面（参照情報形式）
    action_result_active_count: int = 0
    action_result_pattern_distribution: dict[str, int] = field(default_factory=dict)
    action_result_convergence_warning: bool = False
    # メタ感情変動候補断面（感情の推移特徴に基づく変動可能性の情報）
    meta_emotion_candidate_count: int = 0
    meta_emotion_pattern_count: int = 0
    meta_emotion_supply_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotions": dict(self.emotions),
            "mood_valence": self.mood_valence,
            "mood_arousal": self.mood_arousal,
            "recalled_count": self.recalled_count,
            "has_emotional_bindings": self.has_emotional_bindings,
            "episode_count": self.episode_count,
            "tendency_count": self.tendency_count,
            "dominant_tendency": self.dominant_tendency,
            "tendency_strength": self.tendency_strength,
            "caution_bias": self.caution_bias,
            "empathy_bias": self.empathy_bias,
            "dispersion_active": self.dispersion_active,
            "percept_intent": self.percept_intent,
            "percept_valence": self.percept_valence,
            "percept_text_length": self.percept_text_length,
            "self_image_stability": self.self_image_stability,
            "coherence_level": self.coherence_level,
            "strain_level": self.strain_level,
            "narrative_coherence": self.narrative_coherence,
            "other_model_count": self.other_model_count,
            "other_boundary_clarity": self.other_boundary_clarity,
            "has_active_goal": self.has_active_goal,
            "goal_strength": self.goal_strength,
            "motive_count": self.motive_count,
            "expectation_count": self.expectation_count,
            "vector_count": self.vector_count,
            "action_result_active_count": self.action_result_active_count,
            "action_result_pattern_distribution": dict(self.action_result_pattern_distribution),
            "action_result_convergence_warning": self.action_result_convergence_warning,
            "meta_emotion_candidate_count": self.meta_emotion_candidate_count,
            "meta_emotion_pattern_count": self.meta_emotion_pattern_count,
            "meta_emotion_supply_strength": self.meta_emotion_supply_strength,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossSectionInputs:
        return cls(
            emotions=data.get("emotions", {}),
            mood_valence=data.get("mood_valence", 0.0),
            mood_arousal=data.get("mood_arousal", 0.3),
            recalled_count=data.get("recalled_count", 0),
            has_emotional_bindings=data.get("has_emotional_bindings", False),
            episode_count=data.get("episode_count", 0),
            tendency_count=data.get("tendency_count", 0),
            dominant_tendency=data.get("dominant_tendency", ""),
            tendency_strength=data.get("tendency_strength", 0.0),
            caution_bias=data.get("caution_bias", 0.0),
            empathy_bias=data.get("empathy_bias", 0.0),
            dispersion_active=data.get("dispersion_active", False),
            percept_intent=data.get("percept_intent", "unknown"),
            percept_valence=data.get("percept_valence", 0.0),
            percept_text_length=data.get("percept_text_length", 0),
            self_image_stability=data.get("self_image_stability", 0.5),
            coherence_level=data.get("coherence_level", 0.5),
            strain_level=data.get("strain_level", 0.0),
            narrative_coherence=data.get("narrative_coherence", 0.5),
            other_model_count=data.get("other_model_count", 0),
            other_boundary_clarity=data.get("other_boundary_clarity", 0.5),
            has_active_goal=data.get("has_active_goal", False),
            goal_strength=data.get("goal_strength", 0.0),
            motive_count=data.get("motive_count", 0),
            expectation_count=data.get("expectation_count", 0),
            vector_count=data.get("vector_count", 0),
            action_result_active_count=data.get("action_result_active_count", 0),
            action_result_pattern_distribution=data.get("action_result_pattern_distribution", {}),
            action_result_convergence_warning=data.get("action_result_convergence_warning", False),
            meta_emotion_candidate_count=data.get("meta_emotion_candidate_count", 0),
            meta_emotion_pattern_count=data.get("meta_emotion_pattern_count", 0),
            meta_emotion_supply_strength=data.get("meta_emotion_supply_strength", 0.0),
        )


# ── Fragment Extraction ─────────────────────────────────────

def _extract_emotion_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """感情断面 → 特徴断片。"""
    fragments = []
    for emotion, val in inputs.emotions.items():
        if val > 0.1:
            fragments.append(InputFragment(
                section=CrossSection.EMOTION,
                key=f"emotion_{emotion}",
                value=val,
                confidence=min(1.0, val * 1.5),
            ))
    if abs(inputs.mood_valence) > 0.1:
        fragments.append(InputFragment(
            section=CrossSection.EMOTION,
            key="mood_valence",
            value=inputs.mood_valence,
        ))
    if inputs.mood_arousal > 0.4:
        fragments.append(InputFragment(
            section=CrossSection.EMOTION,
            key="mood_arousal",
            value=inputs.mood_arousal,
        ))
    return fragments


def _extract_memory_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """記憶断面 → 特徴断片。"""
    fragments = []
    if inputs.recalled_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.MEMORY,
            key="recall_richness",
            value=min(1.0, inputs.recalled_count / 5.0),
        ))
    if inputs.has_emotional_bindings:
        fragments.append(InputFragment(
            section=CrossSection.MEMORY,
            key="emotional_binding",
            value=0.7,
        ))
    if inputs.episode_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.MEMORY,
            key="episode_presence",
            value=min(1.0, inputs.episode_count / 10.0),
        ))
    return fragments


def _extract_tendency_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """傾向断面 → 特徴断片。"""
    fragments = []
    if inputs.tendency_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.TENDENCY,
            key="tendency_presence",
            value=min(1.0, inputs.tendency_count / 5.0),
        ))
    if inputs.tendency_strength > 0.2:
        fragments.append(InputFragment(
            section=CrossSection.TENDENCY,
            key="tendency_strength",
            value=inputs.tendency_strength,
        ))
    return fragments


def _extract_responsibility_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """責任断面 → 特徴断片。"""
    fragments = []
    if inputs.caution_bias > 0.1:
        fragments.append(InputFragment(
            section=CrossSection.RESPONSIBILITY,
            key="caution",
            value=inputs.caution_bias,
        ))
    if inputs.empathy_bias > 0.1:
        fragments.append(InputFragment(
            section=CrossSection.RESPONSIBILITY,
            key="empathy",
            value=inputs.empathy_bias,
        ))
    if inputs.dispersion_active:
        fragments.append(InputFragment(
            section=CrossSection.RESPONSIBILITY,
            key="dispersion_active",
            value=0.5,
        ))
    return fragments


def _extract_dialogue_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """対話断面 → 特徴断片。"""
    fragments = []
    intent_signals: dict[str, float] = {
        "question": 0.7,
        "sharing": 0.5,
        "complaint": -0.5,
        "greeting": 0.3,
        "joke": 0.6,
        "farewell": -0.3,
    }
    intent_val = intent_signals.get(inputs.percept_intent, 0.0)
    if abs(intent_val) > 0.01:
        fragments.append(InputFragment(
            section=CrossSection.DIALOGUE,
            key=f"intent_{inputs.percept_intent}",
            value=intent_val,
        ))
    if abs(inputs.percept_valence) > 0.1:
        fragments.append(InputFragment(
            section=CrossSection.DIALOGUE,
            key="percept_valence",
            value=inputs.percept_valence,
        ))
    if inputs.percept_text_length > 50:
        fragments.append(InputFragment(
            section=CrossSection.DIALOGUE,
            key="text_richness",
            value=min(1.0, inputs.percept_text_length / 200.0),
        ))
    return fragments


def _extract_self_observation_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """自己観測断面 → 特徴断片。"""
    fragments = []
    if inputs.self_image_stability < 0.4:
        fragments.append(InputFragment(
            section=CrossSection.SELF_OBSERVATION,
            key="self_image_instability",
            value=1.0 - inputs.self_image_stability,
        ))
    if inputs.coherence_level < 0.4:
        fragments.append(InputFragment(
            section=CrossSection.SELF_OBSERVATION,
            key="coherence_low",
            value=1.0 - inputs.coherence_level,
        ))
    if inputs.strain_level > 0.3:
        fragments.append(InputFragment(
            section=CrossSection.SELF_OBSERVATION,
            key="strain",
            value=inputs.strain_level,
        ))
    if inputs.narrative_coherence < 0.4:
        fragments.append(InputFragment(
            section=CrossSection.SELF_OBSERVATION,
            key="narrative_fragmentation",
            value=1.0 - inputs.narrative_coherence,
        ))
    return fragments


def _extract_other_estimation_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """他者推定断面 → 特徴断片。"""
    fragments = []
    if inputs.other_model_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.OTHER_ESTIMATION,
            key="other_model_presence",
            value=min(1.0, inputs.other_model_count / 3.0),
        ))
    if inputs.other_boundary_clarity < 0.4:
        fragments.append(InputFragment(
            section=CrossSection.OTHER_ESTIMATION,
            key="boundary_blur",
            value=1.0 - inputs.other_boundary_clarity,
        ))
    return fragments


def _extract_purpose_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """目的断面 → 特徴断片。"""
    fragments = []
    if inputs.has_active_goal:
        fragments.append(InputFragment(
            section=CrossSection.PURPOSE,
            key="active_goal",
            value=inputs.goal_strength,
        ))
    if inputs.motive_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.PURPOSE,
            key="motive_presence",
            value=min(1.0, inputs.motive_count / 5.0),
        ))
    if inputs.expectation_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.PURPOSE,
            key="expectation_presence",
            value=min(1.0, inputs.expectation_count / 5.0),
        ))
    if inputs.vector_count > 0:
        fragments.append(InputFragment(
            section=CrossSection.PURPOSE,
            key="direction_presence",
            value=min(1.0, inputs.vector_count / 3.0),
        ))
    return fragments


def extract_all_fragments(inputs: CrossSectionInputs) -> list[InputFragment]:
    """全8断面から特徴断片を抽出する。"""
    extractors = [
        _extract_emotion_fragments,
        _extract_memory_fragments,
        _extract_tendency_fragments,
        _extract_responsibility_fragments,
        _extract_dialogue_fragments,
        _extract_self_observation_fragments,
        _extract_other_estimation_fragments,
        _extract_purpose_fragments,
    ]
    all_fragments: list[InputFragment] = []
    for extractor in extractors:
        all_fragments.extend(extractor(inputs))
    return all_fragments


# ── Fragment Unification ────────────────────────────────────

def _unify_fragments(
    fragments: list[InputFragment],
) -> dict[CrossSection, list[InputFragment]]:
    """特徴断片を断面ごとに整理し、重複を除去する。"""
    unified: dict[CrossSection, list[InputFragment]] = {}
    seen_keys: set[str] = set()

    for frag in fragments:
        compound_key = f"{frag.section.value}:{frag.key}"
        if compound_key in seen_keys:
            continue
        seen_keys.add(compound_key)
        if frag.section not in unified:
            unified[frag.section] = []
        unified[frag.section].append(frag)

    return unified


# ── Axis Activation ─────────────────────────────────────────

# 断面の特徴断片キー → 軸への寄与マッピング
_AXIS_ACTIVATION_RULES: dict[PolicyAxis, list[tuple[str, str, float]]] = {
    # (section_value, fragment_key_prefix, weight)
    PolicyAxis.APPROACH: [
        ("emotion", "emotion_love", 1.5),
        ("emotion", "emotion_joy", 1.0),
        ("dialogue", "intent_sharing", 0.8),
        ("purpose", "active_goal", 0.7),
        ("responsibility", "empathy", 1.2),
    ],
    PolicyAxis.HOLD: [
        ("responsibility", "caution", 1.5),
        ("self_observation", "strain", 1.0),
        ("emotion", "emotion_fear", 1.2),
        ("other_estimation", "boundary_blur", 0.8),
    ],
    PolicyAxis.EXPLORE: [
        ("emotion", "emotion_surprise", 1.0),
        ("dialogue", "intent_question", 1.3),
        ("purpose", "direction_presence", 0.9),
        ("memory", "episode_presence", 0.6),
        ("purpose", "motive_presence", 0.8),
    ],
    PolicyAxis.SHIFT: [
        ("self_observation", "narrative_fragmentation", 1.0),
        ("emotion", "emotion_fun", 0.8),
        ("dialogue", "text_richness", 0.5),
        ("tendency", "tendency_presence", -0.5),  # 傾向があると転換しにくい
    ],
    PolicyAxis.MAINTAIN: [
        ("self_observation", "self_image_instability", -1.0),  # 不安定だと維持しない
        ("tendency", "tendency_strength", 1.2),
        ("emotion", "mood_valence", 0.8),
        ("purpose", "expectation_presence", 0.7),
    ],
    PolicyAxis.REPAIR: [
        ("emotion", "emotion_sorrow", 1.5),
        ("dialogue", "percept_valence", -1.2),  # negative valence → repair
        ("responsibility", "empathy", 1.0),
        ("self_observation", "strain", 0.8),
        ("memory", "emotional_binding", 0.6),
    ],
    PolicyAxis.BOUNDARY: [
        ("other_estimation", "boundary_blur", 1.5),
        ("responsibility", "caution", 0.8),
        ("emotion", "emotion_anger", 1.0),
        ("self_observation", "coherence_low", 0.7),
    ],
    PolicyAxis.CONFIRM: [
        ("dialogue", "intent_question", 0.8),
        ("other_estimation", "other_model_presence", 1.0),
        ("self_observation", "coherence_low", 0.9),
        ("purpose", "expectation_presence", 0.7),
    ],
    PolicyAxis.DELEGATE: [
        ("responsibility", "dispersion_active", 1.2),
        ("emotion", "emotion_fear", 0.8),
        ("other_estimation", "other_model_presence", 0.7),
        ("purpose", "active_goal", -0.5),  # ゴールがあると委譲しにくい
    ],
    PolicyAxis.REFLECT: [
        ("self_observation", "strain", 1.3),
        ("self_observation", "self_image_instability", 1.0),
        ("self_observation", "narrative_fragmentation", 0.8),
        ("emotion", "mood_arousal", -0.5),  # 低覚醒で内省しやすい
        ("memory", "recall_richness", 0.6),
    ],
}


def _compute_axis_activations(
    unified: dict[CrossSection, list[InputFragment]],
    history_fragments: list[dict[str, Any]],
) -> dict[str, float]:
    """断面組成から候補軸の活性度を再決定する。

    静的優先列を持たず、都度の断面組成から再決定。
    履歴断片は揺らぎ源として参照するが、単回結果の直写は行わない。
    """
    # Build flat lookup: "section:key" → fragment
    flat_lookup: dict[str, InputFragment] = {}
    for section_fragments in unified.values():
        for frag in section_fragments:
            flat_lookup[f"{frag.section.value}:{frag.key}"] = frag

    activations: dict[str, float] = {}

    for axis, rules in _AXIS_ACTIVATION_RULES.items():
        activation = 0.0
        contributing_sections: set[str] = set()

        for section_val, key_prefix, weight in rules:
            # Find matching fragments
            for compound_key, frag in flat_lookup.items():
                if compound_key.startswith(f"{section_val}:{key_prefix}"):
                    contribution = frag.value * frag.confidence * weight
                    activation += contribution
                    contributing_sections.add(section_val)

        # Apply history perturbation (揺らぎ源)
        if history_fragments:
            history_noise = _compute_history_noise(axis.value, history_fragments)
            activation += history_noise * 0.1  # 揺らぎは微小

        activations[axis.value] = activation

    return activations


def _compute_history_noise(axis_value: str, history_fragments: list[dict[str, Any]]) -> float:
    """履歴断片から揺らぎを計算する。直写ではなく要約断片経由。"""
    relevant = [
        h for h in history_fragments
        if h.get("key", "").startswith(axis_value[:3])
    ]
    if not relevant:
        return 0.0
    avg = sum(h.get("value", 0.0) for h in relevant) / len(relevant)
    # 揺らぎは過去の平均から微小な偏差
    return avg * 0.3


# ── Candidate Generation from Crossings ─────────────────────

def _find_crossing_sections(
    axis: PolicyAxis,
    unified: dict[CrossSection, list[InputFragment]],
) -> list[CrossSection]:
    """ある軸に対して寄与する断面を特定する。"""
    rules = _AXIS_ACTIVATION_RULES.get(axis, [])
    contributing: set[CrossSection] = set()

    for section_val, key_prefix, _ in rules:
        try:
            section = CrossSection(section_val)
        except ValueError:
            continue
        if section not in unified:
            continue
        for frag in unified[section]:
            if frag.key.startswith(key_prefix):
                contributing.add(section)

    return list(contributing)


def _generate_crossing_candidates(
    activations: dict[str, float],
    unified: dict[CrossSection, list[InputFragment]],
    config: ExpansionConfig,
    suppression_map: dict[str, float],
    competition_boost: dict[str, float],
) -> list[ExpandedCandidate]:
    """複数断面の交差で候補を生成する。

    単一断面起点ではなく複数断面の交差で成立させる。
    同一断面の単独支配を抑える。
    """
    candidates: list[ExpandedCandidate] = []

    # 活性度でソートし、上位から候補生成を試行
    sorted_axes = sorted(activations.items(), key=lambda x: x[1], reverse=True)

    for axis_val, activation in sorted_axes:
        if activation <= 0.0:
            continue

        try:
            axis = PolicyAxis(axis_val)
        except ValueError:
            continue

        # 断面交差を確認
        crossing_sections = _find_crossing_sections(axis, unified)
        if len(crossing_sections) < config.min_crossing_sections:
            continue

        # 抑制チェック
        suppression = suppression_map.get(axis_val, 0.0)
        effective_score = activation * (1.0 - suppression)

        # 競合再注入ブースト
        boost = competition_boost.get(axis_val, 0.0)
        effective_score += boost

        # 単一断面支配チェック
        section_contributions = _compute_section_contributions(axis, crossing_sections, unified)
        max_contribution = max(section_contributions.values()) if section_contributions else 0.0
        total_contribution = sum(section_contributions.values()) if section_contributions else 1.0
        if total_contribution > 0 and max_contribution / total_contribution > config.single_section_dominance_cap:
            # 単一断面が支配的 → スコア減衰
            effective_score *= 0.7

        # 成立条件
        conditions = {
            f"{s.value}_contribution": section_contributions.get(s.value, 0.0)
            for s in crossing_sections
        }

        # 競合軸の特定
        competing = [
            a for a, v in sorted_axes
            if a != axis_val and v > activation * 0.5
        ]

        # 抑制要因
        suppression_factors = []
        if suppression > 0.1:
            suppression_factors.append(f"history_suppression({suppression:.2f})")
        if max_contribution / total_contribution > config.single_section_dominance_cap:
            suppression_factors.append("single_section_dominance")

        # 再浮上要因
        resurface_factors = []
        if boost > 0.01:
            resurface_factors.append(f"competition_reinject({boost:.2f})")

        candidate = ExpandedCandidate(
            axis=axis,
            origin_sections=crossing_sections,
            score=max(0.0, effective_score),
            conditions=conditions,
            competing_axes=competing[:3],
            suppression_factors=suppression_factors,
            resurface_factors=resurface_factors,
            is_primary=True,
        )
        candidates.append(candidate)

    return candidates


def _compute_section_contributions(
    axis: PolicyAxis,
    sections: list[CrossSection],
    unified: dict[CrossSection, list[InputFragment]],
) -> dict[str, float]:
    """各断面が軸にどれだけ寄与しているかを計算する。"""
    rules = _AXIS_ACTIVATION_RULES.get(axis, [])
    contributions: dict[str, float] = {}

    for section in sections:
        total = 0.0
        for section_val, key_prefix, weight in rules:
            if section_val != section.value:
                continue
            if section not in unified:
                continue
            for frag in unified[section]:
                if frag.key.startswith(key_prefix):
                    total += abs(frag.value * frag.confidence * weight)
        contributions[section.value] = total

    return contributions


# ── Ensure Competition (単線化防止) ──────────────────────────

def _ensure_competition(
    candidates: list[ExpandedCandidate],
    activations: dict[str, float],
    unified: dict[CrossSection, list[InputFragment]],
    config: ExpansionConfig,
) -> list[ExpandedCandidate]:
    """候補集合に競合系列がない場合は代替候補を補充する。"""
    if len(candidates) > config.linearization_warning_threshold:
        return candidates

    # 単線化状態 → 代替系列を補充
    existing_axes = {c.axis.value for c in candidates}

    for axis in PolicyAxis:
        if axis.value in existing_axes:
            continue

        activation = activations.get(axis.value, 0.0)
        if activation <= -1.0:
            continue

        # 交差条件を緩和して代替候補を生成（min_crossing_sections - 1）
        crossing = _find_crossing_sections(axis, unified)
        if len(crossing) < max(1, config.min_crossing_sections - 1):
            continue

        alt = ExpandedCandidate(
            axis=axis,
            origin_sections=crossing,
            score=max(0.01, activation * 0.5),
            is_primary=False,
            resurface_factors=["linearization_supplement"],
        )
        candidates.append(alt)

        if len(candidates) > config.linearization_warning_threshold:
            break

    return candidates


# ── Suppression Health Check ────────────────────────────────

def _check_suppression_health(state: ExpansionState) -> None:
    """抑制履歴が恒常化する兆候を検知し、抑制状態を緩和する。"""
    config = state.config

    for entry in state.suppression_history:
        if entry.released:
            continue
        if entry.turn_count >= config.suppression_chronic_threshold:
            # 恒常化兆候 → 抑制緩和
            entry.strength *= (1.0 - config.suppression_release_threshold)
            if entry.strength < 0.05:
                entry.released = True
            logger.debug(
                "Suppression relaxed: axis=%s, strength=%.2f, released=%s",
                entry.axis, entry.strength, entry.released,
            )


# ── Main Expander Class ─────────────────────────────────────

class PolicyCandidateExpander:
    """ポリシー候補拡張の主クラス。

    入力は参照専用とし、上流状態の更新経路を持たない。
    出力は候補集合と付随根拠に限定し、確定命令形式で渡さない。
    """

    def __init__(self, config: Optional[ExpansionConfig] = None):
        self._state = ExpansionState(config=config or ExpansionConfig())

    @property
    def state(self) -> ExpansionState:
        return self._state

    def expand_candidates(
        self,
        base_candidates: list[dict[str, Any]],
        inputs: CrossSectionInputs,
    ) -> list[dict[str, Any]]:
        """候補を拡張する。

        base_candidates を参照しつつ、新たな候補を追加する。
        直近出力の全文再投入を禁止し、履歴は要約断片経由でのみ再利用する。

        Args:
            base_candidates: thought.py 由来の既存候補
            inputs: 各断面の参照データ

        Returns:
            拡張された候補のリスト（thought.py dict形式）
        """
        self._state.turn_count += 1

        # 1. 全断面から特徴断片を抽出
        fragments = extract_all_fragments(inputs)

        # 2. 重複・矛盾整理 → 中間表現
        unified = _unify_fragments(fragments)

        # 3. 候補軸の活性度を断面組成から再決定
        activations = _compute_axis_activations(
            unified, self._state.fragment_history,
        )
        self._state.axis_activations = activations

        # 4. 抑制マップ構築
        suppression_map = self._build_suppression_map()

        # 5. 競合再注入ブースト構築
        competition_boost = self._build_competition_boost()

        # 6. 直近採択系列の過度再選択を抑制
        self._apply_recency_suppression(base_candidates)

        # 7. 複数断面の交差で候補生成
        expanded = _generate_crossing_candidates(
            activations, unified, self._state.config,
            suppression_map, competition_boost,
        )

        # 8. 競合系列の確保（単線化防止）
        expanded = _ensure_competition(
            expanded, activations, unified, self._state.config,
        )
        if len(expanded) <= self._state.config.linearization_warning_threshold:
            self._state.linearization_warnings += 1

        # 9. 上位候補を選出（主系列 + 代替系列）
        expanded.sort(key=lambda c: c.score, reverse=True)
        primary = [c for c in expanded if c.is_primary]
        alternatives = [c for c in expanded if not c.is_primary]
        selected = primary[:self._state.config.max_expanded_candidates]
        remaining_slots = self._state.config.max_expanded_candidates - len(selected)
        if remaining_slots > 0:
            selected.extend(alternatives[:remaining_slots])

        # 10. 履歴更新
        self._update_histories(selected, activations)

        # 11. 抑制恒常化チェック
        _check_suppression_health(self._state)

        # 12. 断片履歴更新（揺らぎ源）
        self._update_fragment_history(fragments)

        # 13. thought.py 互換 dict に変換
        result = [c.to_policy_dict() for c in selected]

        self._state.total_expansions += len(result)
        logger.debug(
            "Policy expansion: %d candidates generated (turn=%d, axes=%s)",
            len(result), self._state.turn_count,
            [c.axis.value for c in selected],
        )

        return result

    def _build_suppression_map(self) -> dict[str, float]:
        """現在の抑制状態からマップを構築する。"""
        suppression: dict[str, float] = {}
        for entry in self._state.suppression_history:
            if entry.released:
                continue
            current = suppression.get(entry.axis, 0.0)
            suppression[entry.axis] = max(current, entry.strength)
        return suppression

    def _build_competition_boost(self) -> dict[str, float]:
        """競合履歴から未採択系列の再注入ブーストを計算する。"""
        boost: dict[str, float] = {}
        config = self._state.config

        # 直近の競合履歴のみ参照
        recent = self._state.competition_history[-5:] if self._state.competition_history else []
        for entry in recent:
            for axis in entry.unselected_axes:
                current = boost.get(axis, 0.0)
                boost[axis] = current + config.competition_reinject_boost

        return boost

    def _apply_recency_suppression(self, base_candidates: list[dict[str, Any]]) -> None:
        """直近採択系列の過度再選択を抑制する。

        候補履歴で連続反復していた軸に軽い抑制をかける。
        """
        recent_history = self._state.candidate_history[-5:]
        axis_counts: dict[str, int] = {}
        for entry in recent_history:
            axis_counts[entry.axis] = axis_counts.get(entry.axis, 0) + 1

        for axis_val, count in axis_counts.items():
            if count >= 3:
                # 3回以上連続 → 軽い抑制
                existing = [
                    s for s in self._state.suppression_history
                    if s.axis == axis_val and not s.released
                ]
                if not existing:
                    self._state.suppression_history.append(SuppressionEntry(
                        axis=axis_val,
                        reason="recency_repetition",
                        strength=0.3,
                        turn_created=self._state.turn_count,
                    ))

    def _update_histories(
        self,
        selected: list[ExpandedCandidate],
        activations: dict[str, float],
    ) -> None:
        """候補・抑制・競合履歴を更新する。"""
        config = self._state.config

        # 候補履歴: 残存と希薄化を併置
        for entry in self._state.candidate_history:
            entry.decay_factor *= (1.0 - config.history_decay_rate)

        for candidate in selected:
            self._state.candidate_history.append(HistoryEntry(
                axis=candidate.axis.value,
                origin_sections=[s.value for s in candidate.origin_sections],
                score=candidate.score,
                turn=self._state.turn_count,
            ))

        # 候補履歴上限
        if len(self._state.candidate_history) > config.history_max_entries:
            self._state.candidate_history = self._state.candidate_history[-config.history_max_entries:]

        # 抑制履歴のターンカウント更新
        for entry in self._state.suppression_history:
            if not entry.released:
                entry.turn_count += 1

        # 抑制履歴上限
        active_suppressions = [s for s in self._state.suppression_history if not s.released]
        released_suppressions = [s for s in self._state.suppression_history if s.released]
        if len(active_suppressions) > config.suppression_max_entries:
            active_suppressions = active_suppressions[-config.suppression_max_entries:]
        self._state.suppression_history = active_suppressions + released_suppressions[-5:]

        # 競合履歴
        if len(selected) >= 2:
            best = selected[0]
            others = [c.axis.value for c in selected[1:]]
            gap = selected[0].score - selected[1].score if len(selected) >= 2 else 0.0
            self._state.competition_history.append(CompetitionEntry(
                selected_axis=best.axis.value,
                unselected_axes=others,
                turn=self._state.turn_count,
                score_gap=gap,
            ))

        # 競合履歴上限
        if len(self._state.competition_history) > config.competition_max_entries:
            self._state.competition_history = self._state.competition_history[-config.competition_max_entries:]

    def _update_fragment_history(self, fragments: list[InputFragment]) -> None:
        """断片履歴を更新する。次回生成時の揺らぎ源として参照。"""
        config = self._state.config

        # 要約断片のみ保存（全文再投入を禁止）
        summary = [
            {"key": f.key, "value": f.value, "section": f.section.value}
            for f in fragments[:20]  # 代表的な断片のみ
        ]
        self._state.fragment_history.extend(summary)

        if len(self._state.fragment_history) > config.fragment_history_max:
            self._state.fragment_history = self._state.fragment_history[-config.fragment_history_max:]


# ── Factory ─────────────────────────────────────────────────

def create_expander(
    config: Optional[ExpansionConfig] = None,
) -> PolicyCandidateExpander:
    """PolicyCandidateExpander のファクトリ関数。"""
    return PolicyCandidateExpander(config=config)


def create_config(**kwargs: Any) -> ExpansionConfig:
    """ExpansionConfig のファクトリ関数。"""
    return ExpansionConfig(**kwargs)


# ── Summary for Enrichment ──────────────────────────────────

def get_expansion_summary(
    expander: PolicyCandidateExpander,
) -> dict[str, Any]:
    """候補拡張状態のサマリーを返す（get_prompt_enrichment 用）。"""
    state = expander.state
    active_axes = [
        axis for axis, val in state.axis_activations.items()
        if val > 0.0
    ]
    active_suppressions = [
        s for s in state.suppression_history
        if not s.released
    ]

    return {
        "turn_count": state.turn_count,
        "total_expansions": state.total_expansions,
        "active_axes_count": len(active_axes),
        "active_axes": active_axes[:5],
        "suppression_count": len(active_suppressions),
        "linearization_warnings": state.linearization_warnings,
        "top_axis": max(state.axis_activations, key=state.axis_activations.get) if state.axis_activations else "",
        "top_activation": max(state.axis_activations.values()) if state.axis_activations else 0.0,
    }


def get_expansion_summary_text(
    expander: PolicyCandidateExpander,
) -> str:
    """Prompt enrichment 用のテキストサマリー。"""
    summary = get_expansion_summary(expander)
    if summary["active_axes_count"] == 0:
        return ""

    parts = []
    if summary["top_axis"]:
        label = _AXIS_LABELS.get(
            PolicyAxis(summary["top_axis"]) if summary["top_axis"] in [a.value for a in PolicyAxis] else PolicyAxis.APPROACH,
            summary["top_axis"],
        )
        parts.append(f"主軸={label}({summary['top_activation']:.2f})")
    parts.append(f"活性軸={summary['active_axes_count']}個")
    if summary["suppression_count"] > 0:
        parts.append(f"抑制={summary['suppression_count']}件")

    return ", ".join(parts)
