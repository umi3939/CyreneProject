"""
psyche/reaction.py - Emotion & Drive Reaction Model

Updates PsycheState based on external stimuli (Percept) and time passage.
Handles emotion updates, decay, drive changes, mood update, and **responsibility influence**.

責任（Responsibility）の影響:
- anxiety_baseline: 不安・恐怖系感情のベースラインを上昇
- fear_amplification: 喪失への恐れを増幅

感情振れ幅拡張（Emotion Amplitude Expansion）:
- amplitude_modifier: 感情の変化量（delta）を増減させる
- 方向（符号）は変えず、変化量のみをスケール
- 一時的であり、時間経過で基準値に戻る

ドライブ動態の状態依存化（Drive Dynamics State-Dependent）:
- → reaction_drive_dynamics.py に分離

ムードの自律化（Mood Autonomy）:
- → reaction_mood_update.py に分離
"""

from __future__ import annotations

from typing import Optional

from .fear import fear_drive_boost, fear_emotion_boost
from .responsibility import ResponsibilityInfluence
from .state import DriveVector, EmotionVector, Mood, Percept, PsycheState
from .emotion_amplitude import apply_amplitude_to_delta
from . import coefficient_registry

# Re-export drive dynamics elements (maintains external import paths)
from .reaction_drive_dynamics import (  # noqa: F401
    DriveContextInputs,
    _SECTION_BAND,
    _TOTAL_CHANGE_LIMIT,
    _compute_emotion_drive_coupling,
    _compute_drive_interaction,
    _compute_goal_hierarchy,
    _compute_time_passage,
    _compute_arousal_drive,
    _compute_behavioral_diversity,
    _compute_internal_contradiction,
    _compute_result_diversity_return,
    compute_state_dependent_drive_changes,
)

# Re-export mood update elements (maintains external import paths)
from .reaction_mood_update import (  # noqa: F401
    MoodContextInputs,
    _MOOD_BAND,
    _TRACKING_SPEED_MIN,
    _TRACKING_SPEED_MAX,
    _MOOD_DELTA_LIMIT,
    _derive_mood_targets,
    _derive_tracking_speeds,
    compute_autonomous_mood,
)

# Mapping from percept emotion labels to emotion vector fields
_EMOTION_MAP: dict[str, str] = {
    "happy": "joy",
    "sad": "sorrow",
    "angry": "anger",
    "surprised": "surprise",
    "scared": "fear",
    "loving": "love",
    "teasing": "fun",
    "neutral": "",
}

# How strongly a percept's valence affects each emotion (from coefficient registry)
_emo_coeffs = coefficient_registry.get("emotion_processing")
_VALENCE_POSITIVE = _emo_coeffs["valence_positive"]
_VALENCE_NEGATIVE = _emo_coeffs["valence_negative"]

DECAY_RATE = _emo_coeffs["decay_rate"]  # Per-second exponential decay factor


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def react(
    percept: Percept,
    state: PsycheState,
    delta_time: float = 1.0,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    amplitude_modifier: float = 1.0,
    drive_context: Optional[DriveContextInputs] = None,
    mood_context: Optional[MoodContextInputs] = None,
) -> PsycheState:
    """
    Produce a new PsycheState by reacting to a Percept over delta_time seconds.

    1. Apply emotion stimulus from the Percept (scaled by amplitude_modifier).
    2. Decay all emotions toward zero over time.
    3. Apply responsibility-driven anxiety baseline.
    4. Update drives (social, curiosity, expression) via state-dependent dynamics.
    5. Update mood via autonomous multi-input-source pipeline.

    Args:
        percept: Interpreted stimulus.
        state: Current psychological state (not mutated).
        delta_time: Seconds elapsed since last update.
        responsibility_influence: Optional responsibility influence on emotions.
        amplitude_modifier: Scales emotion change amounts (1.0 = no change).
                          Does NOT change direction, only magnitude.
        drive_context: Optional context for state-dependent drive dynamics.
                      If None, a default context is constructed from available state.
        mood_context: Optional context for autonomous mood update.
                     If None, a default context is constructed from available state.

    Returns:
        New PsycheState reflecting the reaction.
    """
    emo = state.emotions.as_dict()

    # --- 1. Emotion stimulus from Percept ---
    # Direct emotion mapping (with amplitude scaling)
    target_field = _EMOTION_MAP.get(percept.emotion, "")
    if target_field and target_field in emo:
        base_delta = _emo_coeffs["stimulus_base_delta"]
        scaled_delta = apply_amplitude_to_delta(base_delta, amplitude_modifier)
        emo[target_field] = _clamp(emo[target_field] + scaled_delta)

    # Valence-based secondary effects (with amplitude scaling)
    v = percept.emotion_valence
    if v > 0:
        for field, weight in _VALENCE_POSITIVE.items():
            base_delta = v * weight
            scaled_delta = apply_amplitude_to_delta(base_delta, amplitude_modifier)
            emo[field] = _clamp(emo[field] + scaled_delta)
    elif v < 0:
        for field, weight in _VALENCE_NEGATIVE.items():
            base_delta = abs(v) * weight
            scaled_delta = apply_amplitude_to_delta(base_delta, amplitude_modifier)
            emo[field] = _clamp(emo[field] + scaled_delta)

    # --- 2. Time-based emotion decay ---
    decay = DECAY_RATE ** delta_time
    for k in emo:
        emo[k] = _clamp(emo[k] * decay)

    new_emotions = EmotionVector(**emo)

    # --- 2.5 Fear-driven emotion boost ---
    if state.fear_index is not None:
        new_emotions = fear_emotion_boost(state.fear_index, new_emotions)

    # --- 2.6 Responsibility-driven anxiety baseline ---
    # 責任の重みが、不安・恐怖系感情のベースラインを上昇させる
    if responsibility_influence is not None:
        new_emotions = _apply_responsibility_emotion_influence(
            new_emotions, responsibility_influence
        )

    # --- 3. Drive updates (state-dependent dynamics) ---
    drv = state.drives.model_dump()

    # Construct drive context if not provided (安全弁5: 入力不在時は中立化)
    if drive_context is None:
        drive_context = DriveContextInputs(
            emotions=emo,
            mood_valence=state.mood.valence,
            mood_arousal=state.mood.arousal,
            drives=drv.copy(),
            delta_time=delta_time,
            percept_intent=percept.intent,
            percept_emotion=percept.emotion,
            percept_valence=percept.emotion_valence,
            fear_level=state.fear_level,
        )
    else:
        # Ensure emotions and drives reflect current tick's values
        drive_context.emotions = emo
        drive_context.drives = drv.copy()
        drive_context.delta_time = delta_time
        drive_context.percept_intent = percept.intent
        drive_context.percept_emotion = percept.emotion
        drive_context.percept_valence = percept.emotion_valence
        drive_context.mood_valence = state.mood.valence
        drive_context.mood_arousal = state.mood.arousal
        drive_context.fear_level = state.fear_level

    # Compute state-dependent drive changes (pure function, no accumulation)
    drive_deltas = compute_state_dependent_drive_changes(drive_context)

    # Apply changes with clamping (安全弁3: ドライブ値の有効範囲クランプ)
    for axis in ("social", "curiosity", "expression"):
        drv[axis] = _clamp(drv[axis] + drive_deltas[axis])

    new_drives = DriveVector(**drv)

    # --- 3.5 Fear-driven drive boost ---
    if state.fear_index is not None:
        new_drives = fear_drive_boost(state.fear_index, new_drives)

    # --- 4. Autonomous mood update (multi-input-source pipeline) ---
    # Construct mood context if not provided (安全弁4: 入力不在時は中立化)
    if mood_context is None:
        mood_context = MoodContextInputs(
            emotions=new_emotions.as_dict(),
            drives=new_drives.as_dict(),
            current_valence=state.mood.valence,
            current_arousal=state.mood.arousal,
            fear_level=state.fear_level,
            responsibility_anxiety=(
                responsibility_influence.anxiety_baseline
                if responsibility_influence is not None else 0.0
            ),
            delta_time=delta_time,
        )
    else:
        # Ensure emotions/drives reflect current tick's updated values
        mood_context.emotions = new_emotions.as_dict()
        mood_context.drives = new_drives.as_dict()
        mood_context.current_valence = state.mood.valence
        mood_context.current_arousal = state.mood.arousal
        mood_context.fear_level = state.fear_level
        mood_context.delta_time = delta_time
        if responsibility_influence is not None:
            mood_context.responsibility_anxiety = responsibility_influence.anxiety_baseline

    new_valence, new_arousal = compute_autonomous_mood(mood_context)

    # --- 4.5 Responsibility affects mood ---
    # 責任の重みがムードのvalenceを少し下げる（重荷として）
    if responsibility_influence is not None:
        _resp_mood_scale = _emo_coeffs.get("responsibility_mood_penalty_scale", 0.3)
        mood_penalty = responsibility_influence.anxiety_baseline * _resp_mood_scale
        new_valence = new_valence - mood_penalty

    new_mood = Mood(
        valence=_clamp(new_valence, -1.0, 1.0),
        arousal=_clamp(new_arousal, 0.0, 1.0),
    )

    return PsycheState(
        emotions=new_emotions,
        drives=new_drives,
        mood=new_mood,
        identity=state.identity,
        attachment=state.attachment,
        continuity=state.continuity,
        projection=state.projection,
        fear_index=state.fear_index,
    )


def _apply_responsibility_emotion_influence(
    emotions: EmotionVector,
    influence: ResponsibilityInfluence,
) -> EmotionVector:
    """責任の影響を感情に適用する。

    - anxiety_baseline: 恐怖・悲しみのベースラインを上昇
    - fear_amplification: 恐怖を増幅（喪失への恐れ）

    影響は穏やかで、上限が設けられている。
    """
    emo = emotions.as_dict()

    # 不安ベースライン: 恐怖と悲しみの最低値を引き上げ
    anxiety = influence.anxiety_baseline
    if anxiety > 0:
        emo["fear"] = _clamp(max(emo["fear"], anxiety * 0.5))
        emo["sorrow"] = _clamp(max(emo["sorrow"], anxiety * 0.3))

    # 喪失恐怖の増幅
    fear_amp = influence.fear_amplification
    if fear_amp > 0 and emo["fear"] > 0.1:
        emo["fear"] = _clamp(emo["fear"] + fear_amp * 0.2)

    return EmotionVector(**emo)
