"""
psyche/reaction.py - Emotion & Drive Reaction Model

Updates PsycheState based on external stimuli (Percept) and time passage.
Handles emotion updates, decay, drive changes, mood drift, and **responsibility influence**.

責任（Responsibility）の影響:
- anxiety_baseline: 不安・恐怖系感情のベースラインを上昇
- fear_amplification: 喪失への恐れを増幅

感情振れ幅拡張（Emotion Amplitude Expansion）:
- amplitude_modifier: 感情の変化量（delta）を増減させる
- 方向（符号）は変えず、変化量のみをスケール
- 一時的であり、時間経過で基準値に戻る
"""

from __future__ import annotations

from typing import Optional

from .fear import fear_drive_boost, fear_emotion_boost
from .responsibility import ResponsibilityInfluence
from .state import DriveVector, EmotionVector, Mood, Percept, PsycheState
from .emotion_amplitude import apply_amplitude_to_delta

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

# How strongly a percept's valence affects each emotion
_VALENCE_POSITIVE = {"joy": 0.15, "love": 0.05, "fun": 0.05}
_VALENCE_NEGATIVE = {"sorrow": 0.10, "anger": 0.05, "fear": 0.05}

DECAY_RATE = 0.95  # Per-second exponential decay factor


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def react(
    percept: Percept,
    state: PsycheState,
    delta_time: float = 1.0,
    responsibility_influence: Optional[ResponsibilityInfluence] = None,
    amplitude_modifier: float = 1.0,
) -> PsycheState:
    """
    Produce a new PsycheState by reacting to a Percept over delta_time seconds.

    1. Apply emotion stimulus from the Percept (scaled by amplitude_modifier).
    2. Decay all emotions toward zero over time.
    3. Apply responsibility-driven anxiety baseline.
    4. Update drives (social, curiosity, expression).
    5. Drift mood toward current emotion average.

    Args:
        percept: Interpreted stimulus.
        state: Current psychological state (not mutated).
        delta_time: Seconds elapsed since last update.
        responsibility_influence: Optional responsibility influence on emotions.
        amplitude_modifier: Scales emotion change amounts (1.0 = no change).
                          Does NOT change direction, only magnitude.

    Returns:
        New PsycheState reflecting the reaction.
    """
    emo = state.emotions.as_dict()

    # --- 1. Emotion stimulus from Percept ---
    # Direct emotion mapping (with amplitude scaling)
    target_field = _EMOTION_MAP.get(percept.emotion, "")
    if target_field and target_field in emo:
        base_delta = 0.2
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

    # --- 3. Drive updates ---
    drv = state.drives.model_dump()

    # Social: increases over time (loneliness), decreases with conversation
    drv["social"] = _clamp(drv["social"] + 0.02 * delta_time)  # lonely over time
    drv["social"] = _clamp(drv["social"] - 0.15)  # talking reduces loneliness

    # Curiosity: increases over time, decreases with new information
    drv["curiosity"] = _clamp(drv["curiosity"] + 0.01 * delta_time)
    if percept.intent in ("sharing", "question"):
        drv["curiosity"] = _clamp(drv["curiosity"] - 0.10)

    # Expression: increases with strong emotions, decays over time
    max_emo = max(emo.values())
    drv["expression"] = _clamp(drv["expression"] + max_emo * 0.05)
    drv["expression"] = _clamp(drv["expression"] * (0.98 ** delta_time))

    new_drives = DriveVector(**drv)

    # --- 3.5 Fear-driven drive boost ---
    if state.fear_index is not None:
        new_drives = fear_drive_boost(state.fear_index, new_drives)

    # --- 4. Mood drift ---
    # Positive emotions push valence up, negative push down
    positive_sum = new_emotions.joy + new_emotions.love + new_emotions.fun
    negative_sum = new_emotions.sorrow + new_emotions.anger + new_emotions.fear
    instant_valence = (positive_sum - negative_sum) / 3.0  # Normalize to roughly -1..1

    # Arousal tracks overall emotional intensity
    instant_arousal = max(emo.values()) if emo.values() else 0.0

    # Slow exponential moving average
    alpha = 0.1  # Mood inertia (lower = slower change)
    new_valence = state.mood.valence + alpha * (instant_valence - state.mood.valence)
    new_arousal = state.mood.arousal + alpha * (instant_arousal - state.mood.arousal)

    # --- 4.5 Responsibility affects mood ---
    # 責任の重みがムードのvalenceを少し下げる（重荷として）
    if responsibility_influence is not None:
        mood_penalty = responsibility_influence.anxiety_baseline * 0.3
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
