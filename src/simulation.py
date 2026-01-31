"""
src/simulation.py - Long-term behavior observation simulation.

Observation-only mechanism for analyzing how the system behaves over time.
Uses existing logic without modification - only observes and records.

Usage::

    from src.simulation import run_simulation, SimulationConfig

    config = SimulationConfig(turns=100, pattern="mixed")
    results = await run_simulation(config)
    # Results saved to data/simulation_results_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from psyche import (
    PsycheState,
    Percept,
    react,
    recall_by_mood,
    generate_thought_candidates,
    select_policy,
    compute_fear_index,
    ResponsibilityManager,
)
from psyche.responsibility import apply_decay

from src.state_manager import StateManager
from src.memory_manager import MemoryManager
from src.attachment_manager import AttachmentManager
from src.identity_manager import IdentityManager
from src.projection_manager import ProjectionManager

logger = logging.getLogger(__name__)

# ── Predefined Input Patterns ─────────────────────────────────────

INPUT_PATTERNS: dict[str, list[dict]] = {
    "positive": [
        {"text": "ありがとう、嬉しいわ", "emotion": "happy", "valence": 0.7, "intent": "sharing"},
        {"text": "あなたといると楽しい", "emotion": "loving", "valence": 0.8, "intent": "sharing"},
        {"text": "今日はいい日ね", "emotion": "happy", "valence": 0.5, "intent": "greeting"},
    ],
    "negative": [
        {"text": "なんでそんなこと言うの", "emotion": "angry", "valence": -0.6, "intent": "complaint"},
        {"text": "傷ついた…", "emotion": "sad", "valence": -0.7, "intent": "sharing"},
        {"text": "もういいよ", "emotion": "angry", "valence": -0.5, "intent": "complaint"},
    ],
    "confused": [
        {"text": "え、どういう意味？", "emotion": "surprised", "valence": -0.2, "intent": "question"},
        {"text": "よくわからない…", "emotion": "scared", "valence": -0.3, "intent": "complaint"},
        {"text": "ちょっと怖いかも", "emotion": "scared", "valence": -0.4, "intent": "sharing"},
    ],
    "neutral": [
        {"text": "そうなんだ", "emotion": "neutral", "valence": 0.0, "intent": "unknown"},
        {"text": "ふーん", "emotion": "neutral", "valence": 0.0, "intent": "unknown"},
        {"text": "なるほどね", "emotion": "neutral", "valence": 0.1, "intent": "sharing"},
    ],
}

# Mixed pattern sequences for realistic simulation
MIXED_SEQUENCES: list[str] = [
    "positive", "positive", "neutral", "negative", "neutral",
    "positive", "confused", "neutral", "positive", "negative",
]

# Repeated failure sequence (for observing judgment bias changes)
REPEATED_FAILURE_SEQUENCE: list[str] = [
    "negative", "negative", "negative", "confused", "negative",
    "negative", "neutral", "negative", "negative", "negative",
]


# ── Data Structures ───────────────────────────────────────────────

@dataclass
class TurnRecord:
    """Record of a single simulation turn."""
    turn: int
    timestamp: str

    # Input
    input_text: str
    input_emotion: str
    input_valence: float

    # PsycheState snapshot
    emotions: dict[str, float]
    drives: dict[str, float]
    mood_valence: float
    mood_arousal: float
    fear_level: float

    # Responsibility snapshot
    responsibility_total_weight: float
    responsibility_harm: float
    responsibility_confidence: float
    responsibility_pending: int

    # Responsibility influence
    influence_caution: float
    influence_empathy: float
    influence_anxiety: float
    influence_fear_amp: float

    # Policy selected
    policy_label: str
    policy_score: float


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    turns: int = 50
    pattern: Literal["positive", "negative", "confused", "neutral", "mixed", "repeated_failure"] = "mixed"
    user_id: str = "simulation_user"
    time_between_turns: float = 60.0  # Simulated seconds between turns
    decay_hours_per_turn: float = 0.0  # Additional decay hours to simulate (0 = none)


@dataclass
class SimulationResult:
    """Complete simulation results."""
    config: dict
    start_time: str
    end_time: str
    total_turns: int
    records: list[dict] = field(default_factory=list)

    # Summary statistics
    final_responsibility_weight: float = 0.0
    final_harm: float = 0.0
    max_caution_reached: float = 0.0
    policy_distribution: dict[str, int] = field(default_factory=dict)


# ── Simulation Engine ─────────────────────────────────────────────

class SimulationEngine:
    """Engine for running observation-only simulations.

    Uses existing logic without modification - only observes and records.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.records: list[TurnRecord] = []

        # Initialize managers (isolated from production)
        self.state_mgr = StateManager()
        self.responsibility_mgr = ResponsibilityManager()
        self.memory_mgr = MemoryManager(llm_call=self._dummy_llm)
        self.attachment_mgr = AttachmentManager()
        self.identity_mgr = IdentityManager()
        self.projection_mgr = ProjectionManager()

        self._pattern_index = 0

    @staticmethod
    async def _dummy_llm(prompt: str, params: dict | None = None) -> str:
        """Dummy LLM for simulation (not used in LOCAL pipeline)."""
        return "{}"

    def _get_next_input(self, turn: int) -> dict:
        """Get next input based on configured pattern."""
        pattern = self.config.pattern

        if pattern == "mixed":
            seq_idx = turn % len(MIXED_SEQUENCES)
            pattern_key = MIXED_SEQUENCES[seq_idx]
        elif pattern == "repeated_failure":
            seq_idx = turn % len(REPEATED_FAILURE_SEQUENCE)
            pattern_key = REPEATED_FAILURE_SEQUENCE[seq_idx]
        else:
            pattern_key = pattern

        inputs = INPUT_PATTERNS[pattern_key]
        input_idx = turn % len(inputs)
        return inputs[input_idx]

    def _create_percept(self, input_data: dict) -> Percept:
        """Create a Percept from input data."""
        return Percept(
            text=input_data["text"],
            emotion=input_data["emotion"],
            emotion_valence=input_data["valence"],
            intent=input_data["intent"],
            meaning=input_data["text"],
        )

    def _record_turn(
        self,
        turn: int,
        input_data: dict,
        psyche_state: PsycheState,
        policy: dict,
    ) -> TurnRecord:
        """Record state snapshot for a turn."""
        resp_state = self.responsibility_mgr.get_state(self.config.user_id)
        influence = self.responsibility_mgr.get_influence(self.config.user_id)

        return TurnRecord(
            turn=turn,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            input_text=input_data["text"],
            input_emotion=input_data["emotion"],
            input_valence=input_data["valence"],
            emotions=psyche_state.emotions.as_dict(),
            drives=psyche_state.drives.as_dict(),
            mood_valence=psyche_state.mood.valence,
            mood_arousal=psyche_state.mood.arousal,
            fear_level=psyche_state.fear_level,
            responsibility_total_weight=resp_state.total_weight,
            responsibility_harm=resp_state.accumulated_harm,
            responsibility_confidence=resp_state.accumulated_confidence,
            responsibility_pending=resp_state.pending_decisions,
            influence_caution=influence.caution_bias,
            influence_empathy=influence.empathy_bias,
            influence_anxiety=influence.anxiety_baseline,
            influence_fear_amp=influence.fear_amplification,
            policy_label=policy.get("policy_label", "unknown"),
            policy_score=policy.get("_score", 0.0),
        )

    async def run_turn(self, turn: int, psyche_state: PsycheState) -> tuple[PsycheState, TurnRecord]:
        """Execute a single simulation turn using existing logic.

        This follows the exact same flow as src/api.py without modification.
        """
        user_id = self.config.user_id

        # Get input for this turn
        input_data = self._get_next_input(turn)
        percept = self._create_percept(input_data)

        # Evaluate previous decision (same as api.py step 2)
        self._evaluate_previous_decision(user_id, percept)

        # Get responsibility influence AFTER evaluation (same as api.py step 3)
        responsibility_influence = self.responsibility_mgr.get_influence(user_id)

        # Apply optional time decay for simulation
        if self.config.decay_hours_per_turn > 0:
            resp_state = self.responsibility_mgr.get_state(user_id)
            decayed = apply_decay(resp_state, self.config.decay_hours_per_turn)
            # Update via manager's internal state (respecting encapsulation)
            self.responsibility_mgr._data[user_id] = {
                "total_weight": decayed.total_weight,
                "pending_decisions": decayed.pending_decisions,
                "accumulated_harm": decayed.accumulated_harm,
                "accumulated_confidence": decayed.accumulated_confidence,
                "last_updated": decayed.last_updated,
                "recent_decisions": decayed.recent_decisions,
            }

        # React (same as api.py step 4)
        psyche_state = react(
            percept, psyche_state,
            delta_time=self.config.time_between_turns,
            responsibility_influence=responsibility_influence,
        )

        # Recall memories (same as api.py step 3/recall)
        recalled = recall_by_mood(percept, psyche_state, self.memory_mgr, top_k=3)

        # Generate candidates and select policy (same as api.py steps 5-6)
        candidates = generate_thought_candidates(
            psyche_state, percept, recalled, responsibility_influence,
        )
        policy = select_policy(candidates, psyche_state, responsibility_influence)

        # Record the decision (same as api.py step 9c)
        decision_context = {
            "target_partner": user_id,
            "emotional_state": psyche_state.mood.valence_label,
            "fear_level": psyche_state.fear_level,
            "involves_attachment": True,
        }
        self.responsibility_mgr.record_decision(user_id, policy, decision_context)

        # Update attachment (same as api.py step 9b)
        positive = input_data["valence"] > 0
        self.attachment_mgr.update_bond(user_id, "partner", positive=positive, importance=3)

        # Recalculate fear_index (same as api.py step 9d)
        fear_index = compute_fear_index(
            identity_risk=self.identity_mgr.get_risk(),
            attachment_risk=self.attachment_mgr.get_risk(user_id),
            continuity_risk=0.3,  # Fixed for simulation
            projection_risk=self.projection_mgr.get_risk(),
        )

        # Update psyche_state with new fear_index (same as api.py step 9e)
        psyche_state = PsycheState(
            emotions=psyche_state.emotions,
            drives=psyche_state.drives,
            mood=psyche_state.mood,
            identity=psyche_state.identity,
            attachment=psyche_state.attachment,
            continuity=psyche_state.continuity,
            projection=psyche_state.projection,
            fear_index=fear_index,
            loss_aversion=psyche_state.loss_aversion,
            last_updated=datetime.now().isoformat(timespec="seconds"),
        )

        # Record turn data
        record = self._record_turn(turn, input_data, psyche_state, policy)
        self.records.append(record)

        return psyche_state, record

    def _evaluate_previous_decision(self, user_id: str, percept: Percept) -> None:
        """Evaluate previous decision based on current input.

        Same logic as src/api.py _evaluate_previous_decision.
        """
        resp_state = self.responsibility_mgr.get_state(user_id)
        unevaluated = [
            d for d in resp_state.recent_decisions
            if not d.get("evaluated", False)
        ]

        if not unevaluated:
            return

        decision = unevaluated[-1]
        decision_id = decision.get("id")
        if not decision_id:
            return

        # Infer outcome from percept (same logic as api.py)
        valence = percept.emotion_valence
        emotion = percept.emotion
        intent = percept.intent

        if valence > 0.3 or emotion in ("happy", "loving"):
            user_reaction = "positive"
        elif valence < -0.3 or emotion in ("angry", "sad"):
            user_reaction = "negative"
        elif emotion == "scared" or intent == "complaint":
            user_reaction = "confused"
        else:
            user_reaction = "neutral"

        relationship_delta = {"positive": 0.1, "negative": -0.15, "confused": -0.05}.get(user_reaction, 0.0)
        expectation_gap = abs(valence) * 0.5 if user_reaction in ("negative", "confused") else 0.0

        outcome = {
            "user_reaction": user_reaction,
            "relationship_delta": relationship_delta,
            "expectation_gap": expectation_gap,
        }

        self.responsibility_mgr.evaluate_outcome(user_id, decision_id, outcome)

    async def run(self) -> SimulationResult:
        """Run the complete simulation."""
        start_time = datetime.now().isoformat(timespec="seconds")
        psyche_state = PsycheState()

        logger.info("Starting simulation: %d turns, pattern=%s", self.config.turns, self.config.pattern)

        for turn in range(self.config.turns):
            psyche_state, record = await self.run_turn(turn, psyche_state)

            if turn % 10 == 0:
                logger.debug(
                    "Turn %d: policy=%s, caution=%.3f, harm=%.3f",
                    turn, record.policy_label, record.influence_caution, record.responsibility_harm,
                )

        end_time = datetime.now().isoformat(timespec="seconds")

        # Compute summary statistics
        policy_counts: dict[str, int] = {}
        max_caution = 0.0
        for r in self.records:
            policy_counts[r.policy_label] = policy_counts.get(r.policy_label, 0) + 1
            max_caution = max(max_caution, r.influence_caution)

        final_resp = self.responsibility_mgr.get_state(self.config.user_id)

        result = SimulationResult(
            config=asdict(self.config),
            start_time=start_time,
            end_time=end_time,
            total_turns=self.config.turns,
            records=[asdict(r) for r in self.records],
            final_responsibility_weight=final_resp.total_weight,
            final_harm=final_resp.accumulated_harm,
            max_caution_reached=max_caution,
            policy_distribution=policy_counts,
        )

        return result


# ── Public API ────────────────────────────────────────────────────

async def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    """Run a simulation and save results.

    Args:
        config: Simulation configuration. Uses defaults if None.

    Returns:
        SimulationResult with all recorded data.
    """
    if config is None:
        config = SimulationConfig()

    engine = SimulationEngine(config)
    result = await engine.run()

    # Save results to file
    output_path = save_results(result)
    logger.info("Simulation complete. Results saved to %s", output_path)

    return result


def save_results(result: SimulationResult) -> Path:
    """Save simulation results to JSON file."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_results_{timestamp}.json"
    output_path = data_dir / filename

    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def load_results(filepath: Path | str) -> SimulationResult:
    """Load simulation results from JSON file."""
    path = Path(filepath)
    data = json.loads(path.read_text(encoding="utf-8"))

    return SimulationResult(
        config=data["config"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        total_turns=data["total_turns"],
        records=data["records"],
        final_responsibility_weight=data.get("final_responsibility_weight", 0.0),
        final_harm=data.get("final_harm", 0.0),
        max_caution_reached=data.get("max_caution_reached", 0.0),
        policy_distribution=data.get("policy_distribution", {}),
    )
