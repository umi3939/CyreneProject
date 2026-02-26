"""
src/simulation.py - Long-term behavior observation simulation.

Observation-only mechanism for analyzing how the system behaves over time.
Uses PsycheOrchestrator for the full pipeline (all 70+ systems) instead of
calling psyche functions directly.

Usage::

    from src.simulation import run_simulation, SimulationConfig

    config = SimulationConfig(turns=100, pattern="mixed")
    results = await run_simulation(config)
    # Results saved to data/simulation_results_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from psyche.orchestrator import PsycheOrchestrator
from psyche.state import PsycheState, Percept
from psyche.perception import parse_percept
from psyche.expression import render_expression
from psyche.memory_link import recall_with_mood
from psyche.silence_hesitation import is_silence_policy

from src.memory_manager import MemoryManager

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

    # Orchestrator tick
    orchestrator_tick: int = 0

    # Policy selected
    policy_label: str = ""
    policy_score: float = 0.0


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    turns: int = 50
    pattern: Literal["positive", "negative", "confused", "neutral", "mixed", "repeated_failure"] = "mixed"
    user_id: str = "simulation_user"
    time_between_turns: float = 60.0  # Simulated seconds between turns


@dataclass
class SimulationResult:
    """Complete simulation results."""
    config: dict
    start_time: str
    end_time: str
    total_turns: int
    records: list[dict] = field(default_factory=list)

    # Summary statistics
    final_tick_count: int = 0
    policy_distribution: dict[str, int] = field(default_factory=dict)


# ── Simulation Engine ─────────────────────────────────────────────

class SimulationEngine:
    """Engine for running observation-only simulations.

    Uses PsycheOrchestrator for the full pipeline (all 70+ systems)
    instead of calling psyche functions directly.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.records: list[TurnRecord] = []

        # PsycheOrchestrator: full pipeline (isolated instance for simulation)
        self._orchestrator = PsycheOrchestrator(memory_count=0)

        # Memory manager (isolated from production)
        self._memory_mgr = MemoryManager(llm_call=self._dummy_llm)

        # Time tracking for delta
        self._last_update = time.monotonic()

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
        policy: dict,
    ) -> TurnRecord:
        """Record state snapshot for a turn."""
        psyche_state = self._orchestrator.psyche

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
            orchestrator_tick=self._orchestrator.tick_count,
            policy_label=policy.get("policy_label", "unknown"),
            policy_score=policy.get("_score", 0.0),
        )

    async def run_turn(self, turn: int) -> TurnRecord:
        """Execute a single simulation turn using PsycheOrchestrator.

        Follows the same pipeline as brain.py think_text / api.py respond:
        1. Create percept from input
        2. process_text_input (text dialogue processing)
        3. post_response_update (full orchestrator tick)
        4. recall memories
        5. select_policy_dict (policy selection with all biases)
        6. Record turn
        """
        user_id = self.config.user_id

        # Get input for this turn
        input_data = self._get_next_input(turn)
        percept = self._create_percept(input_data)

        # Phase 2: text dialogue input processing
        self._orchestrator.process_text_input(
            text=input_data["text"],
            sender_id=user_id,
            conversation_id=user_id,
        )

        # Phase 3: full orchestrator tick (all 70+ systems)
        now = time.monotonic()
        delta = now - self._last_update
        self._last_update = now
        # Use configured time_between_turns for consistent simulation
        self._orchestrator.post_response_update(
            percept, self.config.time_between_turns, user_id
        )

        # Phase 4: recall memories
        recall_percept = Percept(text=input_data["text"])
        memories = await recall_with_mood(
            recall_percept, self._orchestrator.psyche, self._memory_mgr, top_k=3
        )
        self._orchestrator.set_recalled_memories(memories)

        # Phase 5: policy selection (with all orchestrator biases)
        policy = self._orchestrator.select_policy_dict(
            percept, memories or [], user_id
        )

        # Phase 6: notify self-action perception
        policy_label = policy.get("policy_label", "")
        if policy_label and not is_silence_policy(policy):
            self._orchestrator.notify_self_output(
                response_text=f"[sim] {policy_label}",
                policy_label=policy_label,
            )

        # Record turn data
        record = self._record_turn(turn, input_data, policy)
        self.records.append(record)

        return record

    async def run(self) -> SimulationResult:
        """Run the complete simulation."""
        start_time = datetime.now().isoformat(timespec="seconds")

        logger.info("Starting simulation: %d turns, pattern=%s", self.config.turns, self.config.pattern)

        for turn in range(self.config.turns):
            record = await self.run_turn(turn)

            if turn % 10 == 0:
                logger.debug(
                    "Turn %d: policy=%s, tick=%d",
                    turn, record.policy_label, record.orchestrator_tick,
                )

        end_time = datetime.now().isoformat(timespec="seconds")

        # Compute summary statistics
        policy_counts: dict[str, int] = {}
        for r in self.records:
            policy_counts[r.policy_label] = policy_counts.get(r.policy_label, 0) + 1

        result = SimulationResult(
            config=asdict(self.config),
            start_time=start_time,
            end_time=end_time,
            total_turns=self.config.turns,
            records=[asdict(r) for r in self.records],
            final_tick_count=self._orchestrator.tick_count,
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
        final_tick_count=data.get("final_tick_count", 0),
        policy_distribution=data.get("policy_distribution", {}),
    )
