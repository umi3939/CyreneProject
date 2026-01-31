# -*- coding: utf-8 -*-
"""
Full System Integration Test

All modules integrated and functioning:
1. Psyche (emotions, drives, mood)
2. Fear/Loss (4 pillars)
3. Responsibility (accumulation, decay, influence)
4. Policy selection (with responsibility bias)
5. Memory (recall by mood)
6. Logging control
7. Simulation mechanism
"""

import asyncio
import os
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Start in production mode
if 'CYRENE_DEBUG' in os.environ:
    del os.environ['CYRENE_DEBUG']


async def main():
    print("=" * 70)
    print("FULL SYSTEM INTEGRATION TEST")
    print("=" * 70)

    results = {}

    # ================================================================
    # TEST 1: Core Psyche System
    # ================================================================
    print("\n[1/7] Psyche Core (Emotions, Drives, Mood)")
    print("-" * 50)

    from psyche import PsycheState, Percept, react

    state = PsycheState()

    # Test emotion response
    happy_percept = Percept(text="test", emotion="happy", emotion_valence=0.7, intent="sharing")
    state_after_happy = react(happy_percept, state, delta_time=1.0)

    sad_percept = Percept(text="test", emotion="sad", emotion_valence=-0.6, intent="complaint")
    state_after_sad = react(sad_percept, state_after_happy, delta_time=1.0)

    emotions_work = (
        state_after_happy.emotions.joy > state.emotions.joy and
        state_after_sad.emotions.sorrow > state_after_happy.emotions.sorrow
    )

    mood_works = state_after_sad.mood.valence < state_after_happy.mood.valence

    print(f"  Emotions respond to input: {'OK' if emotions_work else 'FAIL'}")
    print(f"  Mood changes with emotions: {'OK' if mood_works else 'FAIL'}")
    print(f"    joy: {state.emotions.joy:.3f} -> {state_after_happy.emotions.joy:.3f}")
    print(f"    sorrow: {state_after_happy.emotions.sorrow:.3f} -> {state_after_sad.emotions.sorrow:.3f}")
    print(f"    mood: {state_after_happy.mood.valence:.3f} -> {state_after_sad.mood.valence:.3f}")

    results["psyche"] = emotions_work and mood_works

    # ================================================================
    # TEST 2: Fear/Loss System (4 Pillars)
    # ================================================================
    print("\n[2/7] Fear/Loss System (4 Pillars)")
    print("-" * 50)

    from psyche import compute_fear_index

    fear_low = compute_fear_index(0.1, 0.1, 0.1, 0.1)
    fear_high_attachment = compute_fear_index(0.2, 0.9, 0.2, 0.2)
    fear_high_identity = compute_fear_index(0.9, 0.2, 0.2, 0.2)

    fear_computed = fear_low.value < fear_high_attachment.value
    dominant_correct = (
        fear_high_attachment.dominant_fear == "attachment" and
        fear_high_identity.dominant_fear == "identity"
    )

    print(f"  Fear index computed from pillars: {'OK' if fear_computed else 'FAIL'}")
    print(f"  Dominant fear identified: {'OK' if dominant_correct else 'FAIL'}")
    print(f"    Low risk: {fear_low.value:.3f}")
    print(f"    High attachment: {fear_high_attachment.value:.3f} (dominant={fear_high_attachment.dominant_fear})")
    print(f"    High identity: {fear_high_identity.value:.3f} (dominant={fear_high_identity.dominant_fear})")

    results["fear"] = fear_computed and dominant_correct

    # ================================================================
    # TEST 3: Responsibility System
    # ================================================================
    print("\n[3/7] Responsibility System")
    print("-" * 50)

    from psyche import ResponsibilityManager
    from psyche.responsibility import apply_decay, get_influence

    resp_mgr = ResponsibilityManager()
    test_user = "integration_test_user_full"

    # Record decisions with negative outcomes
    for i in range(3):
        policy = {"policy_label": f"test_policy_{i}", "rationale": "test"}
        resp_mgr.record_decision(test_user, policy, {"target_partner": test_user, "fear_level": 0.3})

        resp_state = resp_mgr.get_state(test_user)
        unevaluated = [d for d in resp_state.recent_decisions if not d.get("evaluated", False)]
        if unevaluated:
            resp_mgr.evaluate_outcome(test_user, unevaluated[-1]["id"], {
                "user_reaction": "negative",
                "relationship_delta": -0.2,
                "expectation_gap": 0.4,
            })

    resp_state = resp_mgr.get_state(test_user)
    influence = resp_mgr.get_influence(test_user)

    responsibility_accumulates = resp_state.total_weight > 0.1 and resp_state.accumulated_harm > 0.05
    influence_works = influence.caution_bias > 0 and influence.empathy_bias > 0

    # Test decay
    initial_weight = resp_state.total_weight
    decayed = apply_decay(resp_state, hours_elapsed=48.0)
    decay_works = decayed.total_weight < initial_weight

    print(f"  Responsibility accumulates: {'OK' if responsibility_accumulates else 'FAIL'}")
    print(f"  Influence computed: {'OK' if influence_works else 'FAIL'}")
    print(f"  Time decay works: {'OK' if decay_works else 'FAIL'}")
    print(f"    total_weight: {resp_state.total_weight:.4f}")
    print(f"    accumulated_harm: {resp_state.accumulated_harm:.4f}")
    print(f"    caution_bias: {influence.caution_bias:.4f}")
    print(f"    empathy_bias: {influence.empathy_bias:.4f}")
    print(f"    decay: {initial_weight:.4f} -> {decayed.total_weight:.4f}")

    results["responsibility"] = responsibility_accumulates and influence_works and decay_works

    # ================================================================
    # TEST 4: Policy Selection with Responsibility
    # ================================================================
    print("\n[4/7] Policy Selection (with Responsibility Bias)")
    print("-" * 50)

    from psyche import generate_thought_candidates, select_policy

    state = PsycheState()
    percept = Percept(text="test", emotion="sad", emotion_valence=-0.5, intent="sharing")

    # Without responsibility
    candidates_no_resp = generate_thought_candidates(state, percept, [], None)
    policy_no_resp = select_policy(candidates_no_resp, state, None)

    # With high responsibility
    high_resp_influence = get_influence(resp_state.model_copy(update={
        "total_weight": 0.8,
        "accumulated_harm": 0.6,
    }))

    candidates_with_resp = generate_thought_candidates(state, percept, [], high_resp_influence)
    policy_with_resp = select_policy(candidates_with_resp, state, high_resp_influence)

    policy_generated = len(candidates_no_resp) > 0 and len(candidates_with_resp) > 0

    # High responsibility should favor empathetic policies
    empathetic_policies = ("共感する", "励ます", "質問で会話を広げる")
    bias_affects_policy = policy_with_resp["policy_label"] in empathetic_policies

    print(f"  Candidates generated: {'OK' if policy_generated else 'FAIL'}")
    print(f"  Responsibility biases policy: {'OK' if bias_affects_policy else 'FAIL'}")
    print(f"    Without responsibility: {policy_no_resp['policy_label']}")
    print(f"    With high responsibility: {policy_with_resp['policy_label']}")
    print(f"    (caution={high_resp_influence.caution_bias:.3f}, empathy={high_resp_influence.empathy_bias:.3f})")

    results["policy"] = policy_generated and bias_affects_policy

    # ================================================================
    # TEST 5: Memory System
    # ================================================================
    print("\n[5/7] Memory System (Mood-congruent Recall)")
    print("-" * 50)

    from psyche import recall_by_mood
    from src.memory_manager import MemoryManager

    async def dummy_llm(prompt, params=None):
        return "{}"

    memory_mgr = MemoryManager(llm_call=dummy_llm)

    state = PsycheState()
    percept = Percept(text="test", emotion="happy", emotion_valence=0.5, intent="sharing")

    recalled = recall_by_mood(percept, state, memory_mgr, top_k=5)

    memory_works = isinstance(recalled, list)
    memory_count = len(recalled)

    print(f"  Memory recall works: {'OK' if memory_works else 'FAIL'}")
    print(f"    Recalled {memory_count} memories")
    if recalled:
        print(f"    First: {recalled[0].get('summary', 'N/A')[:40]}...")

    results["memory"] = memory_works

    # ================================================================
    # TEST 6: Logging Control
    # ================================================================
    print("\n[6/7] Logging Control (Production/Debug)")
    print("-" * 50)

    from src.logging_config import is_debug_enabled
    from src.api import _filter_meta_for_production, _filter_state_for_production

    # Production mode (CYRENE_DEBUG not set)
    prod_debug = is_debug_enabled()

    meta = {"emotion": "happy", "internal_score": 0.85, "policy": "test"}
    state_dict = {"emotions": {"joy": 0.5}, "fear_index": 0.2, "last_updated": "2026-01-30"}
    percept = Percept(emotion="happy")

    filtered_meta = _filter_meta_for_production(meta, percept)
    filtered_state = _filter_state_for_production(state_dict)

    production_filters = (
        "internal_score" not in filtered_meta and
        "emotions" not in filtered_state
    )

    # Debug mode
    os.environ["CYRENE_DEBUG"] = "1"
    debug_enabled = is_debug_enabled()

    filtered_meta_debug = _filter_meta_for_production(meta, percept)
    filtered_state_debug = _filter_state_for_production(state_dict)

    debug_shows_all = (
        "internal_score" in filtered_meta_debug or filtered_meta_debug == meta
    )

    # Reset to production
    del os.environ["CYRENE_DEBUG"]

    print(f"  Production mode filters internal data: {'OK' if production_filters else 'FAIL'}")
    print(f"  Debug mode shows full data: {'OK' if debug_shows_all else 'FAIL'}")
    print(f"    Production meta keys: {list(filtered_meta.keys())}")
    print(f"    Debug meta keys: {list(filtered_meta_debug.keys())}")

    results["logging"] = production_filters and debug_shows_all

    # ================================================================
    # TEST 7: Simulation Mechanism
    # ================================================================
    print("\n[7/7] Simulation Mechanism")
    print("-" * 50)

    from src.simulation import SimulationConfig, SimulationEngine

    config = SimulationConfig(turns=10, pattern="mixed", user_id="sim_integration_test")
    engine = SimulationEngine(config)

    sim_state = PsycheState()
    for i in range(10):
        sim_state, record = await engine.run_turn(i, sim_state)

    simulation_works = (
        len(engine.records) == 10 and
        all(r.policy_label for r in engine.records)
    )

    caution_changes = engine.records[-1].influence_caution != engine.records[0].influence_caution

    print(f"  Simulation runs: {'OK' if simulation_works else 'FAIL'}")
    print(f"  State changes over time: {'OK' if caution_changes else 'FAIL'}")
    print(f"    Turns completed: {len(engine.records)}")
    print(f"    Initial caution: {engine.records[0].influence_caution:.4f}")
    print(f"    Final caution: {engine.records[-1].influence_caution:.4f}")

    results["simulation"] = simulation_works

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print("-" * 70)
    print(f"  Total: {sum(results.values())}/{len(results)} passed")
    print("=" * 70)

    if all_passed:
        print("\n*** ALL SYSTEMS INTEGRATED AND FUNCTIONING CORRECTLY ***")
    else:
        print("\n*** SOME TESTS FAILED - CHECK ABOVE FOR DETAILS ***")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
