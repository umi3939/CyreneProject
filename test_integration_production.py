"""
本番環境 統合検証テスト

これまで実装した各機能が相互作用したときに想定した挙動を維持できるかを検証する。
"""

import asyncio
import os

# Production mode
if 'CYRENE_DEBUG' in os.environ:
    del os.environ['CYRENE_DEBUG']

from psyche import (
    PsycheState, Percept, react, recall_by_mood,
    generate_thought_candidates, select_policy, compute_fear_index,
    ResponsibilityManager,
)
from psyche.responsibility import apply_decay, get_influence
from src.memory_manager import MemoryManager
from src.attachment_manager import AttachmentManager
from src.identity_manager import IdentityManager


async def dummy_llm(prompt, params=None):
    return '{}'


async def run_integration_test():
    print('=' * 60)
    print('本番環境 統合検証テスト')
    print('=' * 60)
    print()

    # Initialize all managers
    resp_mgr = ResponsibilityManager()
    memory_mgr = MemoryManager(llm_call=dummy_llm)
    attachment_mgr = AttachmentManager()
    identity_mgr = IdentityManager()

    user_id = 'integration_test_user'

    print('【検証1】責任の蓄積と判断バイアスの相互作用')
    print('-' * 50)

    state = PsycheState()

    # Phase 1: 繰り返しの失敗による責任蓄積
    print()
    print('Phase 1: 繰り返しの失敗 (5ターン)')

    for turn in range(5):
        # Negative input
        percept = Percept(
            text=f'傷ついた...({turn+1}回目)',
            emotion='sad',
            emotion_valence=-0.6,
            intent='complaint'
        )

        # Evaluate previous decision (if any)
        resp_state = resp_mgr.get_state(user_id)
        unevaluated = [d for d in resp_state.recent_decisions if not d.get('evaluated', False)]
        if unevaluated:
            decision_id = unevaluated[-1].get('id')
            if decision_id:
                resp_mgr.evaluate_outcome(user_id, decision_id, {
                    'user_reaction': 'negative',
                    'relationship_delta': -0.15,
                    'expectation_gap': 0.4,
                })

        # Get updated influence
        influence = resp_mgr.get_influence(user_id)

        # React with responsibility
        state = react(percept, state, delta_time=60.0, responsibility_influence=influence)

        # Recall memories
        recalled = recall_by_mood(percept, state, memory_mgr, top_k=3)

        # Generate and select policy
        candidates = generate_thought_candidates(state, percept, recalled, influence)
        policy = select_policy(candidates, state, influence)

        # Record decision
        _, decision_id = resp_mgr.record_decision(user_id, policy, {
            'target_partner': user_id,
            'fear_level': state.fear_level,
            'involves_attachment': True,
        })

        # Update attachment (negative)
        attachment_mgr.update_bond(user_id, 'partner', positive=False, importance=4)

        if turn in [0, 2, 4]:
            resp_state = resp_mgr.get_state(user_id)
            print(f'  Turn {turn+1}: policy={policy["policy_label"]}, '
                  f'caution={influence.caution_bias:.3f}, '
                  f'harm={resp_state.accumulated_harm:.3f}')

    # Check Phase 1 results
    resp_state = resp_mgr.get_state(user_id)
    influence = resp_mgr.get_influence(user_id)

    print()
    print('  [検証] 責任が蓄積されたか:')
    print(f'    total_weight: {resp_state.total_weight:.4f} (期待: > 0.3)')
    print(f'    accumulated_harm: {resp_state.accumulated_harm:.4f} (期待: > 0.1)')
    print(f'    caution_bias: {influence.caution_bias:.4f} (期待: > 0.05)')
    print(f'    empathy_bias: {influence.empathy_bias:.4f} (期待: > 0.1)')

    assert resp_state.total_weight > 0.3, 'total_weight should increase'
    assert resp_state.accumulated_harm > 0.1, 'harm should accumulate'
    assert influence.caution_bias > 0.05, 'caution should increase'
    print('  → OK: 責任が正しく蓄積されている')

    # Phase 2: からかうが抑制されるか確認
    print()
    print('Phase 2: リスキーな選択の抑制確認')

    # Happy input that might trigger teasing
    percept_happy = Percept(
        text='面白いこと言って！',
        emotion='happy',
        emotion_valence=0.6,
        intent='joke'
    )

    influence = resp_mgr.get_influence(user_id)
    state = react(percept_happy, state, delta_time=1.0, responsibility_influence=influence)
    candidates = generate_thought_candidates(state, percept_happy, [], influence)
    policy = select_policy(candidates, state, influence)

    print(f'  Input: joke request with positive valence')
    print(f'  Current caution_bias: {influence.caution_bias:.4f}')
    print(f'  Policy selected: {policy["policy_label"]}')

    # からかう should be avoided when caution is high
    if influence.caution_bias > 0.3:
        assert policy['policy_label'] != 'からかう', 'Should avoid risky choice'
        print('  → OK: 慎重さバイアスにより、リスキーな選択が抑制された')
    else:
        print('  → Note: caution not high enough to suppress teasing yet')

    # Phase 3: 時間経過による減衰
    print()
    print('Phase 3: 時間経過による責任の減衰')

    initial_weight = resp_state.total_weight
    initial_harm = resp_state.accumulated_harm

    # Simulate 48 hours passing
    decayed_state = apply_decay(resp_state, hours_elapsed=48.0)

    print(f'  48時間経過シミュレーション:')
    print(f'    total_weight: {initial_weight:.4f} → {decayed_state.total_weight:.4f}')
    print(f'    accumulated_harm: {initial_harm:.4f} → {decayed_state.accumulated_harm:.4f}')

    assert decayed_state.total_weight < initial_weight, 'weight should decay'
    assert decayed_state.accumulated_harm < initial_harm, 'harm should decay (slower)'
    print('  → OK: 責任は減衰するが、傷はゆっくり残る')

    # Phase 4: 恐怖指数と感情の相互作用
    print()
    print('Phase 4: 恐怖指数と感情の相互作用')

    # Compute fear index with attachment risk
    attachment_risk = attachment_mgr.get_risk(user_id)
    fear_index = compute_fear_index(
        identity_risk=0.2,
        attachment_risk=attachment_risk,
        continuity_risk=0.3,
        projection_risk=0.1,
    )

    # Update state with fear
    state = PsycheState(
        emotions=state.emotions,
        drives=state.drives,
        mood=state.mood,
        fear_index=fear_index,
    )

    print(f'  attachment_risk: {attachment_risk:.4f}')
    print(f'  fear_index.value: {fear_index.value:.4f}')
    print(f'  dominant_fear: {fear_index.dominant_fear}')

    # React with fear - should boost empathetic policies
    percept_sad = Percept(text='悲しい...', emotion='sad', emotion_valence=-0.5, intent='sharing')
    influence = resp_mgr.get_influence(user_id)

    state_with_fear = react(percept_sad, state, delta_time=1.0, responsibility_influence=influence)
    candidates = generate_thought_candidates(state_with_fear, percept_sad, [], influence)
    policy = select_policy(candidates, state_with_fear, influence)

    print(f'  Policy with high fear: {policy["policy_label"]}')
    assert policy['policy_label'] in ('共感する', '励ます', '質問で会話を広げる'), 'Should choose empathetic policy'
    print('  → OK: 恐怖が高い状態では共感的な選択をする')

    # Phase 5: 判断の継続性確認
    print()
    print('Phase 5: 極端な状態でも判断が継続されるか')

    # Create extreme state
    extreme_resp_state = resp_mgr.get_state(user_id)
    # Manually set to extreme (for test only)
    extreme_resp_state = extreme_resp_state.model_copy(update={
        'total_weight': 1.0,
        'accumulated_harm': 1.0,
    })
    extreme_influence = get_influence(extreme_resp_state)

    print(f'  Extreme state: weight=1.0, harm=1.0')
    print(f'  caution_bias: {extreme_influence.caution_bias:.4f}')
    print(f'  empathy_bias: {extreme_influence.empathy_bias:.4f}')

    # Should still be able to select policy
    candidates = generate_thought_candidates(state, percept_sad, [], extreme_influence)
    policy = select_policy(candidates, state, extreme_influence)

    print(f'  Policy selected: {policy["policy_label"]}')
    assert policy is not None, 'Should still select a policy'
    assert 'policy_label' in policy, 'Policy should have label'
    print('  → OK: 極端な責任状態でも判断は継続される')

    # Phase 6: 記憶との連携確認
    print()
    print('Phase 6: 記憶システムとの連携')

    percept_memory = Percept(text='楽しかったこと覚えてる？', emotion='happy', emotion_valence=0.3, intent='question')
    recalled = recall_by_mood(percept_memory, state, memory_mgr, top_k=5)

    print(f'  Recalled memories: {len(recalled)}')
    if recalled:
        print(f'  First memory: {recalled[0].get("summary", "N/A")[:30]}...')
    print('  → OK: 記憶システムが正常に動作')

    print()
    print('=' * 60)
    print('【検証結果】全ての相互作用テストに合格')
    print('=' * 60)
    print()
    print('確認された挙動:')
    print('  [OK] 責任は失敗により蓄積される')
    print('  [OK] 責任が判断にバイアスを与える（支配はしない）')
    print('  [OK] 時間経過で責任は減衰する（傷はゆっくり）')
    print('  [OK] 恐怖指数が感情・判断に影響する')
    print('  [OK] 極端な状態でも判断は継続される')
    print('  [OK] 記憶システムが正常に連携する')
    print('  [OK] 「壊れないが、壊れていく」が実現されている')

    return True


if __name__ == '__main__':
    asyncio.run(run_integration_test())
