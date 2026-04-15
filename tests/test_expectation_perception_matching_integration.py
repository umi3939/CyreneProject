"""
Tests for orchestrator integration of expectation_perception_matching.

Phase 2: orchestrator.py, orchestrator_5tick_phases.py, orchestrator_enrichment.py
への統合テスト。
"""

from __future__ import annotations

import time
import uuid

from psyche.expectation_formation import (
    ExpectationBasis,
    ExpectationCandidate,
    ExpectationSourceType,
    ExpectationStore,
)
from psyche.expectation_perception_matching import (
    ExpectationPerceptionMatcher,
    MatchingState,
)
from psyche.state import Percept

# =============================================================================
# Helpers
# =============================================================================

def _make_candidate(description: str = "test") -> ExpectationCandidate:
    return ExpectationCandidate(
        expectation_id=uuid.uuid4().hex[:12],
        source_type=ExpectationSourceType.REPETITION,
        basis=ExpectationBasis.PATTERN_CONTINUATION,
        description=description,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        freshness=0.8,
        strength=0.6,
        reference_count=0,
        evidence_ids=(),
        competing_ids=(),
        revision_count=0,
        undetermined_aspects=(),
    )


def _make_store(candidates: list[ExpectationCandidate] | None = None) -> ExpectationStore:
    cands = candidates or []
    return ExpectationStore(
        expectations=tuple(cands),
        evidence_links=(),
        total_expectations_created=len(cands),
        total_revisions=0,
        total_expirations=0,
        average_freshness=0.5,
        average_strength=0.5,
        active_expectation_count=len(cands),
        competing_pair_count=0,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        description="test store",
    )


# =============================================================================
# Orchestrator Import Tests
# =============================================================================

class TestOrchestratorImport:
    """orchestratorにインポートが追加されていること。"""

    def test_import_exists(self):
        """orchestrator.pyにexpectation_perception_matchingのインポートがあること。"""
        import psyche.orchestrator as orch_mod
        assert hasattr(orch_mod, "ExpectationPerceptionMatcher")
        assert hasattr(orch_mod, "MatchingState")

    def test_orchestrator_has_matcher(self):
        """PsycheOrchestratorにmatcherインスタンスがあること。"""
        import inspect

        from psyche.orchestrator import PsycheOrchestrator
        source = inspect.getsource(PsycheOrchestrator.__init__)
        assert "_expectation_perception_matcher" in source


# =============================================================================
# FieldDef Tests
# =============================================================================

class TestFieldDef:
    """save/loadフィールド定義のテスト。"""

    def test_field_definition_exists(self):
        """FIELD_DEFINITIONSにexpectation_perception_matching_stateが含まれること。"""
        from psyche.orchestrator import FIELD_DEFINITIONS
        keys = [fd.key for fd in FIELD_DEFINITIONS]
        assert "expectation_perception_matching_state" in keys

    def test_field_definition_correct_attr(self):
        """FieldDefのattr_pathが正しいこと。"""
        from psyche.orchestrator import FIELD_DEFINITIONS
        fd = next(
            f for f in FIELD_DEFINITIONS
            if f.key == "expectation_perception_matching_state"
        )
        assert fd.attr_path == "_expectation_perception_matcher"


# =============================================================================
# Phase Integration Tests
# =============================================================================

class TestPhaseIntegration:
    """Phase 26d2が存在すること。"""

    def test_phase_26d2_in_source(self):
        """orchestrator_5tick_phases.pyにPhase 26d2のコメントがあること。"""
        import inspect

        import psyche.orchestrator_5tick_phases as phases_mod
        source = inspect.getsource(phases_mod)
        assert "26d2" in source

    def test_phase_26d2_calls_matcher(self):
        """Phase 26d2でmatcherのprocessが呼ばれること。"""
        import inspect

        import psyche.orchestrator_5tick_phases as phases_mod
        source = inspect.getsource(phases_mod)
        assert "_expectation_perception_matcher" in source


# =============================================================================
# Enrichment Tests
# =============================================================================

class TestEnrichmentIntegration:
    """enrichmentにexpectation_perception_matchingが追加されていること。"""

    def test_enrichment_in_memory_section(self):
        """orchestrator_enrichment.pyの記憶・内省セクションに項目があること。"""
        import inspect

        import psyche.orchestrator_enrichment as enrich_mod
        source = inspect.getsource(enrich_mod)
        assert "_expectation_perception_matcher" in source
        assert "予期照合" in source


# =============================================================================
# Save/Load Roundtrip Tests
# =============================================================================

class TestSaveLoadRoundtrip:
    """matcherの状態がsave/load経由で保持されること。"""

    def test_matcher_state_roundtrip(self):
        """MatchingStateのto_dict/from_dictラウンドトリップ。"""
        matcher = ExpectationPerceptionMatcher()
        c = _make_candidate(description="test roundtrip")
        store = _make_store([c])
        percept = Percept(
            text="hello",
            meaning="greeting",
            intent="greeting",
            topics=["greeting"],
        )
        matcher.process(store, percept, tick=5)

        state = matcher.state
        d = state.to_dict()
        restored = MatchingState.from_dict(d)

        assert len(restored.records) == 1
        assert restored.records[0].tick == 5
        assert restored.records[0].expectation_description == "test roundtrip"
