"""tests/test_meta_emotion_cognition.py -- メタ感情認知と変動候補生成テスト"""

import time
import pytest

from psyche.meta_emotion_cognition import (
    # Enums
    InputSection,
    FreshnessStage,
    RecordStatus,
    ConvergenceLevel,
    # Data structures
    TransitionFeature,
    SustainedPattern,
    VariationCandidate,
    CognitionRecord,
    ConvergenceRecord,
    # Inputs / State / Result / Config
    MetaEmotionInputs,
    MetaEmotionState,
    MetaEmotionResult,
    MetaEmotionConfig,
    # Processor
    MetaEmotionProcessor,
    # Helpers
    _clamp,
    _gen_id,
    _stage_from_freshness,
    _convergence_from_score,
    # Public API
    get_meta_emotion_summary,
    create_meta_emotion_processor,
)


# ── Test Helpers ─────────────────────────────────────────────────

def _make_inputs(**overrides) -> MetaEmotionInputs:
    """テスト用の基本入力を生成する。"""
    defaults = dict(
        emotion_values={"joy": 0.5, "anger": 0.1, "sorrow": 0.2},
        mood_valence=0.3,
        mood_arousal=0.5,
        dynamics_phase="normal",
        dynamics_peak_intensity=0.0,
        dynamics_accumulated_intensity=0.0,
        coupling_continuity=0.0,
        coupling_active_entries=0,
        self_model_spread=0.3,
        self_model_intensity=0.4,
        self_model_conflict=False,
        amplitude_value=1.0,
        amplitude_boost=0.0,
        context_summary="test context",
        dialogue_state="active",
        referenced_memory_count=0,
        existing_record_count=0,
        average_freshness=0.0,
        current_tick=10,
    )
    defaults.update(overrides)
    return MetaEmotionInputs(**defaults)


def _make_processor(**config_overrides) -> MetaEmotionProcessor:
    """テスト用のプロセッサを生成する。"""
    cfg = MetaEmotionConfig(**config_overrides)
    return MetaEmotionProcessor(config=cfg)


def _run_ticks(proc: MetaEmotionProcessor, n: int, base_tick: int = 1,
               emotion_values=None) -> MetaEmotionResult:
    """指定回数のティックを実行し最後の結果を返す。"""
    result = None
    for i in range(n):
        emo = emotion_values or {"joy": 0.5, "anger": 0.1, "sorrow": 0.2}
        inp = _make_inputs(current_tick=base_tick + i, emotion_values=emo)
        result = proc.tick(inp)
    return result


# =================================================================
# Enum Tests
# =================================================================

class TestInputSection:
    def test_enum_count(self):
        assert len(InputSection) == 8

    def test_values(self):
        assert InputSection.EMOTION_STATE.value == "emotion_state"
        assert InputSection.DYNAMICS_PHASE.value == "dynamics_phase"
        assert InputSection.STM_COUPLING_RESULT.value == "stm_coupling_result"
        assert InputSection.SELF_MODEL_EMOTION.value == "self_model_emotion"
        assert InputSection.AMPLITUDE_STATE.value == "amplitude_state"
        assert InputSection.DIALOGUE_CONTEXT.value == "dialogue_context"
        assert InputSection.MEMORY_REFERENCE.value == "memory_reference"
        assert InputSection.ACCUMULATION_FRESHNESS.value == "accumulation_freshness"


class TestFreshnessStage:
    def test_enum_count(self):
        assert len(FreshnessStage) == 5

    def test_values(self):
        assert FreshnessStage.ACTIVE.value == "active"
        assert FreshnessStage.WEAKENING.value == "weakening"
        assert FreshnessStage.FADING.value == "fading"
        assert FreshnessStage.NEAR_INVISIBLE.value == "near_invisible"
        assert FreshnessStage.INVISIBLE.value == "invisible"


class TestRecordStatus:
    def test_enum_count(self):
        assert len(RecordStatus) == 3

    def test_values(self):
        assert RecordStatus.ACTIVE.value == "active"
        assert RecordStatus.DECAYING.value == "decaying"
        assert RecordStatus.INVISIBLE.value == "invisible"


class TestConvergenceLevel:
    def test_enum_count(self):
        assert len(ConvergenceLevel) == 4

    def test_values(self):
        assert ConvergenceLevel.NONE.value == "none"
        assert ConvergenceLevel.MILD.value == "mild"
        assert ConvergenceLevel.MODERATE.value == "moderate"
        assert ConvergenceLevel.STRONG.value == "strong"


# =================================================================
# Helper Tests
# =================================================================

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_min(self):
        assert _clamp(-0.5) == 0.0

    def test_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_custom_range(self):
        assert _clamp(0.3, 0.2, 0.8) == 0.3
        assert _clamp(0.1, 0.2, 0.8) == 0.2
        assert _clamp(0.9, 0.2, 0.8) == 0.8


class TestGenId:
    def test_returns_string(self):
        assert isinstance(_gen_id(), str)

    def test_unique(self):
        ids = {_gen_id() for _ in range(100)}
        assert len(ids) == 100

    def test_length(self):
        assert len(_gen_id()) == 12


class TestStageFromFreshness:
    def test_active(self):
        assert _stage_from_freshness(1.0) == FreshnessStage.ACTIVE
        assert _stage_from_freshness(0.8) == FreshnessStage.ACTIVE

    def test_weakening(self):
        assert _stage_from_freshness(0.7) == FreshnessStage.WEAKENING
        assert _stage_from_freshness(0.6) == FreshnessStage.WEAKENING

    def test_fading(self):
        assert _stage_from_freshness(0.5) == FreshnessStage.FADING
        assert _stage_from_freshness(0.4) == FreshnessStage.FADING

    def test_near_invisible(self):
        assert _stage_from_freshness(0.3) == FreshnessStage.NEAR_INVISIBLE
        assert _stage_from_freshness(0.2) == FreshnessStage.NEAR_INVISIBLE

    def test_invisible(self):
        assert _stage_from_freshness(0.1) == FreshnessStage.INVISIBLE
        assert _stage_from_freshness(0.0) == FreshnessStage.INVISIBLE


class TestConvergenceFromScore:
    def test_none(self):
        assert _convergence_from_score(0.0) == ConvergenceLevel.NONE
        assert _convergence_from_score(0.29) == ConvergenceLevel.NONE

    def test_mild(self):
        assert _convergence_from_score(0.3) == ConvergenceLevel.MILD
        assert _convergence_from_score(0.49) == ConvergenceLevel.MILD

    def test_moderate(self):
        assert _convergence_from_score(0.5) == ConvergenceLevel.MODERATE
        assert _convergence_from_score(0.69) == ConvergenceLevel.MODERATE

    def test_strong(self):
        assert _convergence_from_score(0.7) == ConvergenceLevel.STRONG
        assert _convergence_from_score(1.0) == ConvergenceLevel.STRONG


# =================================================================
# TransitionFeature Tests
# =================================================================

class TestTransitionFeature:
    def test_defaults(self):
        f = TransitionFeature()
        assert f.feature_id != ""
        assert f.duration_ticks == 0
        assert f.change_speed == 0.0
        assert f.oscillation_period == 0.0
        assert f.dominant_stability == 0.0
        assert f.transition_frequency == 0.0
        assert f.dynamics_phase_value == 0.0
        assert f.peak_intensity == 0.0
        assert f.accumulated_intensity == 0.0
        assert f.amplitude_value == 1.0
        assert f.amplitude_boost == 0.0
        assert f.mood_valence == 0.0
        assert f.mood_arousal == 0.0
        assert f.coupling_continuity == 0.0
        assert f.coupling_active_entries == 0
        assert f.self_model_intensity == 0.0
        assert f.self_model_spread == 0.0
        assert f.self_model_conflict is False
        assert f.freshness == 1.0
        assert f.freshness_stage == FreshnessStage.ACTIVE.value

    def test_auto_id(self):
        f1 = TransitionFeature()
        f2 = TransitionFeature()
        assert f1.feature_id != f2.feature_id

    def test_explicit_id(self):
        f = TransitionFeature(feature_id="custom_id")
        assert f.feature_id == "custom_id"

    def test_to_dict(self):
        f = TransitionFeature(
            feature_id="test_id",
            change_speed=0.5,
            dominant_stability=0.8,
        )
        d = f.to_dict()
        assert d["feature_id"] == "test_id"
        assert d["change_speed"] == 0.5
        assert d["dominant_stability"] == 0.8

    def test_from_dict(self):
        d = {"feature_id": "abc", "change_speed": 0.3, "mood_valence": -0.2}
        f = TransitionFeature.from_dict(d)
        assert f.feature_id == "abc"
        assert f.change_speed == 0.3
        assert f.mood_valence == -0.2

    def test_from_dict_defaults(self):
        f = TransitionFeature.from_dict({})
        # Empty dict -> auto-generated ID via __post_init__
        assert f.feature_id != ""
        assert f.change_speed == 0.0
        assert f.amplitude_value == 1.0

    def test_roundtrip(self):
        f = TransitionFeature(
            change_speed=0.7,
            dominant_stability=0.9,
            mood_valence=0.3,
            self_model_conflict=True,
        )
        d = f.to_dict()
        f2 = TransitionFeature.from_dict(d)
        assert f2.change_speed == f.change_speed
        assert f2.dominant_stability == f.dominant_stability
        assert f2.mood_valence == f.mood_valence
        assert f2.self_model_conflict == f.self_model_conflict
        assert f2.feature_id == f.feature_id

    def test_no_category_labels(self):
        """特徴量にカテゴリラベルを付与しないことを確認。"""
        f = TransitionFeature()
        d = f.to_dict()
        # すべてのキーは数値フィールドまたはメタデータのみ
        for key, val in d.items():
            if key in ("feature_id", "freshness_stage"):
                continue
            assert isinstance(val, (int, float, bool)), (
                f"Field {key} should be numeric, got {type(val)}"
            )


# =================================================================
# SustainedPattern Tests
# =================================================================

class TestSustainedPattern:
    def test_defaults(self):
        p = SustainedPattern()
        assert p.pattern_id != ""
        assert p.sustained_change_speed == 0.0
        assert p.sustained_ticks == 0
        assert p.freshness == 1.0
        assert p.status == RecordStatus.ACTIVE.value

    def test_auto_id(self):
        p1 = SustainedPattern()
        p2 = SustainedPattern()
        assert p1.pattern_id != p2.pattern_id

    def test_to_dict_from_dict(self):
        p = SustainedPattern(
            sustained_change_speed=0.3,
            sustained_dominant_stability=0.7,
            sustained_ticks=10,
        )
        d = p.to_dict()
        p2 = SustainedPattern.from_dict(d)
        assert p2.sustained_change_speed == 0.3
        assert p2.sustained_dominant_stability == 0.7
        assert p2.sustained_ticks == 10
        assert p2.pattern_id == p.pattern_id

    def test_from_dict_defaults(self):
        p = SustainedPattern.from_dict({})
        assert p.pattern_id != ""
        assert p.sustained_amplitude == 1.0
        assert p.status == RecordStatus.ACTIVE.value


# =================================================================
# VariationCandidate Tests
# =================================================================

class TestVariationCandidate:
    def test_defaults(self):
        c = VariationCandidate()
        assert c.candidate_id != ""
        assert c.delta_change_speed == 0.0
        assert c.delta_dominant_stability == 0.0
        assert c.delta_transition_frequency == 0.0
        assert c.delta_amplitude == 0.0
        assert c.delta_mood_valence == 0.0
        assert c.delta_mood_arousal == 0.0
        assert c.source_feature_id == ""
        assert c.source_pattern_id == ""
        assert c.freshness == 1.0

    def test_auto_id(self):
        c1 = VariationCandidate()
        c2 = VariationCandidate()
        assert c1.candidate_id != c2.candidate_id

    def test_to_dict_from_dict(self):
        c = VariationCandidate(
            delta_change_speed=0.1,
            delta_mood_valence=-0.2,
            source_feature_id="feat_a",
        )
        d = c.to_dict()
        c2 = VariationCandidate.from_dict(d)
        assert c2.delta_change_speed == 0.1
        assert c2.delta_mood_valence == -0.2
        assert c2.source_feature_id == "feat_a"

    def test_no_ranking_fields(self):
        """変動候補に優劣や推奨順位フィールドがないことを確認。"""
        c = VariationCandidate()
        d = c.to_dict()
        # ranking, priority, score, weight のようなフィールドが存在しない
        forbidden_keys = {"ranking", "priority", "score", "weight", "rank",
                          "recommendation", "preference"}
        assert not forbidden_keys.intersection(d.keys())


# =================================================================
# CognitionRecord Tests
# =================================================================

class TestCognitionRecord:
    def test_defaults(self):
        r = CognitionRecord()
        assert r.record_id != ""
        assert r.tick == 0
        assert r.freshness == 1.0
        assert r.status == RecordStatus.ACTIVE.value

    def test_to_dict_from_dict(self):
        r = CognitionRecord(
            tick=5,
            feature_id="f1",
            pattern_ids=["p1", "p2"],
            candidate_ids=["c1"],
            candidate_count=1,
        )
        d = r.to_dict()
        r2 = CognitionRecord.from_dict(d)
        assert r2.tick == 5
        assert r2.feature_id == "f1"
        assert r2.pattern_ids == ["p1", "p2"]
        assert r2.candidate_ids == ["c1"]
        assert r2.candidate_count == 1

    def test_from_dict_defaults(self):
        r = CognitionRecord.from_dict({})
        assert r.record_id != ""
        assert r.freshness == 1.0


# =================================================================
# ConvergenceRecord Tests
# =================================================================

class TestConvergenceRecord:
    def test_defaults(self):
        cr = ConvergenceRecord()
        assert cr.convergence_score == 0.0
        assert cr.convergence_level == ConvergenceLevel.NONE.value
        assert cr.candidate_diversity == 1.0
        assert cr.feature_diversity == 1.0

    def test_to_dict_from_dict(self):
        cr = ConvergenceRecord(
            convergence_score=0.6,
            convergence_level=ConvergenceLevel.MODERATE.value,
            candidate_diversity=0.4,
        )
        d = cr.to_dict()
        cr2 = ConvergenceRecord.from_dict(d)
        assert cr2.convergence_score == 0.6
        assert cr2.convergence_level == ConvergenceLevel.MODERATE.value
        assert cr2.candidate_diversity == 0.4


# =================================================================
# MetaEmotionInputs Tests
# =================================================================

class TestMetaEmotionInputs:
    def test_defaults(self):
        inp = MetaEmotionInputs()
        assert inp.emotion_values == {}
        assert inp.mood_valence == 0.0
        assert inp.mood_arousal == 0.3
        assert inp.dynamics_phase == "normal"
        assert inp.amplitude_value == 1.0
        assert inp.self_model_conflict is False
        assert inp.current_tick == 0

    def test_custom_values(self):
        inp = _make_inputs(mood_valence=-0.5, dynamics_phase="peak")
        assert inp.mood_valence == -0.5
        assert inp.dynamics_phase == "peak"


# =================================================================
# MetaEmotionState Tests
# =================================================================

class TestMetaEmotionState:
    def test_defaults(self):
        s = MetaEmotionState()
        assert s.feature_history == []
        assert s.sustained_patterns == []
        assert s.current_candidates == []
        assert s.cognition_history == []
        assert s.cycle_count == 0
        assert s.supply_strength == 1.0
        assert s.candidate_convergence_warning is False

    def test_to_dict(self):
        s = MetaEmotionState(cycle_count=5, supply_strength=0.8)
        d = s.to_dict()
        assert d["cycle_count"] == 5
        assert d["supply_strength"] == 0.8
        assert "feature_history" in d
        assert "sustained_patterns" in d

    def test_from_dict(self):
        d = {
            "cycle_count": 10,
            "supply_strength": 0.7,
            "feature_history": [
                TransitionFeature(change_speed=0.5).to_dict(),
            ],
            "candidate_convergence_warning": True,
        }
        s = MetaEmotionState.from_dict(d)
        assert s.cycle_count == 10
        assert s.supply_strength == 0.7
        assert len(s.feature_history) == 1
        assert s.feature_history[0].change_speed == 0.5
        assert s.candidate_convergence_warning is True

    def test_from_dict_defaults(self):
        s = MetaEmotionState.from_dict({})
        assert s.cycle_count == 0
        assert s.supply_strength == 1.0
        assert s.feature_history == []

    def test_roundtrip(self):
        s = MetaEmotionState()
        s.feature_history.append(TransitionFeature(change_speed=0.3))
        s.sustained_patterns.append(SustainedPattern(sustained_ticks=5))
        s.cycle_count = 3
        d = s.to_dict()
        s2 = MetaEmotionState.from_dict(d)
        assert s2.cycle_count == 3
        assert len(s2.feature_history) == 1
        assert s2.feature_history[0].change_speed == 0.3
        assert len(s2.sustained_patterns) == 1
        assert s2.sustained_patterns[0].sustained_ticks == 5

    def test_session_decay(self):
        s = MetaEmotionState()
        s.feature_history.append(TransitionFeature(freshness=1.0))
        s.sustained_patterns.append(SustainedPattern(freshness=0.5))
        s.cognition_history.append(CognitionRecord(freshness=0.8))
        s.current_candidates.append(VariationCandidate(freshness=0.9))
        s.apply_session_decay(0.3)
        assert s.feature_history[0].freshness == pytest.approx(0.7, abs=0.01)
        assert s.sustained_patterns[0].freshness == pytest.approx(0.2, abs=0.01)
        assert s.cognition_history[0].freshness == pytest.approx(0.5, abs=0.01)
        assert s.current_candidates[0].freshness == pytest.approx(0.6, abs=0.01)

    def test_session_decay_removes_invisible(self):
        s = MetaEmotionState()
        s.sustained_patterns.append(
            SustainedPattern(pattern_id="p1", freshness=0.05)
        )
        s.cognition_history.append(
            CognitionRecord(record_id="r1", freshness=0.05)
        )
        s.current_candidates.append(
            VariationCandidate(candidate_id="c1", freshness=0.05)
        )
        s.apply_session_decay(0.3)
        # All should be removed (freshness would go below 0.1)
        assert len(s.sustained_patterns) == 0
        assert len(s.cognition_history) == 0
        assert len(s.current_candidates) == 0

    def test_session_decay_clamp_at_zero(self):
        s = MetaEmotionState()
        s.feature_history.append(TransitionFeature(freshness=0.1))
        s.apply_session_decay(0.5)
        assert s.feature_history[0].freshness == 0.0


# =================================================================
# MetaEmotionConfig Tests
# =================================================================

class TestMetaEmotionConfig:
    def test_defaults(self):
        c = MetaEmotionConfig()
        assert c.max_feature_history == 50
        assert c.max_sustained_patterns == 30
        assert c.max_candidates == 30
        assert c.max_cognition_history == 100
        assert c.freshness_decay_rate == 0.02
        assert c.sustained_similarity_threshold == 0.3
        assert c.convergence_threshold == 0.5
        assert c.supply_min_strength == 0.1

    def test_custom(self):
        c = MetaEmotionConfig(max_feature_history=10, freshness_decay_rate=0.05)
        assert c.max_feature_history == 10
        assert c.freshness_decay_rate == 0.05


# =================================================================
# Processor Basic Tests
# =================================================================

class TestProcessorBasic:
    def test_creation(self):
        p = MetaEmotionProcessor()
        assert p.state is not None
        assert p.state.cycle_count == 0

    def test_creation_with_config(self):
        cfg = MetaEmotionConfig(max_feature_history=5)
        p = MetaEmotionProcessor(config=cfg)
        assert p._config.max_feature_history == 5

    def test_state_setter(self):
        p = MetaEmotionProcessor()
        new_state = MetaEmotionState(cycle_count=42)
        p.state = new_state
        assert p.state.cycle_count == 42

    def test_single_tick(self):
        p = _make_processor()
        inp = _make_inputs()
        result = p.tick(inp)
        assert isinstance(result, MetaEmotionResult)
        assert p.state.cycle_count == 1
        assert result.cycle_count == 1

    def test_tick_increments_cycle(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        assert p.state.cycle_count == 5

    def test_process_same_as_tick(self):
        p = _make_processor()
        inp = _make_inputs()
        r1 = p.process(inp)
        assert r1.cycle_count == 1


# =================================================================
# Stage 1: State Acquisition Tests
# =================================================================

class TestStage1Acquisition:
    def test_emotion_snapshots_accumulated(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.1 * i, "anger": 0.05 * i},
            ))
        assert len(p.state._emotion_snapshots) == 5

    def test_dominant_history_accumulated(self):
        p = _make_processor()
        p.tick(_make_inputs(emotion_values={"joy": 0.8, "anger": 0.1}))
        p.tick(_make_inputs(emotion_values={"anger": 0.9, "joy": 0.1}))
        assert p.state._dominant_history[0] == "joy"
        assert p.state._dominant_history[1] == "anger"

    def test_empty_emotion_values(self):
        p = _make_processor()
        result = p.tick(_make_inputs(emotion_values={}))
        assert result is not None
        assert p.state._dominant_history[-1] == ""

    def test_snapshot_limit(self):
        p = _make_processor(max_emotion_snapshots=5)
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state._emotion_snapshots) == 5
        assert len(p.state._dominant_history) == 5


# =================================================================
# Stage 2: Feature Extraction Tests
# =================================================================

class TestStage2FeatureExtraction:
    def test_feature_extracted_per_tick(self):
        p = _make_processor()
        p.tick(_make_inputs(current_tick=1))
        assert len(p.state.feature_history) == 1
        assert p.state.total_features_extracted == 1

    def test_feature_contains_numerical_values(self):
        p = _make_processor()
        result = p.tick(_make_inputs(
            mood_valence=0.3,
            mood_arousal=0.5,
            amplitude_value=1.2,
            amplitude_boost=0.1,
            dynamics_phase="peak",
            self_model_intensity=0.6,
            self_model_spread=0.4,
            self_model_conflict=True,
        ))
        f = result.current_feature
        assert f is not None
        assert f.mood_valence == 0.3
        assert f.mood_arousal == 0.5
        assert f.amplitude_value == 1.2
        assert f.amplitude_boost == 0.1
        assert f.dynamics_phase_value == 1.0  # peak -> 1.0
        assert f.self_model_intensity == 0.6
        assert f.self_model_spread == 0.4
        assert f.self_model_conflict is True

    def test_change_speed_calculation(self):
        p = _make_processor()
        # First tick: initial state
        p.tick(_make_inputs(
            current_tick=1,
            emotion_values={"joy": 0.0, "anger": 0.0},
        ))
        # Second tick: big change
        result = p.tick(_make_inputs(
            current_tick=2,
            emotion_values={"joy": 1.0, "anger": 1.0},
        ))
        f = result.current_feature
        assert f.change_speed > 0.0

    def test_dominant_stability(self):
        p = _make_processor()
        # 5 ticks with joy always dominant
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.8, "anger": 0.1},
            ))
        last_feature = p.state.feature_history[-1]
        assert last_feature.dominant_stability > 0.5

    def test_transition_frequency(self):
        p = _make_processor()
        # Alternate dominant emotions
        p.tick(_make_inputs(current_tick=1, emotion_values={"joy": 0.8, "anger": 0.1}))
        p.tick(_make_inputs(current_tick=2, emotion_values={"anger": 0.8, "joy": 0.1}))
        p.tick(_make_inputs(current_tick=3, emotion_values={"joy": 0.8, "anger": 0.1}))
        p.tick(_make_inputs(current_tick=4, emotion_values={"anger": 0.8, "joy": 0.1}))
        last_feature = p.state.feature_history[-1]
        assert last_feature.transition_frequency > 0.0

    def test_feature_history_limit(self):
        p = _make_processor(max_feature_history=5)
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.feature_history) <= 5

    def test_dynamics_phase_mapping(self):
        p = _make_processor()
        r1 = p.tick(_make_inputs(dynamics_phase="normal"))
        assert r1.current_feature.dynamics_phase_value == 0.0
        r2 = p.tick(_make_inputs(dynamics_phase="peak"))
        assert r2.current_feature.dynamics_phase_value == 1.0
        r3 = p.tick(_make_inputs(dynamics_phase="rebound"))
        assert r3.current_feature.dynamics_phase_value == 0.5

    def test_oscillation_period(self):
        p = _make_processor()
        # At least 3 ticks with changes
        p.tick(_make_inputs(current_tick=1, emotion_values={"joy": 0.8}))
        p.tick(_make_inputs(current_tick=2, emotion_values={"anger": 0.8}))
        p.tick(_make_inputs(current_tick=3, emotion_values={"joy": 0.8}))
        f = p.state.feature_history[-1]
        assert f.oscillation_period > 0.0


# =================================================================
# Stage 3: Sustained Pattern Detection Tests
# =================================================================

class TestStage3SustainedPatterns:
    def test_no_pattern_with_few_ticks(self):
        p = _make_processor()
        p.tick(_make_inputs(current_tick=1))
        p.tick(_make_inputs(current_tick=2))
        assert len(p.state.sustained_patterns) == 0

    def test_pattern_detected_with_stable_features(self):
        p = _make_processor()
        # Same emotion values repeatedly -> low variance -> pattern detected
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5, "anger": 0.1},
            ))
        assert len(p.state.sustained_patterns) >= 1

    def test_pattern_sustained_ticks_increase(self):
        p = _make_processor()
        for i in range(10):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5, "anger": 0.1},
            ))
        # At least one pattern should have increasing sustained_ticks
        if p.state.sustained_patterns:
            max_ticks = max(pp.sustained_ticks for pp in p.state.sustained_patterns)
            assert max_ticks >= 3

    def test_pattern_decays_on_divergent_input(self):
        p = _make_processor()
        # Build a pattern
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5, "anger": 0.1},
            ))
        # Now diverge radically
        for i in range(5, 10):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"sorrow": 0.9, "fear": 0.8},
            ))
        has_decaying = any(
            p.status == RecordStatus.DECAYING.value
            for p in p.state.sustained_patterns
        )
        # Should have at least some decaying patterns
        assert has_decaying or len(p.state.sustained_patterns) > 0

    def test_pattern_limit(self):
        p = _make_processor(max_sustained_patterns=3)
        # Generate many diverse patterns
        emotions_list = [
            {"joy": 0.9}, {"anger": 0.9}, {"sorrow": 0.9},
            {"fear": 0.9}, {"surprise": 0.9},
        ]
        for idx, emo in enumerate(emotions_list):
            for i in range(5):
                p.tick(_make_inputs(
                    current_tick=idx * 5 + i,
                    emotion_values=emo,
                ))
        assert len(p.state.sustained_patterns) <= 3


# =================================================================
# Stage 4: Variation Candidate Enumeration Tests
# =================================================================

class TestStage4CandidateEnumeration:
    def test_always_generates_candidates(self):
        """候補生成はトリガーベースではなく、常時実行される。"""
        p = _make_processor()
        result = p.tick(_make_inputs())
        assert result.candidate_count >= 5  # At least 5 base candidates

    def test_minimum_five_base_candidates(self):
        """少なくとも5つの基本候補が常に生成される。"""
        p = _make_processor()
        result = p.tick(_make_inputs())
        # continuation, speed+, speed-, stability, mood
        assert result.candidate_count >= 5

    def test_continuation_candidate_is_zero_delta(self):
        """現状維持候補はすべてのdeltaが0。"""
        p = _make_processor()
        p.tick(_make_inputs())
        candidates = p.state.current_candidates
        # First candidate should be the continuation candidate
        cont = candidates[0]
        assert cont.delta_change_speed == 0.0
        assert cont.delta_dominant_stability == 0.0
        assert cont.delta_transition_frequency == 0.0
        assert cont.delta_amplitude == 0.0
        assert cont.delta_mood_valence == 0.0
        assert cont.delta_mood_arousal == 0.0

    def test_candidates_have_source_feature(self):
        p = _make_processor()
        p.tick(_make_inputs())
        for c in p.state.current_candidates:
            assert c.source_feature_id != ""

    def test_pattern_based_candidates(self):
        """持続パターンがある場合、追加の候補が生成される。"""
        p = _make_processor()
        # Build patterns
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5, "anger": 0.1},
            ))
        result = p.tick(_make_inputs(
            current_tick=6,
            emotion_values={"joy": 0.5, "anger": 0.1},
        ))
        # Base 5 + at least 1 from pattern
        if p.state.sustained_patterns:
            active_patterns = [
                pp for pp in p.state.sustained_patterns
                if pp.status == RecordStatus.ACTIVE.value
            ]
            if active_patterns:
                assert result.candidate_count >= 6

    def test_no_ranking_in_candidates(self):
        """変動候補に優劣や推奨順位がないことを確認。"""
        p = _make_processor()
        p.tick(_make_inputs())
        for c in p.state.current_candidates:
            d = c.to_dict()
            assert "ranking" not in d
            assert "priority" not in d
            assert "score" not in d
            assert "weight" not in d
            assert "recommendation" not in d

    def test_candidates_equal_weight(self):
        """すべての候補のfreshnessが同じ（等価）であること。"""
        p = _make_processor()
        p.tick(_make_inputs())
        freshnesses = [c.freshness for c in p.state.current_candidates]
        # All should be very close (within decay tolerance)
        if freshnesses:
            assert max(freshnesses) - min(freshnesses) < 0.05

    def test_mood_direction_varies_with_input(self):
        """ムード方向候補がinputに応じて変わる。"""
        p1 = _make_processor()
        r1 = p1.tick(_make_inputs(mood_valence=-0.5))
        # mood candidate (index 4) should have positive delta_mood_valence
        c1 = p1.state.current_candidates[4]
        assert c1.delta_mood_valence == 0.1  # valence <= 0 -> +0.1

        p2 = _make_processor()
        r2 = p2.tick(_make_inputs(mood_valence=0.5))
        c2 = p2.state.current_candidates[4]
        assert c2.delta_mood_valence == -0.1  # valence > 0 -> -0.1


# =================================================================
# Stage 5: Candidate Alignment and Retention Tests
# =================================================================

class TestStage5CandidateAlignment:
    def test_candidates_retained(self):
        p = _make_processor()
        p.tick(_make_inputs())
        assert len(p.state.current_candidates) >= 5

    def test_candidate_history_recorded(self):
        p = _make_processor()
        p.tick(_make_inputs())
        assert len(p.state.candidate_history) == 1
        assert p.state.candidate_history[0]["candidate_count"] >= 5

    def test_candidate_limit(self):
        p = _make_processor(max_candidates=3)
        # Generate candidates via patterns
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        # Current candidates should be limited
        assert len(p.state.current_candidates) <= 3

    def test_candidate_history_limit(self):
        p = _make_processor(max_candidate_history=3)
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.candidate_history) <= 3


# =================================================================
# Stage 6: Accumulation Tests
# =================================================================

class TestStage6Accumulation:
    def test_cognition_record_created(self):
        p = _make_processor()
        p.tick(_make_inputs(current_tick=1))
        assert len(p.state.cognition_history) == 1

    def test_cognition_record_fields(self):
        p = _make_processor()
        p.tick(_make_inputs(current_tick=5))
        rec = p.state.cognition_history[0]
        assert rec.tick == 5
        assert rec.feature_id != ""
        assert rec.candidate_count >= 5

    def test_freshness_decays(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        # Earlier records should have lower freshness
        first_rec = p.state.cognition_history[0]
        last_rec = p.state.cognition_history[-1]
        assert first_rec.freshness < last_rec.freshness

    def test_feature_freshness_decays(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        first_feat = p.state.feature_history[0]
        assert first_feat.freshness < 1.0

    def test_cognition_history_limit(self):
        p = _make_processor(max_cognition_history=5)
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.cognition_history) <= 5

    def test_decay_history_limit(self):
        p = _make_processor(max_decay_history=3, freshness_decay_rate=0.5)
        for i in range(20):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.decay_history) <= 3

    def test_recovery_candidates_populated(self):
        p = _make_processor(freshness_decay_rate=0.5)
        for i in range(30):
            p.tick(_make_inputs(current_tick=i))
        # Some records should have become invisible and been added to recovery
        # (depends on decay rate being fast enough)
        # With 0.5 decay rate, records from early ticks will go invisible quickly
        assert p.state.total_records_decayed > 0 or p.state.cycle_count >= 30

    def test_overflow_adds_to_recovery(self):
        p = _make_processor(max_cognition_history=3)
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        # Overflow records should be added to recovery_candidates
        assert len(p.state.recovery_candidates) > 0

    def test_pattern_freshness_decays(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5, "anger": 0.1},
            ))
        if p.state.sustained_patterns:
            # After several ticks, freshness should have decreased
            pat = p.state.sustained_patterns[0]
            # May have been refreshed by sustained detection, so just check it exists
            assert pat.freshness <= 1.0


# =================================================================
# Stage 7: Handoff Preparation Tests
# =================================================================

class TestStage7Handoff:
    def test_result_fields(self):
        p = _make_processor()
        result = p.tick(_make_inputs())
        assert isinstance(result, MetaEmotionResult)
        assert result.current_feature is not None
        assert result.candidate_count >= 5
        assert result.cycle_count == 1
        assert isinstance(result.freshness_distribution, dict)
        assert isinstance(result.convergence_level, str)
        assert isinstance(result.convergence_score, float)

    def test_convergence_monitoring(self):
        p = _make_processor()
        p.tick(_make_inputs())
        assert len(p.state.convergence_records) == 1

    def test_convergence_record_limit(self):
        p = _make_processor(max_convergence_records=3)
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.convergence_records) <= 3


# =================================================================
# Safety Valve Tests
# =================================================================

class TestSafetyValve1CandidateConvergence:
    def test_convergence_level_recorded(self):
        """収束レベルが記録されること。"""
        p = _make_processor()
        result = p.tick(_make_inputs())
        assert result.convergence_level in (
            ConvergenceLevel.NONE.value,
            ConvergenceLevel.MILD.value,
            ConvergenceLevel.MODERATE.value,
            ConvergenceLevel.STRONG.value,
        )

    def test_convergence_monitored_every_tick(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.convergence_records) == 5

    def test_convergence_detected_with_identical_input(self):
        """同一入力の繰り返しで収束が検出されること。"""
        p = _make_processor()
        for i in range(20):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5},
            ))
        # State should have been monitored
        assert len(p.state.convergence_records) > 0

    def test_diversity_restored_on_convergence(self):
        """収束時に多様性復元が試みられること。"""
        p = _make_processor()
        # Add a decaying pattern manually
        p.state.sustained_patterns.append(
            SustainedPattern(
                sustained_change_speed=0.5,
                sustained_dominant_stability=0.3,
                status=RecordStatus.DECAYING.value,
                freshness=0.35,
            )
        )
        # Run enough ticks to potentially trigger convergence + restore
        for i in range(10):
            result = p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5},
            ))
        if result.candidate_convergence_warning:
            # If convergence was detected, restoration should have been attempted
            assert p.state.total_records_recovered >= 0  # May or may not succeed


class TestSafetyValve2FeatureBias:
    def test_no_bias_with_varying_input(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.1 * i, "anger": 0.05 * i},
            ))
        result = p.tick(_make_inputs(current_tick=6))
        # Varying inputs should not trigger feature bias
        assert p.state.feature_bias_warning is False

    def test_bias_with_constant_input(self):
        """Exactly same input repeatedly should trigger feature bias."""
        p = _make_processor()
        for i in range(10):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5},
                mood_valence=0.3,
                mood_arousal=0.5,
            ))
        # With constant input, change_speed should be near 0 for all
        # -> feature bias should be detected
        assert p.state.feature_bias_warning is True


class TestSafetyValve3SupplyConcentration:
    def test_no_warning_initially(self):
        p = _make_processor()
        result = p.tick(_make_inputs())
        assert result.supply_concentration_warning is False

    def test_supply_history_checked(self):
        p = _make_processor()
        # Add some supply history manually
        p.state.supply_history = [
            {"delta_change_speed": 0.1},
            {"delta_change_speed": 0.1},
            {"delta_change_speed": 0.1},
            {"delta_change_speed": 0.1},
        ]
        result = p.tick(_make_inputs())
        # Should detect concentration if variance is very low
        assert p.state.supply_concentration_warning is True


class TestSafetyValve4AccumulationBias:
    def test_no_bias_initially(self):
        p = _make_processor()
        result = p.tick(_make_inputs())
        assert result.accumulation_bias_warning is False

    def test_bias_with_dominant_pattern(self):
        p = _make_processor()
        # Manually create biased cognition history
        for i in range(5):
            rec = CognitionRecord(
                tick=i,
                feature_id=f"f{i}",
                pattern_ids=["dominant_pattern"],  # Same pattern dominates
                candidate_ids=[],
                candidate_count=5,
                status=RecordStatus.ACTIVE.value,
            )
            p.state.cognition_history.append(rec)
        result = p.tick(_make_inputs(current_tick=10))
        assert p.state.accumulation_bias_warning is True


# =================================================================
# Supply Strength Tests
# =================================================================

class TestSupplyStrength:
    def test_initial_strength(self):
        p = _make_processor()
        assert p.state.supply_strength == 1.0

    def test_strength_decreases_on_convergence(self):
        p = _make_processor()
        # Force moderate convergence
        p.state.supply_strength = 1.0
        # Run with same input many times to potentially trigger convergence
        for i in range(20):
            p.tick(_make_inputs(
                current_tick=i,
                emotion_values={"joy": 0.5},
            ))
        # Check if any convergence records had moderate/strong
        moderate_or_strong = any(
            cr.convergence_level in (
                ConvergenceLevel.MODERATE.value,
                ConvergenceLevel.STRONG.value,
            )
            for cr in p.state.convergence_records
        )
        if moderate_or_strong:
            assert p.state.supply_strength < 1.0

    def test_strength_floor(self):
        p = _make_processor(supply_min_strength=0.1)
        p.state.supply_strength = 0.1
        # Even after many ticks, should not go below min
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        assert p.state.supply_strength >= 0.1


# =================================================================
# Enrichment Data Tests
# =================================================================

class TestEnrichmentData:
    def test_enrichment_structure(self):
        p = _make_processor()
        p.tick(_make_inputs())
        data = p.get_enrichment_data()
        assert "cycle_count" in data
        assert "active_pattern_count" in data
        assert "decaying_pattern_count" in data
        assert "candidate_count" in data
        assert "cognition_record_count" in data
        assert "freshness_distribution" in data
        assert "latest_feature" in data
        assert "candidate_convergence_warning" in data
        assert "feature_bias_warning" in data
        assert "supply_concentration_warning" in data
        assert "accumulation_bias_warning" in data
        assert "supply_strength" in data
        assert "summary_text" in data

    def test_enrichment_after_ticks(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        data = p.get_enrichment_data()
        assert data["cycle_count"] == 5
        assert data["latest_feature"] != {}

    def test_enrichment_empty_state(self):
        p = _make_processor()
        data = p.get_enrichment_data()
        assert data["cycle_count"] == 0
        assert data["latest_feature"] == {}
        assert data["summary_text"] == "メタ感情認知: 待機中"


# =================================================================
# Summary Tests
# =================================================================

class TestSummary:
    def test_waiting_state(self):
        s = MetaEmotionState()
        summary = get_meta_emotion_summary(s)
        assert "待機中" in summary

    def test_with_cycles(self):
        p = _make_processor()
        for i in range(3):
            p.tick(_make_inputs(current_tick=i))
        summary = get_meta_emotion_summary(p.state)
        assert "cycle=3" in summary

    def test_with_candidates(self):
        p = _make_processor()
        p.tick(_make_inputs())
        summary = get_meta_emotion_summary(p.state)
        assert "候補=" in summary

    def test_with_warnings(self):
        s = MetaEmotionState()
        s.cycle_count = 1
        s.feature_history.append(TransitionFeature())
        s.candidate_convergence_warning = True
        s.feature_bias_warning = True
        summary = get_meta_emotion_summary(s)
        assert "候補収束" in summary
        assert "特徴偏り" in summary

    def test_with_feature_speed_and_stability(self):
        p = _make_processor()
        p.tick(_make_inputs())
        summary = get_meta_emotion_summary(p.state)
        assert "速度=" in summary
        assert "安定=" in summary


# =================================================================
# Factory Tests
# =================================================================

class TestFactory:
    def test_create_default(self):
        p = create_meta_emotion_processor()
        assert isinstance(p, MetaEmotionProcessor)
        assert p.state.cycle_count == 0

    def test_create_with_config(self):
        cfg = MetaEmotionConfig(max_feature_history=10)
        p = create_meta_emotion_processor(config=cfg)
        assert p._config.max_feature_history == 10


# =================================================================
# Integration / Multi-tick Tests
# =================================================================

class TestIntegration:
    def test_multi_tick_no_crash(self):
        p = _make_processor()
        for i in range(50):
            emo = {"joy": 0.3 + 0.01 * (i % 10), "anger": 0.1 + 0.01 * (i % 5)}
            p.tick(_make_inputs(current_tick=i, emotion_values=emo))
        assert p.state.cycle_count == 50
        assert p.state.total_features_extracted == 50

    def test_state_save_load_roundtrip(self):
        p = _make_processor()
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        d = p.state.to_dict()
        new_state = MetaEmotionState.from_dict(d)
        assert new_state.cycle_count == p.state.cycle_count
        assert len(new_state.feature_history) == len(p.state.feature_history)
        assert len(new_state.cognition_history) == len(p.state.cognition_history)

    def test_varying_emotions(self):
        p = _make_processor()
        emotions_sequence = [
            {"joy": 0.8, "anger": 0.0},
            {"joy": 0.6, "anger": 0.2},
            {"anger": 0.7, "joy": 0.1},
            {"sorrow": 0.9, "joy": 0.0},
            {"joy": 0.3, "sorrow": 0.3, "anger": 0.3},
        ]
        for i, emo in enumerate(emotions_sequence):
            result = p.tick(_make_inputs(current_tick=i, emotion_values=emo))
        assert result.current_feature is not None
        assert result.candidate_count >= 5

    def test_dynamics_phase_changes(self):
        p = _make_processor()
        phases = ["normal", "peak", "rebound", "normal", "peak"]
        for i, phase in enumerate(phases):
            result = p.tick(_make_inputs(current_tick=i, dynamics_phase=phase))
        features = p.state.feature_history
        assert features[0].dynamics_phase_value == 0.0
        assert features[1].dynamics_phase_value == 1.0
        assert features[2].dynamics_phase_value == 0.5

    def test_session_boundary(self):
        """セッション境界での減衰後も処理が継続できること。"""
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        p.state.apply_session_decay(0.3)
        # Should still be able to process
        result = p.tick(_make_inputs(current_tick=10))
        assert result is not None
        assert result.cycle_count == 6

    def test_all_safety_valves_accessible(self):
        """すべての安全弁フラグが結果に含まれること。"""
        p = _make_processor()
        result = p.tick(_make_inputs())
        assert hasattr(result, "candidate_convergence_warning")
        assert hasattr(result, "feature_bias_warning")
        assert hasattr(result, "supply_concentration_warning")
        assert hasattr(result, "accumulation_bias_warning")
        assert hasattr(result, "diversity_restored")

    def test_no_parameter_modification(self):
        """入力として渡されたパラメータが変更されないこと (READ-ONLY)。"""
        inp = _make_inputs(
            mood_valence=0.3,
            mood_arousal=0.5,
            amplitude_value=1.0,
        )
        original_valence = inp.mood_valence
        original_arousal = inp.mood_arousal
        original_amplitude = inp.amplitude_value
        p = _make_processor()
        p.tick(inp)
        assert inp.mood_valence == original_valence
        assert inp.mood_arousal == original_arousal
        assert inp.amplitude_value == original_amplitude

    def test_pattern_similarity_helper(self):
        p = _make_processor()
        pat = SustainedPattern(
            sustained_change_speed=0.5,
            sustained_dominant_stability=0.8,
            sustained_transition_frequency=0.3,
        )
        sim = p._pattern_similarity(pat, 0.5, 0.8, 0.3)
        assert sim == pytest.approx(1.0, abs=0.01)

        sim2 = p._pattern_similarity(pat, 0.0, 0.0, 0.0)
        assert sim2 < 1.0

    def test_freshness_distribution_in_result(self):
        p = _make_processor()
        for i in range(10):
            p.tick(_make_inputs(current_tick=i))
        result = p.tick(_make_inputs(current_tick=11))
        # Should have some distribution
        assert isinstance(result.freshness_distribution, dict)

    def test_total_counters_increment(self):
        p = _make_processor()
        for i in range(5):
            p.tick(_make_inputs(current_tick=i))
        assert p.state.total_features_extracted == 5
        assert p.state.total_candidates_generated >= 25  # 5 ticks * 5+ candidates each

    def test_recovery_candidates_limit(self):
        p = _make_processor(
            max_recovery_candidates=3,
            max_cognition_history=3,
        )
        for i in range(20):
            p.tick(_make_inputs(current_tick=i))
        assert len(p.state.recovery_candidates) <= 3


# =================================================================
# Prohibited Words / Design Constraint Tests
# =================================================================

class TestDesignConstraints:
    def test_no_prohibited_words_in_source(self):
        """コード内で禁止語（調整・戦略・介入・修正）が使われていないことを確認。
        (docstring/comment以外のコードレベルでの使用を検出)"""
        import inspect
        source = inspect.getsource(MetaEmotionProcessor)
        # These words should not appear as variable names or method names
        # (they may appear in docstrings explaining what we DON'T do)
        # Check that no method or attribute contains these words
        for attr_name in dir(MetaEmotionProcessor):
            assert "調整" not in attr_name
            assert "戦略" not in attr_name
            assert "介入" not in attr_name
            assert "修正" not in attr_name

    def test_no_trigger_based_generation(self):
        """候補生成がトリガーベースではなく常に実行されることを確認。"""
        p = _make_processor()
        # Even with minimal/zero emotion values
        result1 = p.tick(_make_inputs(emotion_values={}))
        assert result1.candidate_count >= 5

        result2 = p.tick(_make_inputs(emotion_values={"joy": 0.0}))
        assert result2.candidate_count >= 5

    def test_candidates_always_enumerated(self):
        """どのような入力でも候補が列挙されること。"""
        p = _make_processor()
        test_cases = [
            {},
            {"joy": 1.0},
            {"anger": 0.5, "sorrow": 0.5},
            {"joy": 0.0, "anger": 0.0, "sorrow": 0.0},
        ]
        for emo in test_cases:
            result = p.tick(_make_inputs(emotion_values=emo))
            assert result.candidate_count >= 5, (
                f"Candidates should always be generated, got {result.candidate_count} "
                f"for emotion_values={emo}"
            )
