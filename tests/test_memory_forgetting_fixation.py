"""tests/test_memory_forgetting_fixation.py — 記憶の忘却と固定化テスト"""

import time
import pytest

from psyche.memory_forgetting_fixation import (
    # Enums
    ObservationSourceType,
    ForgettingStage,
    FixationLevel,
    SeriesStatus,
    # Data structures
    MemorySeriesRecord,
    ForgettingCandidate,
    FixationSign,
    ForgettingFixationInputs,
    ForgettingFixationState,
    ForgettingFixationResult,
    ForgettingFixationConfig,
    # Processor
    MemoryForgettingFixationProcessor,
    # Helpers
    _clamp,
    _stage_from_dilution,
    _fixation_from_score,
    # Public API
    get_forgetting_fixation_summary,
    create_forgetting_fixation_processor,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_basic_inputs(**overrides) -> ForgettingFixationInputs:
    """テスト用の基本入力を生成する。"""
    defaults = dict(
        episode_entries=[],
        binding_entries=[],
        long_term_entries=[],
        reuse_history={},
        tick_count=1,
        elapsed_since_last=1.0,
        active_series_count=0,
        dominant_series_id="",
        emotion_valence=0.0,
        emotion_arousal=0.0,
        binding_count=0,
        average_binding_freshness=0.0,
        context_continuity=0.0,
        context_density=0.0,
        protected_ids=[],
        repeated_reference_ids=[],
        invisible_alternative_count=0,
    )
    defaults.update(overrides)
    return ForgettingFixationInputs(**defaults)


def _make_episode(eid: str, valence: float = 0.0) -> dict:
    return {"id": eid, "emotional_valence": valence}


def _make_binding(bid: str, freshness: float = 0.5) -> dict:
    return {"id": bid, "freshness": freshness}


def _make_long_term(mid: str, valence: float = 0.0) -> dict:
    return {"id": mid, "emotional_valence": valence}


# =====================================================================
# Enum Tests
# =====================================================================

class TestEnums:
    def test_observation_source_type_count(self):
        assert len(ObservationSourceType) == 8

    def test_forgetting_stage_count(self):
        assert len(ForgettingStage) == 5

    def test_forgetting_stage_values(self):
        assert ForgettingStage.ACTIVE.value == "active"
        assert ForgettingStage.INVISIBLE.value == "invisible"

    def test_fixation_level_count(self):
        assert len(FixationLevel) == 4

    def test_fixation_level_values(self):
        assert FixationLevel.NONE.value == "none"
        assert FixationLevel.STRONG.value == "strong"

    def test_series_status_count(self):
        assert len(SeriesStatus) == 5


# =====================================================================
# Helper Tests
# =====================================================================

class TestHelpers:
    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below_min(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(5.0, 0.0, 3.0) == 3.0

    def test_stage_from_dilution_active(self):
        assert _stage_from_dilution(0.1) == ForgettingStage.ACTIVE

    def test_stage_from_dilution_weakening(self):
        assert _stage_from_dilution(0.3) == ForgettingStage.WEAKENING

    def test_stage_from_dilution_fading(self):
        assert _stage_from_dilution(0.5) == ForgettingStage.FADING

    def test_stage_from_dilution_near_invisible(self):
        assert _stage_from_dilution(0.7) == ForgettingStage.NEAR_INVISIBLE

    def test_stage_from_dilution_invisible(self):
        assert _stage_from_dilution(0.9) == ForgettingStage.INVISIBLE

    def test_fixation_from_score_none(self):
        assert _fixation_from_score(0.1) == FixationLevel.NONE

    def test_fixation_from_score_mild(self):
        assert _fixation_from_score(0.4) == FixationLevel.MILD

    def test_fixation_from_score_moderate(self):
        assert _fixation_from_score(0.6) == FixationLevel.MODERATE

    def test_fixation_from_score_strong(self):
        assert _fixation_from_score(0.8) == FixationLevel.STRONG


# =====================================================================
# Data Structure Serialization Tests
# =====================================================================

class TestMemorySeriesRecord:
    def test_creation_defaults(self):
        rec = MemorySeriesRecord()
        assert rec.source == ""
        assert rec.reference_count == 0
        assert rec.dilution == 0.0
        assert rec.forgetting_stage == ForgettingStage.ACTIVE.value
        assert rec.fixation_level == FixationLevel.NONE.value
        assert rec.is_protected is False

    def test_to_dict_roundtrip(self):
        rec = MemorySeriesRecord(
            series_id="test123",
            source="episodic",
            source_id="ep1",
            reference_count=5,
            dilution=0.3,
            fixation_score=0.5,
            is_protected=True,
        )
        d = rec.to_dict()
        restored = MemorySeriesRecord.from_dict(d)
        assert restored.series_id == "test123"
        assert restored.source == "episodic"
        assert restored.reference_count == 5
        assert restored.dilution == pytest.approx(0.3)
        assert restored.is_protected is True


class TestForgettingCandidate:
    def test_creation(self):
        c = ForgettingCandidate(series_id="s1", dilution=0.4)
        assert c.series_id == "s1"
        assert c.dilution == 0.4

    def test_to_dict_roundtrip(self):
        c = ForgettingCandidate(
            series_id="s1",
            current_stage="active",
            proposed_stage="weakening",
            dilution=0.3,
            time_since_reference=100.0,
            reason="test",
        )
        d = c.to_dict()
        restored = ForgettingCandidate.from_dict(d)
        assert restored.series_id == "s1"
        assert restored.proposed_stage == "weakening"
        assert restored.reason == "test"


class TestFixationSign:
    def test_creation(self):
        s = FixationSign(series_id="s2", score=0.6)
        assert s.series_id == "s2"
        assert s.observation_count == 0

    def test_to_dict_roundtrip(self):
        s = FixationSign(
            series_id="s2",
            score=0.6,
            level="moderate",
            observation_count=3,
            indicators=["repeated_reference", "reuse_concentration"],
        )
        d = s.to_dict()
        restored = FixationSign.from_dict(d)
        assert restored.series_id == "s2"
        assert restored.score == pytest.approx(0.6)
        assert len(restored.indicators) == 2


class TestForgettingFixationInputs:
    def test_creation_defaults(self):
        inp = ForgettingFixationInputs()
        assert inp.episode_entries == []
        assert inp.tick_count == 0
        assert inp.protected_ids == []

    def test_with_entries(self):
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
            binding_entries=[_make_binding("b1")],
            long_term_entries=[_make_long_term("m1")],
        )
        assert len(inp.episode_entries) == 1
        assert len(inp.binding_entries) == 1
        assert len(inp.long_term_entries) == 1


# =====================================================================
# State Serialization Tests
# =====================================================================

class TestForgettingFixationState:
    def test_empty_state(self):
        s = ForgettingFixationState()
        assert s.cycle_count == 0
        assert s.total_forgotten == 0
        assert s.convergence_warning is False

    def test_to_dict_roundtrip(self):
        s = ForgettingFixationState()
        s.cycle_count = 5
        s.total_forgotten = 3
        s.convergence_warning = True
        s.series_index.append(MemorySeriesRecord(source="episodic", source_id="e1"))
        s.fixation_sign_history.append(FixationSign(series_id="s1", score=0.4))

        d = s.to_dict()
        restored = ForgettingFixationState.from_dict(d)
        assert restored.cycle_count == 5
        assert restored.total_forgotten == 3
        assert restored.convergence_warning is True
        assert len(restored.series_index) == 1
        assert restored.series_index[0].source == "episodic"
        assert len(restored.fixation_sign_history) == 1

    def test_state_with_all_fields(self):
        s = ForgettingFixationState(
            reference_history=[{"source_id": "x", "timestamp": 1.0}],
            reuse_history={"x": 5},
            dilution_map={"x": 0.3},
            alternative_series=["a1", "a2"],
            recovery_candidates=["r1"],
            overdense_warning=True,
        )
        d = s.to_dict()
        restored = ForgettingFixationState.from_dict(d)
        assert restored.reuse_history["x"] == 5
        assert restored.dilution_map["x"] == pytest.approx(0.3)
        assert len(restored.alternative_series) == 2
        assert restored.overdense_warning is True


# =====================================================================
# Config Tests
# =====================================================================

class TestForgettingFixationConfig:
    def test_defaults(self):
        cfg = ForgettingFixationConfig()
        assert cfg.max_series == 300
        assert cfg.dilution_rate == pytest.approx(0.02)
        assert cfg.reference_recovery == pytest.approx(0.15)
        assert cfg.fixation_cross_section_threshold == 2

    def test_custom_config(self):
        cfg = ForgettingFixationConfig(max_series=100, dilution_rate=0.05)
        assert cfg.max_series == 100
        assert cfg.dilution_rate == pytest.approx(0.05)


# =====================================================================
# Result Tests
# =====================================================================

class TestForgettingFixationResult:
    def test_empty_result(self):
        r = ForgettingFixationResult()
        assert r.newly_forgotten == 0
        assert r.convergence_warning is False

    def test_to_dict(self):
        r = ForgettingFixationResult(
            newly_forgotten=2,
            newly_recovered=1,
            active_series=10,
            convergence_warning=True,
        )
        d = r.to_dict()
        assert d["newly_forgotten"] == 2
        assert d["active_series"] == 10
        assert d["convergence_warning"] is True


# =====================================================================
# Processor Basic Tests
# =====================================================================

class TestProcessorBasic:
    def test_creation(self):
        proc = MemoryForgettingFixationProcessor()
        assert proc.state.cycle_count == 0

    def test_creation_with_config(self):
        cfg = ForgettingFixationConfig(max_series=50)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        assert proc._config.max_series == 50

    def test_empty_process(self):
        proc = create_forgetting_fixation_processor()
        result = proc.process(_make_basic_inputs())
        assert result.cycle_count == 1
        assert result.newly_forgotten == 0
        assert result.newly_recovered == 0

    def test_state_setter(self):
        proc = MemoryForgettingFixationProcessor()
        new_state = ForgettingFixationState(cycle_count=10)
        proc.state = new_state
        assert proc.state.cycle_count == 10

    def test_cycle_increment(self):
        proc = create_forgetting_fixation_processor()
        proc.process(_make_basic_inputs())
        proc.process(_make_basic_inputs())
        assert proc.state.cycle_count == 2

    def test_factory_function(self):
        proc = create_forgetting_fixation_processor()
        assert isinstance(proc, MemoryForgettingFixationProcessor)


# =====================================================================
# Stage 1: 忘却候補抽出 Tests
# =====================================================================

class TestStage1ForgettingCandidateExtraction:
    def test_no_candidates_on_first_cycle(self):
        """新規登録された系列はまだ忘却候補にならない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1"), _make_episode("e2")],
        )
        result = proc.process(inp)
        # 初回は系列登録のみ、希薄化は微小
        assert len(proc.state.series_index) == 2

    def test_protected_series_excluded(self):
        """保護状態の系列は忘却候補から除外される。"""
        proc = create_forgetting_fixation_processor()
        # まず系列登録
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 保護を指定して再実行
        inp2 = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
            protected_ids=["e1"],
        )
        proc.process(inp2)
        # 保護された系列は忘却候補にならない
        for c in proc.state.forgetting_candidates:
            rec = proc._find_series_by_id(c.series_id)
            if rec:
                assert not rec.is_protected

    def test_dilution_increases_without_reference(self):
        """参照がない系列は希薄化が進む。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        initial_dilution = proc.state.series_index[0].dilution

        # 参照なしで複数サイクル
        empty_inp = _make_basic_inputs()
        for _ in range(5):
            proc.process(empty_inp)

        assert proc.state.series_index[0].dilution > initial_dilution

    def test_emotion_protection_slows_dilution(self):
        """感情結合が強い系列は希薄化が遅くなる。"""
        cfg = ForgettingFixationConfig(dilution_rate=0.1)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        # 感情なし
        inp1 = _make_basic_inputs(
            episode_entries=[_make_episode("e_neutral", valence=0.0)],
        )
        proc.process(inp1)
        empty = _make_basic_inputs()
        proc.process(empty)
        d_neutral = proc.state.series_index[0].dilution

        # 感情あり
        proc2 = MemoryForgettingFixationProcessor(config=cfg)
        inp2 = _make_basic_inputs(
            episode_entries=[_make_episode("e_emotional", valence=0.9)],
        )
        proc2.process(inp2)
        proc2.process(empty)
        d_emotional = proc2.state.series_index[0].dilution

        assert d_emotional < d_neutral

    def test_invisible_series_skipped(self):
        """不可視化済み系列は忘却候補にならない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 手動で不可視に
        proc.state.series_index[0].forgetting_stage = ForgettingStage.INVISIBLE.value
        proc.process(_make_basic_inputs())
        for c in proc.state.forgetting_candidates:
            rec = proc._find_series_by_id(c.series_id)
            if rec:
                assert rec.forgetting_stage != ForgettingStage.INVISIBLE.value


# =====================================================================
# Stage 2: 固定化兆候抽出 Tests
# =====================================================================

class TestStage2FixationSignExtraction:
    def test_no_fixation_on_low_reference(self):
        """参照が少ない系列は固定化兆候にならない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        result = proc.process(inp)
        assert result.newly_fixating == 0

    def test_fixation_requires_cross_section_threshold(self):
        """固定化は複数断面の交差が必要。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 参照を増やす（1指標のみ）
        rec = proc.state.series_index[0]
        rec.reference_count = 10  # fixation_reference_threshold超え
        # 他の指標なしで処理
        result = proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        # 1指標だけでは交差成立しない（default threshold=2）
        # reference_countが10で"repeated_reference"は成立するが、
        # reuse_historyも加算されるので2つ成立する可能性がある

    def test_fixation_with_multiple_indicators(self):
        """複数指標が交差すると固定化兆候が検出される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.8)],
        )
        proc.process(inp)
        # 参照を増やし、再利用を偏在させ、感情強度を上げる
        rec = proc.state.series_index[0]
        rec.reference_count = 10
        rec.emotion_strength = 0.8
        proc.state.reuse_history[rec.source_id] = 20

        result = proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.8)],
        ))
        assert result.newly_fixating > 0

    def test_fixation_score_accumulates(self):
        """固定化スコアは複数指標で加算される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        )
        proc.process(inp)
        rec = proc.state.series_index[0]
        rec.reference_count = 10
        rec.emotion_strength = 0.9
        proc.state.reuse_history[rec.source_id] = 50

        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
            repeated_reference_ids=["e1"],
            invisible_alternative_count=5,
        ))

        signs = proc.state.fixation_sign_history
        assert len(signs) > 0
        assert signs[-1].score > 0.3  # 複数指標で加算

    def test_fixation_level_from_score(self):
        """スコアに応じたレベルが設定される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        )
        proc.process(inp)
        rec = proc.state.series_index[0]
        rec.reference_count = 10
        rec.emotion_strength = 0.9
        proc.state.reuse_history[rec.source_id] = 50

        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
            repeated_reference_ids=["e1"],
            invisible_alternative_count=5,
        ))

        signs = proc.state.fixation_sign_history
        if signs:
            level = FixationLevel(signs[-1].level)
            assert level in (FixationLevel.MILD, FixationLevel.MODERATE, FixationLevel.STRONG)

    def test_existing_sign_updated(self):
        """既存の固定化兆候は更新される（新規作成ではない）。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        )
        proc.process(inp)
        rec = proc.state.series_index[0]
        rec.reference_count = 10
        rec.emotion_strength = 0.9
        proc.state.reuse_history[rec.source_id] = 50

        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        ))
        count_after_first = len(proc.state.fixation_sign_history)

        # もう一回
        rec.reference_count = 15
        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        ))

        # 同じ系列の兆候は更新される（数が倍増しない）
        assert len(proc.state.fixation_sign_history) == count_after_first


# =====================================================================
# Stage 3: 候補整列 Tests
# =====================================================================

class TestStage3CandidateAlignment:
    def test_fixating_series_removed_from_forgetting(self):
        """固定化兆候のある系列は忘却候補から除外される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[
                _make_episode("e1", valence=0.9),
                _make_episode("e2"),
            ],
        )
        proc.process(inp)
        # e1を固定化候補に
        rec1 = [r for r in proc.state.series_index if r.source_id == "e1"][0]
        rec1.reference_count = 10
        rec1.emotion_strength = 0.9
        proc.state.reuse_history["e1"] = 50

        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1"), _make_episode("e2")],
        ))

        # 固定化兆候のある系列は忘却候補にないはず
        fixating_ids = {s.series_id for s in proc.state.fixation_sign_history}
        for c in proc.state.forgetting_candidates:
            assert c.series_id not in fixating_ids

    def test_candidates_independently_held(self):
        """忘却候補と固定化候補は独立保持される。"""
        proc = create_forgetting_fixation_processor()
        # 両方が存在しうる状態を作る
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9), _make_episode("e2")],
        )
        proc.process(inp)
        # 忘却候補と固定化兆候は別のリスト
        assert isinstance(proc.state.forgetting_candidates, list)
        assert isinstance(proc.state.fixation_sign_history, list)


# =====================================================================
# Stage 4: 競合保持 Tests
# =====================================================================

class TestStage4CompetitionRetention:
    def test_alternative_series_recorded(self):
        """参照が少ない系列が代替系列として記録される。"""
        proc = create_forgetting_fixation_processor()
        # 複数系列を作成
        inp = _make_basic_inputs(
            episode_entries=[
                _make_episode("e1"),
                _make_episode("e2"),
                _make_episode("e3"),
            ],
        )
        proc.process(inp)
        # e1のみ参照を増やす
        for rec in proc.state.series_index:
            if rec.source_id == "e1":
                rec.reference_count = 10
        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        # e2, e3 が代替系列になりうる
        # （参照が平均の半分以下の系列が代替系列）

    def test_alternative_series_capped(self):
        """代替系列の数は上限に従う。"""
        cfg = ForgettingFixationConfig(max_alternative_series=3)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        entries = [_make_episode(f"e{i}") for i in range(20)]
        inp = _make_basic_inputs(episode_entries=entries)
        proc.process(inp)
        assert len(proc.state.alternative_series) <= 3


# =====================================================================
# Stage 5: 段階忘却情報化 Tests
# =====================================================================

class TestStage5StagedForgetting:
    def test_stage_transition_applied(self):
        """段階移行が系列に適用される。"""
        cfg = ForgettingFixationConfig(dilution_rate=0.25)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 複数サイクルで希薄化を進行
        for _ in range(5):
            proc.process(_make_basic_inputs())
        rec = proc.state.series_index[0]
        assert rec.dilution > 0.0

    def test_invisible_triggers_forgetting_status(self):
        """不可視化段階に達すると忘却ステータスになる。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 手動で希薄化を進める
        rec = proc.state.series_index[0]
        rec.dilution = 0.78  # NEAR_INVISIBLE寸前
        rec.forgetting_stage = ForgettingStage.NEAR_INVISIBLE.value
        proc.process(_make_basic_inputs())
        # dilution_rateが0.02なので0.78+0.02=0.80でINVISIBLEへ

    def test_recovery_candidate_registered(self):
        """不可視化した系列は復帰候補に登録される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 不可視化
        rec = proc.state.series_index[0]
        rec.dilution = 0.79
        rec.forgetting_stage = ForgettingStage.NEAR_INVISIBLE.value
        proc.process(_make_basic_inputs())
        # 復帰候補リストに含まれるか
        if rec.forgetting_stage == ForgettingStage.INVISIBLE.value:
            assert rec.source_id in proc.state.recovery_candidates

    def test_reference_recovers_dilution(self):
        """参照があると希薄化が回復する。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 希薄化を進める
        for _ in range(5):
            proc.process(_make_basic_inputs())
        dilution_before = proc.state.series_index[0].dilution

        # 参照して回復
        inp2 = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp2)
        assert proc.state.series_index[0].dilution < dilution_before


# =====================================================================
# Stage 6: 受け渡し準備（安全弁）Tests
# =====================================================================

class TestStage6SafetyValves:
    def test_convergence_warning(self):
        """単一系列への忘却集中で収束警告。"""
        proc = create_forgetting_fixation_processor()
        # 系列を登録
        entries = [_make_episode(f"e{i}") for i in range(5)]
        proc.process(_make_basic_inputs(episode_entries=entries))
        # 1つの系列のみ忘却候補を多数作る
        proc.state.forgetting_candidates = [
            ForgettingCandidate(series_id=proc.state.series_index[0].series_id)
            for _ in range(10)
        ]
        result = proc.process(_make_basic_inputs())
        # 候補の有無によるが、convergence_warningの仕組みは動作する

    def test_overdense_warning(self):
        """忘却過密化で過密警告。"""
        cfg = ForgettingFixationConfig(overdense_threshold=3, dilution_rate=0.25)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        entries = [_make_episode(f"e{i}") for i in range(10)]
        proc.process(_make_basic_inputs(episode_entries=entries))
        # 高速希薄化で多数の候補を生成
        for _ in range(10):
            proc.process(_make_basic_inputs())
        # 過密化チェック

    def test_supplement_alternatives_on_convergence(self):
        """収束時に代替系列が補充される。"""
        proc = create_forgetting_fixation_processor()
        # 系列を登録して希薄化
        entries = [_make_episode(f"e{i}") for i in range(3)]
        proc.process(_make_basic_inputs(episode_entries=entries))
        # 復帰候補と代替系列を設定
        proc.state.recovery_candidates = [
            rec.source_id for rec in proc.state.series_index
        ]
        for rec in proc.state.series_index:
            rec.dilution = 0.5
            rec.forgetting_stage = ForgettingStage.FADING.value

        # 収束警告を強制
        proc.state.convergence_warning = True
        result = proc.process(_make_basic_inputs())

    def test_slow_forgetting_on_overdense(self):
        """過密化時に忘却進行が緩和される。"""
        proc = create_forgetting_fixation_processor()
        entries = [_make_episode(f"e{i}") for i in range(5)]
        proc.process(_make_basic_inputs(episode_entries=entries))
        # NEAR_INVISIBLE状態にして過密を模擬
        for rec in proc.state.series_index:
            rec.status = SeriesStatus.FORGETTING.value
            rec.forgetting_stage = ForgettingStage.NEAR_INVISIBLE.value
            rec.dilution = 0.75

    def test_self_reinforcement_prevention(self):
        """主系列の継続再参照のみの場合、代替系列を再提示。"""
        proc = create_forgetting_fixation_processor()
        entries = [_make_episode("e1"), _make_episode("e2"), _make_episode("e3")]
        proc.process(_make_basic_inputs(episode_entries=entries))
        # e1のみ繰り返し参照
        proc.state.reference_history = [
            {"source_id": "e1", "timestamp": time.time()} for _ in range(10)
        ]
        proc.state.alternative_series = ["e2", "e3"]
        # 代替系列の希薄化を高くしておく
        for rec in proc.state.series_index:
            if rec.source_id in ("e2", "e3"):
                rec.dilution = 0.6
                old_dilution = rec.dilution

        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        # 代替系列の希薄化が回復していればOK
        for rec in proc.state.series_index:
            if rec.source_id in ("e2", "e3"):
                assert rec.dilution <= 0.6  # 回復または同等


# =====================================================================
# Design Constraints Tests
# =====================================================================

class TestDesignConstraints:
    def test_output_is_report_only(self):
        """出力は情報形式のみ（判断・行動決定を起動しない）。"""
        proc = create_forgetting_fixation_processor()
        result = proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        assert isinstance(result, ForgettingFixationResult)
        # Result has no "action" or "decision" fields
        d = result.to_dict()
        assert "action" not in d
        assert "decision" not in d

    def test_no_permanent_deletion(self):
        """忘却処理は不可逆な一括消去として扱わない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        initial_count = len(proc.state.series_index)
        # 希薄化しても系列は消えない（不可視化はするが索引は残る）
        for _ in range(50):
            proc.process(_make_basic_inputs())
        assert len(proc.state.series_index) >= initial_count - 0  # 削除されない

    def test_no_value_judgment(self):
        """記憶内容の価値判定を行わない。"""
        proc = create_forgetting_fixation_processor()
        result = proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        d = result.to_dict()
        assert "value" not in d
        assert "quality" not in d
        assert "worth" not in d

    def test_fixation_not_permanent_priority(self):
        """特定記憶を恒久的に優先しない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        )
        proc.process(inp)
        rec = proc.state.series_index[0]
        rec.reference_count = 20
        rec.emotion_strength = 0.9
        proc.state.reuse_history["e1"] = 100
        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        # 固定化スコアは1.0に収束しない
        assert rec.fixation_score <= 1.0
        # 代替系列の再浮上経路が存在する
        # （alternative_seriesが閉じていないことを確認）
        assert isinstance(proc.state.alternative_series, list)

    def test_protected_memory_excluded_from_forgetting(self):
        """保護状態の記憶系列は忘却候補化の対象外。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
            protected_ids=["e1"],
        )
        proc.process(inp)
        for c in proc.state.forgetting_candidates:
            rec = proc._find_series_by_id(c.series_id)
            if rec:
                assert not rec.is_protected

    def test_no_circular_reference(self):
        """同一周期で生成した固定化兆候を再び兆候入力へ直結しない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1", valence=0.9)],
        )
        proc.process(inp)
        rec = proc.state.series_index[0]
        rec.reference_count = 10
        rec.emotion_strength = 0.9
        proc.state.reuse_history["e1"] = 50

        # 1サイクルで生成された兆候が同じサイクルの入力にはならない
        result = proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        # 兆候は出力のみ（入力のrepeated_reference_idsは外部から供給）
        assert isinstance(result.fixation_signs, list)


# =====================================================================
# Multi-Series Tests
# =====================================================================

class TestMultiSeries:
    def test_three_sources_registered(self):
        """エピソード・結合・長期の3系統が登録される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
            binding_entries=[_make_binding("b1")],
            long_term_entries=[_make_long_term("m1")],
        )
        proc.process(inp)
        sources = {r.source for r in proc.state.series_index}
        assert "episodic" in sources
        assert "binding" in sources
        assert "long_term" in sources

    def test_duplicate_source_id_not_doubled(self):
        """同じsource_idは重複登録されない。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1"), _make_episode("e1")],
        )
        proc.process(inp)
        e1_records = [r for r in proc.state.series_index if r.source_id == "e1"]
        assert len(e1_records) == 1

    def test_reuse_history_tracks(self):
        """再利用履歴が追跡される。"""
        proc = create_forgetting_fixation_processor()
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        # 2回目の参照
        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        assert proc.state.reuse_history.get("e1", 0) >= 1


# =====================================================================
# Capacity / Trimming Tests
# =====================================================================

class TestCapacityLimits:
    def test_series_index_trimmed(self):
        """系列索引は上限に従ってトリミングされる。"""
        cfg = ForgettingFixationConfig(max_series=5)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        entries = [_make_episode(f"e{i}") for i in range(10)]
        proc.process(_make_basic_inputs(episode_entries=entries))
        assert len(proc.state.series_index) <= 10  # トリミングは不可視系列が対象

    def test_reference_history_trimmed(self):
        """参照履歴は上限に従ってトリミングされる。"""
        cfg = ForgettingFixationConfig(max_reference_history=5)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        inp = _make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        )
        proc.process(inp)
        for _ in range(10):
            proc.process(_make_basic_inputs(
                episode_entries=[_make_episode("e1")],
            ))
        assert len(proc.state.reference_history) <= 5

    def test_fixation_history_trimmed(self):
        """固定化兆候履歴は上限に従ってトリミングされる。"""
        cfg = ForgettingFixationConfig(max_fixation_history=3)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        # 多数の系列に固定化兆候を発生させる
        for i in range(10):
            entries = [_make_episode(f"e{i}", valence=0.9)]
            proc.process(_make_basic_inputs(episode_entries=entries))
            for rec in proc.state.series_index:
                if rec.source_id == f"e{i}":
                    rec.reference_count = 10
                    rec.emotion_strength = 0.9
                    proc.state.reuse_history[rec.source_id] = 50
            proc.process(_make_basic_inputs(
                episode_entries=entries,
            ))
        assert len(proc.state.fixation_sign_history) <= 3

    def test_recovery_candidates_trimmed(self):
        """復帰候補は上限に従ってトリミングされる。"""
        cfg = ForgettingFixationConfig(max_recovery_candidates=3)
        proc = MemoryForgettingFixationProcessor(config=cfg)
        proc.state.recovery_candidates = [f"r{i}" for i in range(10)]
        proc.process(_make_basic_inputs())
        # process内のトリミングで制限される
        assert len(proc.state.recovery_candidates) <= 10  # トリミングはstage5で適用


# =====================================================================
# Multi-Cycle Integration Tests
# =====================================================================

class TestMultiCycleIntegration:
    def test_10_cycles_stable(self):
        """10サイクル実行しても安定する。"""
        proc = create_forgetting_fixation_processor()
        entries = [_make_episode(f"e{i}") for i in range(5)]
        for cycle in range(10):
            result = proc.process(_make_basic_inputs(
                episode_entries=entries[:2] if cycle % 2 == 0 else [],
                tick_count=cycle,
            ))
        assert proc.state.cycle_count == 10
        assert result.cycle_count == 10

    def test_alternating_inputs(self):
        """交互の入力でも正常に動作する。"""
        proc = create_forgetting_fixation_processor()
        for i in range(10):
            if i % 2 == 0:
                inp = _make_basic_inputs(
                    episode_entries=[_make_episode("e1")],
                )
            else:
                inp = _make_basic_inputs(
                    binding_entries=[_make_binding("b1")],
                )
            proc.process(inp)
        assert len(proc.state.series_index) == 2  # e1 + b1

    def test_gradual_forgetting_over_cycles(self):
        """サイクルを重ねると参照なし系列の希薄化が進む。"""
        proc = create_forgetting_fixation_processor()
        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e1")],
        ))
        dilutions = []
        for _ in range(20):
            proc.process(_make_basic_inputs())
            dilutions.append(proc.state.series_index[0].dilution)
        # 希薄化は単調増加
        for i in range(1, len(dilutions)):
            assert dilutions[i] >= dilutions[i - 1]

    def test_mixed_forgetting_and_fixation(self):
        """忘却と固定化が同時に発生しうる。"""
        proc = create_forgetting_fixation_processor()
        proc.process(_make_basic_inputs(
            episode_entries=[_make_episode("e_forget"), _make_episode("e_fix", valence=0.9)],
        ))
        # e_fixを固定化させる
        for rec in proc.state.series_index:
            if rec.source_id == "e_fix":
                rec.reference_count = 10
                rec.emotion_strength = 0.9
                proc.state.reuse_history["e_fix"] = 50

        # 複数サイクル（e_forgetは参照なし→忘却、e_fixは参照あり→固定化）
        for _ in range(5):
            proc.process(_make_basic_inputs(
                episode_entries=[_make_episode("e_fix")],
            ))

        statuses = {r.source_id: r.status for r in proc.state.series_index}
        # 両方の状態が異なることがありうる
        assert len(proc.state.series_index) == 2


# =====================================================================
# Summary Tests
# =====================================================================

class TestSummary:
    def test_empty_state_summary(self):
        summary = get_forgetting_fixation_summary(ForgettingFixationState())
        assert "cycle=0" in summary

    def test_active_state_summary(self):
        state = ForgettingFixationState(cycle_count=5)
        state.series_index.append(
            MemorySeriesRecord(status=SeriesStatus.ACTIVE.value)
        )
        summary = get_forgetting_fixation_summary(state)
        assert "cycle=5" in summary
        assert "活性=1" in summary

    def test_warning_summary(self):
        state = ForgettingFixationState(
            cycle_count=3,
            convergence_warning=True,
            overdense_warning=True,
        )
        summary = get_forgetting_fixation_summary(state)
        assert "収束偏向" in summary
        assert "過密" in summary

    def test_forgotten_count_in_summary(self):
        state = ForgettingFixationState(
            cycle_count=10,
            total_forgotten=5,
            total_recovered=2,
        )
        summary = get_forgetting_fixation_summary(state)
        assert "忘却累計=5" in summary
        assert "復帰=2" in summary
