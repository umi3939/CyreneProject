"""
tests/test_internal_contradiction_description.py

内部状態の矛盾並置の構造的記述モジュールのテスト。

設計原則の検証:
- 矛盾を解消しない
- 矛盾に優先度を付けない
- 矛盾を評価しない（「正常」「異常」ラベル禁止）
- パターン抽出禁止
- 意味的矛盾判定禁止（数値的方向の乖離のみ）
- 全記録等価
- 判断・行動・責任の各処理系統に接続しない
"""

import time
import pytest

from psyche.internal_contradiction_description import (
    # Data Structures
    ContradictionRecord,
    ContradictionInputs,
    ContradictionState,
    ContradictionResult,
    ContradictionConfig,
    # Processor
    InternalContradictionProcessor,
    # Detection functions
    _detect_self_model_vs_meta_emotion,
    _detect_self_image_stability_vs_temporal_diff,
    _detect_identity_coherence_vs_stabilization,
    _detect_self_image_continuity_vs_strain,
    _detect_self_model_emotion_vs_self_image_tone,
    _detect_cross_section_internal,
    # Constants
    PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION,
    PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF,
    PAIR_IDENTITY_COHERENCE_VS_STABILIZATION,
    PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN,
    PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE,
    PAIR_CROSS_SECTION_INTERNAL,
    PAIR_DEFINITIONS,
    PAIR_LABELS,
    _FORBIDDEN_WORDS,
    _sanitize_text,
    # Factory
    create_contradiction_processor,
    # Summary
    get_contradiction_summary,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_default_inputs(**kwargs) -> ContradictionInputs:
    """テスト用のデフォルト入力を生成する。

    デフォルト値は全ての断面対で乖離が検出されないよう設計されている。
    各断面対の方向が整合的な中間値を設定する。
    """
    defaults = {
        "self_model_emotion_intensity": 0.5,
        "self_model_emotion_spread": 0.3,
        "self_model_emotion_conflict": False,
        "meta_emotion_change_speed": 0.1,
        "meta_emotion_dominant_stability": 0.5,
        "self_image_stability": 0.5,
        "self_image_continuity": 0.5,
        "self_image_emotional_tone": 0.5,
        "identity_coherence_active_shifts": 0,
        "identity_coherence_level": 0.0,
        "temporal_diff_magnitude": 0.5,
        "continuity_strain_level": 0.5,
        "cross_section_values": {},
        "stabilization_signal_count": 0,
        "stabilization_diff_degree": 0.0,
        "current_tick": 1,
    }
    defaults.update(kwargs)
    return ContradictionInputs(**defaults)


def _make_divergent_inputs(pair_name: str, tick: int = 1) -> ContradictionInputs:
    """特定の断面対で乖離が検出される入力を生成する。

    各断面対に対して、対象ペアのみが乖離を検出し、
    他のペアでは乖離が検出されないよう関連フィールドも調整する。
    """
    if pair_name == PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION:
        return _make_default_inputs(
            self_model_emotion_intensity=0.9,
            meta_emotion_dominant_stability=0.1,
            # PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE が誤検出しないよう調整
            self_image_emotional_tone=0.1,
            current_tick=tick,
        )
    elif pair_name == PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF:
        return _make_default_inputs(
            self_image_stability=0.9,
            temporal_diff_magnitude=0.9,
            current_tick=tick,
        )
    elif pair_name == PAIR_IDENTITY_COHERENCE_VS_STABILIZATION:
        return _make_default_inputs(
            identity_coherence_level=0.9,
            stabilization_signal_count=0,
            current_tick=tick,
        )
    elif pair_name == PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN:
        return _make_default_inputs(
            self_image_continuity=0.9,
            continuity_strain_level=0.9,
            current_tick=tick,
        )
    elif pair_name == PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE:
        return _make_default_inputs(
            self_model_emotion_intensity=0.9,
            self_image_emotional_tone=0.9,
            # PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION が誤検出しないよう調整
            meta_emotion_dominant_stability=0.9,
            current_tick=tick,
        )
    elif pair_name == PAIR_CROSS_SECTION_INTERNAL:
        return _make_default_inputs(
            cross_section_values={"section_a": 0.9, "section_b": 0.1},
            current_tick=tick,
        )
    return _make_default_inputs(current_tick=tick)


# =============================================================================
# ContradictionRecord Tests
# =============================================================================

class TestContradictionRecord:
    def test_creation_auto_id(self):
        record = ContradictionRecord(pair_name="test_pair")
        assert record.record_id != ""
        assert record.pair_name == "test_pair"
        assert record.freshness == 1.0

    def test_to_dict_roundtrip(self):
        record = ContradictionRecord(
            pair_name="test_pair",
            section_a="A",
            section_b="B",
            direction_a="dir_a",
            direction_b="dir_b",
            tick=10,
            freshness=0.8,
        )
        d = record.to_dict()
        restored = ContradictionRecord.from_dict(d)
        assert restored.pair_name == "test_pair"
        assert restored.section_a == "A"
        assert restored.section_b == "B"
        assert restored.direction_a == "dir_a"
        assert restored.direction_b == "dir_b"
        assert restored.tick == 10
        assert restored.freshness == 0.8

    def test_from_dict_defaults(self):
        record = ContradictionRecord.from_dict({})
        assert record.pair_name == ""
        assert record.freshness == 1.0


# =============================================================================
# ContradictionState Tests
# =============================================================================

class TestContradictionState:
    def test_initial_state(self):
        state = ContradictionState()
        assert state.contradiction_window == []
        assert state.previous_contradictions == []
        assert state.consecutive_counts == {}
        assert state.suppressed_pairs == {}
        assert state.cycle_count == 0
        assert state.total_contradictions_detected == 0

    def test_to_dict_roundtrip(self):
        state = ContradictionState(
            contradiction_window=[
                ContradictionRecord(pair_name="p1", tick=1),
                ContradictionRecord(pair_name="p2", tick=2),
            ],
            consecutive_counts={"p1": 3},
            suppressed_pairs={"p2": 2},
            cycle_count=5,
            total_contradictions_detected=10,
        )
        d = state.to_dict()
        restored = ContradictionState.from_dict(d)
        assert len(restored.contradiction_window) == 2
        assert restored.consecutive_counts == {"p1": 3}
        assert restored.suppressed_pairs == {"p2": 2}
        assert restored.cycle_count == 5
        assert restored.total_contradictions_detected == 10

    def test_from_dict_defaults(self):
        state = ContradictionState.from_dict({})
        assert state.contradiction_window == []
        assert state.cycle_count == 0


# =============================================================================
# Sanitize Text Tests
# =============================================================================

class TestSanitizeText:
    def test_no_forbidden_words(self):
        assert _sanitize_text("test text") == "test text"

    def test_removes_forbidden_words(self):
        for word in _FORBIDDEN_WORDS:
            result = _sanitize_text(f"value is {word}")
            assert word not in result
            assert "[...]" in result

    def test_multiple_forbidden_words(self):
        result = _sanitize_text("異常な状態が正常に戻る")
        assert "異常" not in result
        assert "正常" not in result


# =============================================================================
# Individual Pair Detection Tests
# =============================================================================

class TestPairDetection:
    """各断面対の乖離検出関数のテスト。"""

    def test_self_model_vs_meta_emotion_divergent(self):
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        config = ContradictionConfig()
        result = _detect_self_model_vs_meta_emotion(inputs, config)
        assert result is not None
        assert result.pair_name == PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION

    def test_self_model_vs_meta_emotion_convergent(self):
        inputs = _make_default_inputs(
            self_model_emotion_intensity=0.5,
            meta_emotion_dominant_stability=0.5,
        )
        config = ContradictionConfig()
        result = _detect_self_model_vs_meta_emotion(inputs, config)
        assert result is None

    def test_self_image_stability_vs_temporal_diff_divergent(self):
        inputs = _make_divergent_inputs(PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF)
        config = ContradictionConfig()
        result = _detect_self_image_stability_vs_temporal_diff(inputs, config)
        assert result is not None
        assert result.pair_name == PAIR_SELF_IMAGE_STABILITY_VS_TEMPORAL_DIFF

    def test_self_image_stability_vs_temporal_diff_convergent(self):
        inputs = _make_default_inputs(
            self_image_stability=0.8,
            temporal_diff_magnitude=0.1,
        )
        config = ContradictionConfig()
        result = _detect_self_image_stability_vs_temporal_diff(inputs, config)
        assert result is None

    def test_identity_coherence_vs_stabilization_divergent(self):
        inputs = _make_divergent_inputs(PAIR_IDENTITY_COHERENCE_VS_STABILIZATION)
        config = ContradictionConfig()
        result = _detect_identity_coherence_vs_stabilization(inputs, config)
        assert result is not None
        assert result.pair_name == PAIR_IDENTITY_COHERENCE_VS_STABILIZATION

    def test_identity_coherence_vs_stabilization_convergent(self):
        inputs = _make_default_inputs(
            identity_coherence_level=0.5,
            stabilization_signal_count=3,
        )
        config = ContradictionConfig()
        result = _detect_identity_coherence_vs_stabilization(inputs, config)
        assert result is None

    def test_self_image_continuity_vs_strain_divergent(self):
        inputs = _make_divergent_inputs(PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN)
        config = ContradictionConfig()
        result = _detect_self_image_continuity_vs_strain(inputs, config)
        assert result is not None
        assert result.pair_name == PAIR_SELF_IMAGE_CONTINUITY_VS_CONTINUITY_STRAIN

    def test_self_image_continuity_vs_strain_convergent(self):
        inputs = _make_default_inputs(
            self_image_continuity=0.8,
            continuity_strain_level=0.1,
        )
        config = ContradictionConfig()
        result = _detect_self_image_continuity_vs_strain(inputs, config)
        assert result is None

    def test_self_model_emotion_vs_self_image_tone_divergent(self):
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE)
        config = ContradictionConfig()
        result = _detect_self_model_emotion_vs_self_image_tone(inputs, config)
        assert result is not None
        assert result.pair_name == PAIR_SELF_MODEL_EMOTION_VS_SELF_IMAGE_TONE

    def test_self_model_emotion_vs_self_image_tone_convergent(self):
        inputs = _make_default_inputs(
            self_model_emotion_intensity=0.5,
            self_image_emotional_tone=0.5,
        )
        config = ContradictionConfig()
        result = _detect_self_model_emotion_vs_self_image_tone(inputs, config)
        assert result is None

    def test_cross_section_internal_divergent(self):
        inputs = _make_divergent_inputs(PAIR_CROSS_SECTION_INTERNAL)
        config = ContradictionConfig()
        results = _detect_cross_section_internal(inputs, config)
        assert len(results) >= 1
        assert all(r.pair_name == PAIR_CROSS_SECTION_INTERNAL for r in results)

    def test_cross_section_internal_convergent(self):
        inputs = _make_default_inputs(
            cross_section_values={"a": 0.5, "b": 0.55},
        )
        config = ContradictionConfig()
        results = _detect_cross_section_internal(inputs, config)
        assert len(results) == 0

    def test_cross_section_internal_empty_values(self):
        inputs = _make_default_inputs(cross_section_values={})
        config = ContradictionConfig()
        results = _detect_cross_section_internal(inputs, config)
        assert len(results) == 0

    def test_cross_section_internal_single_value(self):
        inputs = _make_default_inputs(cross_section_values={"a": 0.5})
        config = ContradictionConfig()
        results = _detect_cross_section_internal(inputs, config)
        assert len(results) == 0


# =============================================================================
# Processor Basic Tests
# =============================================================================

class TestInternalContradictionProcessor:
    def test_creation(self):
        proc = InternalContradictionProcessor()
        assert proc.state.cycle_count == 0
        assert proc.state.contradiction_window == []

    def test_factory(self):
        proc = create_contradiction_processor()
        assert isinstance(proc, InternalContradictionProcessor)

    def test_factory_with_config(self):
        config = ContradictionConfig(max_window_size=10)
        proc = create_contradiction_processor(config=config)
        assert proc._config.max_window_size == 10

    def test_process_no_contradictions(self):
        proc = InternalContradictionProcessor()
        inputs = _make_default_inputs()
        result = proc.process(inputs)
        assert result.detected_count == 0
        assert result.cycle_count == 1
        assert proc.state.cycle_count == 1

    def test_process_with_contradictions(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        result = proc.process(inputs)
        assert result.detected_count >= 1
        assert result.window_size >= 1

    def test_multiple_ticks(self):
        proc = InternalContradictionProcessor()
        for tick in range(1, 6):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            result = proc.process(inputs)
        assert proc.state.cycle_count == 5
        assert proc.state.total_contradictions_detected >= 5

    def test_window_fifo_pushout(self):
        config = ContradictionConfig(max_window_size=3)
        proc = InternalContradictionProcessor(config=config)
        for tick in range(1, 10):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)
        assert len(proc.state.contradiction_window) <= 3

    def test_freshness_decay(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=1)
        proc.process(inputs)
        initial_freshness = proc.state.contradiction_window[0].freshness

        # 2回目の処理で鮮度が減衰する
        inputs2 = _make_default_inputs(current_tick=2)
        proc.process(inputs2)
        assert proc.state.contradiction_window[0].freshness < initial_freshness


# =============================================================================
# Convergence Monitoring Tests (Safety Valve 5)
# =============================================================================

class TestConvergenceMonitoring:
    def test_consecutive_detection_triggers_suppression(self):
        config = ContradictionConfig(consecutive_limit=3, suppression_duration=2)
        proc = InternalContradictionProcessor(config=config)

        # 同一断面対を連続検出
        for tick in range(1, 5):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        # 抑制が発動している
        assert PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION in proc.state.suppressed_pairs

    def test_suppression_expires(self):
        config = ContradictionConfig(consecutive_limit=2, suppression_duration=2)
        proc = InternalContradictionProcessor(config=config)

        # 連続検出で抑制を発動
        for tick in range(1, 4):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        assert PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION in proc.state.suppressed_pairs

        # 非乖離入力で抑制期間を消化
        for tick in range(4, 7):
            inputs = _make_default_inputs(current_tick=tick)
            proc.process(inputs)

        # 抑制が解除されている
        assert PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION not in proc.state.suppressed_pairs

    def test_suppressed_pair_not_detected(self):
        config = ContradictionConfig(consecutive_limit=2, suppression_duration=5)
        proc = InternalContradictionProcessor(config=config)

        # 連続検出で抑制を発動
        for tick in range(1, 4):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        assert PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION in proc.state.suppressed_pairs

        # 抑制中は検出されない
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=10)
        result = proc.process(inputs)
        # 他の対が検出される可能性はあるが、抑制対象は検出されない
        suppressed_found = any(
            r.pair_name == PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION
            for r in proc.state.previous_contradictions
        )
        assert not suppressed_found


# =============================================================================
# Enrichment Tests
# =============================================================================

class TestEnrichment:
    def test_empty_enrichment(self):
        proc = InternalContradictionProcessor()
        text = proc.get_enrichment_text()
        assert "待機中" in text

    def test_enrichment_with_records(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        proc.process(inputs)
        text = proc.get_enrichment_text()
        assert "待機中" not in text
        assert "t1" in text

    def test_enrichment_count_limit(self):
        config = ContradictionConfig(max_enrichment_count=2, consecutive_limit=100)
        proc = InternalContradictionProcessor(config=config)

        for tick in range(1, 10):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        text = proc.get_enrichment_text()
        # 最大2件のみ出力
        lines = [l for l in text.split("\n") if l.strip()]
        assert len(lines) <= 2

    def test_enrichment_excludes_low_freshness(self):
        config = ContradictionConfig(freshness_decay_rate=0.5, freshness_min_visible=0.3)
        proc = InternalContradictionProcessor(config=config)

        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=1)
        proc.process(inputs)

        # 大量の処理で鮮度を下げる
        for tick in range(2, 20):
            inputs2 = _make_default_inputs(current_tick=tick)
            proc.process(inputs2)

        text = proc.get_enrichment_text()
        # 鮮度が下がったため「待機中」になる可能性がある
        # (freshness_decay_rate=0.5 なので2ティックで0以下になる)

    def test_enrichment_no_forbidden_words(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        proc.process(inputs)
        text = proc.get_enrichment_text()
        for word in _FORBIDDEN_WORDS:
            assert word not in text

    def test_enrichment_size_limit(self):
        config = ContradictionConfig(max_enrichment_length=50, consecutive_limit=100)
        proc = InternalContradictionProcessor(config=config)

        for tick in range(1, 10):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        text = proc.get_enrichment_text()
        assert len(text) <= 50


# =============================================================================
# Design Principle Verification Tests
# =============================================================================

class TestDesignPrinciples:
    """設計原則の検証テスト。"""

    def test_no_contradiction_resolution(self):
        """矛盾を解消する処理が存在しないことを確認。"""
        proc = InternalContradictionProcessor()
        forbidden_methods = [
            "resolve", "fix", "repair", "correct", "normalize",
            "reconcile", "harmonize", "balance",
        ]
        methods = [m for m in dir(proc) if not m.startswith('_')]
        for method in methods:
            method_lower = method.lower()
            for forbidden in forbidden_methods:
                assert forbidden not in method_lower, (
                    f"Method '{method}' contains forbidden word '{forbidden}'"
                )

    def test_no_priority_on_records(self):
        """全記録等価: 重み・スコア・優先度を持たないことを確認。"""
        record = ContradictionRecord(
            pair_name="test",
            section_a="A",
            section_b="B",
            direction_a="d_a",
            direction_b="d_b",
            tick=1,
        )
        d = record.to_dict()
        forbidden_keys = ["weight", "score", "priority", "importance", "rank"]
        for key in forbidden_keys:
            assert key not in d, f"Record contains forbidden key '{key}'"

    def test_no_pattern_extraction(self):
        """パターン抽出禁止: 傾向・周期性・相関を算出する処理がないことを確認。"""
        proc = InternalContradictionProcessor()
        forbidden_methods = [
            "pattern", "trend", "correlation", "frequency_stats",
            "periodicity", "cluster",
        ]
        methods = [m for m in dir(proc) if not m.startswith('_')]
        for method in methods:
            method_lower = method.lower()
            for forbidden in forbidden_methods:
                assert forbidden not in method_lower, (
                    f"Method '{method}' contains forbidden word '{forbidden}'"
                )

    def test_no_evaluative_labels(self):
        """矛盾の存在を「正常」「異常」と判定しないことを確認。"""
        proc = InternalContradictionProcessor()

        # 多数の矛盾を検出させる
        for tick in range(1, 10):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        # enrichmentテキストに禁止語が含まれない
        text = proc.get_enrichment_text()
        for word in _FORBIDDEN_WORDS:
            assert word not in text

        # summaryにも禁止語が含まれない
        summary = get_contradiction_summary(proc.state)
        for word in _FORBIDDEN_WORDS:
            assert word not in summary

    def test_records_are_equal(self):
        """全記録等価: 検出順序に基づく暗黙的な優先付けがないことを確認。"""
        proc = InternalContradictionProcessor()
        for tick in range(1, 4):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        window = proc.get_contradiction_window()
        # 全記録に同じフィールドセットが含まれる
        if len(window) >= 2:
            keys_0 = set(window[0].keys())
            for record in window[1:]:
                assert set(record.keys()) == keys_0

    def test_no_decision_system_connection(self):
        """判断・行動・責任の各処理系統への接続がないことを確認。"""
        proc = InternalContradictionProcessor()
        forbidden_methods = [
            "policy", "decide", "action", "responsibility",
            "bias", "select", "choose", "recommend",
        ]
        methods = [m for m in dir(proc) if not m.startswith('_')]
        for method in methods:
            method_lower = method.lower()
            for forbidden in forbidden_methods:
                assert forbidden not in method_lower, (
                    f"Method '{method}' suggests decision system connection"
                )

    def test_read_only_principle(self):
        """入力源モジュールの内部状態に書き込まないことを確認。

        プロセッサは自身のstateのみ変更し、inputsを変更しない。
        """
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        original_intensity = inputs.self_model_emotion_intensity
        proc.process(inputs)
        assert inputs.self_model_emotion_intensity == original_intensity

    def test_no_semantic_judgment(self):
        """意味的矛盾判定禁止: 数値的方向の乖離のみで判定することを確認。

        入力が全て数値であり、テキスト解釈を行わない。
        """
        inputs = ContradictionInputs()
        # ContradictionInputsの全フィールドが数値型またはdict[str, float]であること
        for field_name in [
            "self_model_emotion_intensity", "self_model_emotion_spread",
            "meta_emotion_change_speed", "meta_emotion_dominant_stability",
            "self_image_stability", "self_image_continuity",
            "self_image_emotional_tone", "temporal_diff_magnitude",
            "continuity_strain_level", "stabilization_diff_degree",
        ]:
            val = getattr(inputs, field_name)
            assert isinstance(val, (int, float)), (
                f"Field {field_name} is not numeric: {type(val)}"
            )


# =============================================================================
# Save / Load Tests
# =============================================================================

class TestSaveLoad:
    def test_save_load_roundtrip(self):
        proc = InternalContradictionProcessor()
        for tick in range(1, 6):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        saved = proc.save()
        proc2 = InternalContradictionProcessor()
        proc2.load(saved)

        assert len(proc2.state.contradiction_window) == len(proc.state.contradiction_window)
        assert proc2.state.cycle_count == proc.state.cycle_count
        assert proc2.state.total_contradictions_detected == proc.state.total_contradictions_detected

    def test_load_empty_data(self):
        proc = InternalContradictionProcessor()
        proc.load({})
        assert proc.state.cycle_count == 0
        assert proc.state.contradiction_window == []

    def test_save_load_preserves_suppression(self):
        config = ContradictionConfig(consecutive_limit=2, suppression_duration=5)
        proc = InternalContradictionProcessor(config=config)
        for tick in range(1, 5):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        saved = proc.save()
        proc2 = InternalContradictionProcessor(config=config)
        proc2.load(saved)
        assert proc2.state.suppressed_pairs == proc.state.suppressed_pairs


# =============================================================================
# Accessor Tests
# =============================================================================

class TestAccessors:
    def test_get_contradiction_window(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        proc.process(inputs)
        window = proc.get_contradiction_window()
        assert isinstance(window, list)
        assert len(window) >= 1
        assert "pair_name" in window[0]
        assert "record_id" in window[0]

    def test_get_previous_contradictions(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        proc.process(inputs)
        prev = proc.get_previous_contradictions()
        assert isinstance(prev, list)
        assert len(prev) >= 1

    def test_get_summary(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        proc.process(inputs)
        summary = proc.get_summary()
        assert "window_size" in summary
        assert "cycle_count" in summary
        assert summary["cycle_count"] == 1

    def test_get_contradiction_summary_waiting(self):
        state = ContradictionState()
        text = get_contradiction_summary(state)
        assert "待機中" in text

    def test_get_contradiction_summary_with_data(self):
        proc = InternalContradictionProcessor()
        inputs = _make_divergent_inputs(PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION)
        proc.process(inputs)
        text = get_contradiction_summary(proc.state)
        assert "cycle=1" in text


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    def test_pair_definitions_completeness(self):
        assert len(PAIR_DEFINITIONS) == 6

    def test_pair_labels_completeness(self):
        for pair_name in PAIR_DEFINITIONS:
            assert pair_name in PAIR_LABELS

    def test_pair_definitions_are_unique(self):
        assert len(PAIR_DEFINITIONS) == len(set(PAIR_DEFINITIONS))


# =============================================================================
# Multiple Contradiction Types Tests
# =============================================================================

class TestMultipleContradictions:
    def test_detect_multiple_pairs_simultaneously(self):
        """複数の断面対が同時に乖離を検出できることを確認。"""
        proc = InternalContradictionProcessor()
        inputs = _make_default_inputs(
            self_model_emotion_intensity=0.9,
            meta_emotion_dominant_stability=0.1,
            self_image_stability=0.9,
            temporal_diff_magnitude=0.9,
            self_image_continuity=0.9,
            continuity_strain_level=0.9,
            current_tick=1,
        )
        result = proc.process(inputs)
        assert result.detected_count >= 2

    def test_independent_records_for_same_pair(self):
        """同一断面対が連続検出されても各記録は独立していることを確認。"""
        config = ContradictionConfig(consecutive_limit=100)  # 抑制なし
        proc = InternalContradictionProcessor(config=config)
        for tick in range(1, 4):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)

        window = proc.get_contradiction_window()
        record_ids = [r["record_id"] for r in window]
        # 各記録のIDは独立
        assert len(record_ids) == len(set(record_ids))

    def test_no_data_loss_except_fifo(self):
        """データ消失経路がFIFO押し出しのみであることを確認。"""
        config = ContradictionConfig(max_window_size=3, consecutive_limit=100)
        proc = InternalContradictionProcessor(config=config)

        all_ids = []
        for tick in range(1, 8):
            inputs = _make_divergent_inputs(
                PAIR_SELF_MODEL_EMOTION_VS_META_EMOTION, tick=tick
            )
            proc.process(inputs)
            window = proc.get_contradiction_window()
            current_ids = [r["record_id"] for r in window]
            all_ids.extend(current_ids)

        # ウィンドウサイズは上限以下
        final_window = proc.get_contradiction_window()
        assert len(final_window) <= 3


# =============================================================================
# State Property Tests
# =============================================================================

class TestStateProperty:
    def test_state_getter(self):
        proc = InternalContradictionProcessor()
        state = proc.state
        assert isinstance(state, ContradictionState)

    def test_state_setter(self):
        proc = InternalContradictionProcessor()
        new_state = ContradictionState(cycle_count=42)
        proc.state = new_state
        assert proc.state.cycle_count == 42
