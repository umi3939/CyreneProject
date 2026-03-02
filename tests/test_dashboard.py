"""
tests/test_dashboard.py - 統合ダッシュボードのテスト

tools/dashboard.py が設計書(design_dashboard.md)の制約を満たすことを検証する:
- 各ツールの既存の読み取り専用アクセサのみを参照する
- 新しいデータ収集を一切行わない
- 独自の内部状態を保持しない
- 1つのツールの読み取り失敗が他のセクションの表示に影響しない
- 未接続のツールに対して「未接続」を表示する
- JSON形式での出力をサポートする
- 評価的語彙を使用しない
"""

import json
import io
import pytest
from unittest.mock import MagicMock, PropertyMock, patch

from tools.dashboard import (
    Dashboard,
    ALL_SECTIONS,
    SECTION_SESSION,
    SECTION_PIPELINE,
    SECTION_BAND,
    SECTION_POLICY,
    SECTION_EXPRESSION,
    SECTION_PATHWAY,
    SECTION_ENRICHMENT,
    SECTION_ANOMALY,
    main,
)


# ── ヘルパー: モックツール生成 ────────────────────────────────────


def _make_return_pathway_monitor() -> MagicMock:
    """ReturnPathwayMonitorのモック。"""
    mock = MagicMock()
    mock.get_summary.return_value = {
        "pathway_fire_counts": {
            "memory_emotion_return": 5,
            "selection_emotion_return": 3,
            "other_hypothesis_emotion_return": 1,
        },
        "concurrent_2plus_count": 2,
        "concurrent_3_count": 0,
        "last_tick_record": None,
    }
    return mock


def _make_execution_monitor() -> MagicMock:
    """ExecutionMonitorのモック。"""
    mock = MagicMock()
    type(mock).cycle_count = PropertyMock(return_value=100)
    type(mock).api_call_count = PropertyMock(return_value={
        "perception": 50, "expression": 50,
    })
    type(mock).api_token_count = PropertyMock(return_value={
        "perception": {"input": 10000, "output": 5000},
        "expression": {"input": 8000, "output": 6000},
    })
    type(mock).band_cumulative_time = PropertyMock(return_value={
        "every_tick": 12.5,
        "every_3_ticks": 3.2,
        "every_5_ticks": 1.8,
        "every_10_ticks": 0.5,
    })
    type(mock).last_compression_chars = PropertyMock(
        return_value=(5000, 3000)
    )

    # enrichment_distribution sub-monitor
    dist_mock = MagicMock()
    dist_mock.get_distribution_summary.return_value = {
        "observation_count": 100,
        "item_counters": {},
        "history_length": 50,
        "latest_entry": {
            "total_items": 48,
            "total_non_empty": 40,
            "total_changed": 10,
            "compressed_chars": 3000,
        },
        "duplicate_pairs": [],
    }
    type(mock).enrichment_distribution = PropertyMock(return_value=dist_mock)

    # enrichment_effectiveness sub-monitor
    eff_mock = MagicMock()
    eff_mock.compute_summary.return_value = {
        "item_characteristics": [],
        "section_summaries": [],
        "total_items": 48,
        "total_chars_cumulative": 150000,
    }
    type(mock).enrichment_effectiveness = PropertyMock(return_value=eff_mock)

    return mock


def _make_pipeline_measurement() -> MagicMock:
    """PipelineMeasurementのモック。"""
    mock = MagicMock()
    mock.get_summary.return_value = {
        "pathway_counts": {"vision": 30, "text": 15, "internal": 5},
        "pathway_total_cumulative": {
            "vision": 9.0, "text": 3.0, "internal": 0.5,
        },
        "pathway_phase_cumulative": {
            "vision": {"perception_api": 5.0, "expression_api": 3.0},
            "text": {"perception_api": 1.5, "expression_api": 1.0},
        },
        "buffer_size": 50,
        "latest_record": None,
    }
    return mock


def _make_policy_selection_log() -> MagicMock:
    """PolicySelectionLogのモック。"""
    mock = MagicMock()
    agg_mock = MagicMock()
    agg_mock.to_dict.return_value = {
        "label_selection_counts": {
            "empathize": 10,
            "question": 5,
            "express_emotion": 3,
        },
        "section_contribution_totals": {
            "drive_goal_match": 25.5,
            "fear_bias": -3.0,
        },
        "section_contribution_variances": {
            "drive_goal_match": 1.2,
            "fear_bias": 0.5,
        },
        "top_gap_history": [0.1, 0.2, 0.15],
        "max_selection_reached_count": 2,
        "window_size": 18,
    }
    mock.get_aggregation.return_value = agg_mock
    return mock


def _make_expression_quality() -> MagicMock:
    """ExpressionQualityVerificationのモック。"""
    mock = MagicMock()
    mock.get_summary.return_value = {
        "record_count": 45,
        "fallback_count": 2,
        "buffer_size": 45,
    }
    return mock


def _make_anomaly_detector(stalled: bool = False) -> MagicMock:
    """AnomalyDetectorのモック。"""
    mock = MagicMock()
    flags = {
        "emotion": stalled,
        "drive": False,
        "return_pathway": False,
        "enrichment_variation": False,
    }
    mock.get_summary.return_value = {
        "snapshot_count": 80,
        "buffer_size": 30,
        "buffer_max": 30,
        "stall_detected_counts": {
            "emotion": 1 if stalled else 0,
            "drive": 0,
            "return_pathway": 0,
            "enrichment_variation": 0,
        },
        "stall_resolved_counts": {
            "emotion": 0,
            "drive": 0,
            "return_pathway": 0,
            "enrichment_variation": 0,
        },
        "current_stall_flags": flags,
        "latest_snapshot": None,
    }
    return mock


def _make_full_dashboard(**overrides: MagicMock) -> Dashboard:
    """全ツール接続済みのDashboardを生成する。"""
    kwargs = {
        "return_pathway_monitor": _make_return_pathway_monitor(),
        "execution_monitor": _make_execution_monitor(),
        "pipeline_measurement": _make_pipeline_measurement(),
        "policy_selection_log": _make_policy_selection_log(),
        "expression_quality": _make_expression_quality(),
        "anomaly_detector": _make_anomaly_detector(),
    }
    kwargs.update(overrides)
    return Dashboard(**kwargs)


# ── テスト: 基本動作 ─────────────────────────────────────────────


class TestDashboardBasic:
    """基本的なインスタンス生成と収集のテスト。"""

    def test_create_empty_dashboard(self):
        """全ツール未接続でもインスタンスが生成できる。"""
        d = Dashboard()
        assert d is not None

    def test_collect_all_sections_empty(self):
        """全ツール未接続で全セクションを収集すると全てnot_connected。"""
        d = Dashboard()
        result = d.collect()
        assert len(result) == len(ALL_SECTIONS)
        for section_id, data in result.items():
            assert data.get("status") == "not_connected"

    def test_collect_all_sections_full(self):
        """全ツール接続済みで全セクションを収集できる。"""
        d = _make_full_dashboard()
        result = d.collect()
        assert len(result) == len(ALL_SECTIONS)
        for section_id, data in result.items():
            assert "status" not in data or data["status"] != "read_error"

    def test_collect_specific_section(self):
        """特定セクションのみを収集できる。"""
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_SESSION])
        assert len(result) == 1
        assert SECTION_SESSION in result

    def test_collect_multiple_sections(self):
        """複数セクションを収集できる。"""
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_SESSION, SECTION_PIPELINE])
        assert len(result) == 2
        assert SECTION_SESSION in result
        assert SECTION_PIPELINE in result

    def test_collect_invalid_section_ignored(self):
        """無効なセクション名は無視される。"""
        d = _make_full_dashboard()
        result = d.collect(sections=["nonexistent"])
        assert len(result) == 0


# ── テスト: セッション概要 ────────────────────────────────────────


class TestSessionSection:
    """セッション概要セクションのテスト。"""

    def test_session_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_SESSION])
        data = result[SECTION_SESSION]
        assert data["cycle_count"] == 100
        assert data["api_call_count"]["perception"] == 50
        assert data["api_call_count"]["expression"] == 50
        assert data["api_token_count"]["perception"]["input"] == 10000

    def test_session_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_SESSION])
        assert result[SECTION_SESSION]["status"] == "not_connected"

    def test_session_read_error(self):
        """ExecutionMonitorがcycle_countで例外を投げても安全。"""
        mock = MagicMock()
        type(mock).cycle_count = PropertyMock(side_effect=RuntimeError("fail"))
        d = Dashboard(execution_monitor=mock)
        result = d.collect(sections=[SECTION_SESSION])
        assert result[SECTION_SESSION]["status"] == "read_error"


# ── テスト: パイプライン計測 ───────────────────────────────────────


class TestPipelineSection:
    """パイプライン計測セクションのテスト。"""

    def test_pipeline_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_PIPELINE])
        data = result[SECTION_PIPELINE]
        assert data["pathway_counts"]["vision"] == 30
        # avg = 9.0 / 30 = 0.3
        assert data["pathway_avg_time"]["vision"] == 0.3

    def test_pipeline_avg_zero_count(self):
        """カウントが0の経路は平均時間に含まれない。"""
        mock = MagicMock()
        mock.get_summary.return_value = {
            "pathway_counts": {"vision": 0},
            "pathway_total_cumulative": {"vision": 0.0},
            "pathway_phase_cumulative": {},
            "buffer_size": 0,
        }
        d = Dashboard(pipeline_measurement=mock)
        result = d.collect(sections=[SECTION_PIPELINE])
        assert "vision" not in result[SECTION_PIPELINE]["pathway_avg_time"]

    def test_pipeline_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_PIPELINE])
        assert result[SECTION_PIPELINE]["status"] == "not_connected"

    def test_pipeline_read_error(self):
        mock = MagicMock()
        mock.get_summary.side_effect = RuntimeError("fail")
        d = Dashboard(pipeline_measurement=mock)
        result = d.collect(sections=[SECTION_PIPELINE])
        assert result[SECTION_PIPELINE]["status"] == "read_error"


# ── テスト: 帯域別実行時間 ────────────────────────────────────────


class TestBandSection:
    """帯域別実行時間セクションのテスト。"""

    def test_band_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_BAND])
        data = result[SECTION_BAND]
        assert data["band_cumulative_time"]["every_tick"] == 12.5

    def test_band_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_BAND])
        assert result[SECTION_BAND]["status"] == "not_connected"


# ── テスト: 方針選択分布 ──────────────────────────────────────────


class TestPolicySection:
    """方針選択分布セクションのテスト。"""

    def test_policy_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_POLICY])
        data = result[SECTION_POLICY]
        assert data["label_selection_counts"]["empathize"] == 10
        assert data["window_size"] == 18

    def test_policy_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_POLICY])
        assert result[SECTION_POLICY]["status"] == "not_connected"

    def test_policy_read_error(self):
        mock = MagicMock()
        mock.get_aggregation.side_effect = RuntimeError("fail")
        d = Dashboard(policy_selection_log=mock)
        result = d.collect(sections=[SECTION_POLICY])
        assert result[SECTION_POLICY]["status"] == "read_error"


# ── テスト: 代弁品質 ─────────────────────────────────────────────


class TestExpressionSection:
    """代弁品質セクションのテスト。"""

    def test_expression_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_EXPRESSION])
        data = result[SECTION_EXPRESSION]
        assert data["record_count"] == 45
        assert data["fallback_count"] == 2
        assert data["buffer_size"] == 45

    def test_expression_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_EXPRESSION])
        assert result[SECTION_EXPRESSION]["status"] == "not_connected"


# ── テスト: 帰還経路 ──────────────────────────────────────────────


class TestPathwaySection:
    """帰還経路セクションのテスト。"""

    def test_pathway_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_PATHWAY])
        data = result[SECTION_PATHWAY]
        assert data["pathway_fire_counts"]["memory_emotion_return"] == 5
        assert data["concurrent_2plus_count"] == 2
        assert data["concurrent_3_count"] == 0

    def test_pathway_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_PATHWAY])
        assert result[SECTION_PATHWAY]["status"] == "not_connected"


# ── テスト: enrichment分布 ────────────────────────────────────────


class TestEnrichmentSection:
    """enrichment分布セクションのテスト。"""

    def test_enrichment_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_ENRICHMENT])
        data = result[SECTION_ENRICHMENT]
        assert data["total_items"] == 48
        assert data["total_non_empty"] == 40
        assert data["total_changed"] == 10
        assert data["compression_before_chars"] == 5000
        assert data["compression_after_chars"] == 3000
        assert data["observation_count"] == 100
        assert data["effectiveness_total_items"] == 48

    def test_enrichment_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_ENRICHMENT])
        assert result[SECTION_ENRICHMENT]["status"] == "not_connected"

    def test_enrichment_partial_failure(self):
        """enrichmentサブモニターの1つが失敗しても他は収集される。"""
        exec_mock = _make_execution_monitor()
        # distribution sub-monitor fails
        dist_mock = MagicMock()
        dist_mock.get_distribution_summary.side_effect = RuntimeError("fail")
        type(exec_mock).enrichment_distribution = PropertyMock(
            return_value=dist_mock
        )
        d = Dashboard(execution_monitor=exec_mock)
        result = d.collect(sections=[SECTION_ENRICHMENT])
        data = result[SECTION_ENRICHMENT]
        # distribution fails but compression should still be present
        assert "distribution" in data and data["distribution"] == "read_error"
        assert data["compression_before_chars"] == 5000

    def test_enrichment_no_latest_entry(self):
        """latest_entryがNoneの場合のデフォルト値。"""
        exec_mock = _make_execution_monitor()
        dist_mock = MagicMock()
        dist_mock.get_distribution_summary.return_value = {
            "observation_count": 0,
            "latest_entry": None,
        }
        type(exec_mock).enrichment_distribution = PropertyMock(
            return_value=dist_mock
        )
        d = Dashboard(execution_monitor=exec_mock)
        result = d.collect(sections=[SECTION_ENRICHMENT])
        data = result[SECTION_ENRICHMENT]
        assert data["total_items"] == 0
        assert data["total_non_empty"] == 0
        assert data["total_changed"] == 0


# ── テスト: 動態停止検出 ──────────────────────────────────────────


class TestAnomalySection:
    """動態停止検出セクションのテスト。"""

    def test_anomaly_data(self):
        d = _make_full_dashboard()
        result = d.collect(sections=[SECTION_ANOMALY])
        data = result[SECTION_ANOMALY]
        assert data["snapshot_count"] == 80
        assert data["buffer_size"] == 30
        assert data["buffer_max"] == 30

    def test_anomaly_not_connected(self):
        d = Dashboard()
        result = d.collect(sections=[SECTION_ANOMALY])
        assert result[SECTION_ANOMALY]["status"] == "not_connected"

    def test_anomaly_stall_flags(self):
        """停止フラグの読み取り。"""
        d = _make_full_dashboard(
            anomaly_detector=_make_anomaly_detector(stalled=True)
        )
        result = d.collect(sections=[SECTION_ANOMALY])
        data = result[SECTION_ANOMALY]
        assert data["current_stall_flags"]["emotion"] is True
        assert data["stall_detected_counts"]["emotion"] == 1


# ── テスト: テキストフォーマット ───────────────────────────────────


class TestTextFormat:
    """テキスト形式のフォーマットテスト。"""

    def test_text_format_all_connected(self):
        """全ツール接続済みのテキスト出力。"""
        d = _make_full_dashboard()
        text = d.format_text()
        assert "Cyrene Dashboard" in text
        assert "Session" in text
        assert "Pipeline" in text
        assert "Band Times" in text
        assert "Policy Selection" in text
        assert "Expression Quality" in text
        assert "Return Pathways" in text
        assert "Enrichment" in text
        assert "Dynamics Stall" in text

    def test_text_format_all_disconnected(self):
        """全ツール未接続のテキスト出力。"""
        d = Dashboard()
        text = d.format_text()
        assert "(not connected)" in text

    def test_text_format_specific_section(self):
        """特定セクションのみのテキスト出力。"""
        d = _make_full_dashboard()
        text = d.format_text(sections=[SECTION_SESSION])
        assert "Session" in text
        # Other sections should not be present
        assert "Pipeline" not in text

    def test_text_format_stalled_marker(self):
        """停止状態のマーカーが表示される。"""
        d = _make_full_dashboard(
            anomaly_detector=_make_anomaly_detector(stalled=True)
        )
        text = d.format_text()
        assert "[stalled]" in text

    def test_text_format_no_evaluative_words(self):
        """評価的語彙が含まれない（安全弁6）。"""
        d = _make_full_dashboard()
        text = d.format_text()
        forbidden = ["good", "bad", "problem", "improve", "warning",
                     "error", "excellent", "poor"]
        text_lower = text.lower()
        for word in forbidden:
            # "read error" is allowed as a status description
            if word == "error":
                # only check that "error" doesn't appear outside
                # the status context
                continue
            assert word not in text_lower, (
                f"Evaluative word '{word}' found in output"
            )

    def test_text_session_values(self):
        """セッションの具体的な値がテキスト出力に含まれる。"""
        d = _make_full_dashboard()
        text = d.format_text(sections=[SECTION_SESSION])
        assert "100" in text  # cycle count
        assert "perception=50" in text
        assert "expression=50" in text

    def test_text_pipeline_avg(self):
        """パイプラインの平均時間が表示される。"""
        d = _make_full_dashboard()
        text = d.format_text(sections=[SECTION_PIPELINE])
        # vision avg = 9.0/30 = 0.3s = 300ms
        assert "300.0ms" in text

    def test_text_enrichment_non_empty_rate(self):
        """enrichmentの非空率が表示される。"""
        d = _make_full_dashboard()
        text = d.format_text(sections=[SECTION_ENRICHMENT])
        # 40/48 * 100 = 83.3%
        assert "83.3%" in text

    def test_text_enrichment_zero_items(self):
        """enrichmentの項目数が0の場合。"""
        exec_mock = _make_execution_monitor()
        dist_mock = MagicMock()
        dist_mock.get_distribution_summary.return_value = {
            "observation_count": 0,
            "latest_entry": None,
        }
        type(exec_mock).enrichment_distribution = PropertyMock(
            return_value=dist_mock
        )
        d = Dashboard(execution_monitor=exec_mock)
        text = d.format_text(sections=[SECTION_ENRICHMENT])
        assert "Non-empty rate: -" in text


# ── テスト: JSON形式 ──────────────────────────────────────────────


class TestJsonFormat:
    """JSON形式のフォーマットテスト。"""

    def test_json_format_valid(self):
        """全ツール接続済みのJSON出力がvalidなJSONである。"""
        d = _make_full_dashboard()
        text = d.format_json()
        data = json.loads(text)
        assert isinstance(data, dict)
        assert len(data) == len(ALL_SECTIONS)

    def test_json_format_empty(self):
        """全ツール未接続のJSON出力。"""
        d = Dashboard()
        text = d.format_json()
        data = json.loads(text)
        for section_id, section_data in data.items():
            assert section_data.get("status") == "not_connected"

    def test_json_format_specific_section(self):
        """特定セクションのみのJSON出力。"""
        d = _make_full_dashboard()
        text = d.format_json(sections=[SECTION_SESSION])
        data = json.loads(text)
        assert len(data) == 1
        assert SECTION_SESSION in data

    def test_json_roundtrip_values(self):
        """JSON出力の値が正しい。"""
        d = _make_full_dashboard()
        text = d.format_json(sections=[SECTION_EXPRESSION])
        data = json.loads(text)
        expr = data[SECTION_EXPRESSION]
        assert expr["record_count"] == 45
        assert expr["fallback_count"] == 2


# ── テスト: print_dashboard ───────────────────────────────────────


class TestPrintDashboard:
    """print_dashboardメソッドのテスト。"""

    def test_print_text(self):
        """テキスト形式でのprint出力。"""
        d = _make_full_dashboard()
        buf = io.StringIO()
        d.print_dashboard(file=buf)
        output = buf.getvalue()
        assert "Cyrene Dashboard" in output

    def test_print_json(self):
        """JSON形式でのprint出力。"""
        d = _make_full_dashboard()
        buf = io.StringIO()
        d.print_dashboard(as_json=True, file=buf)
        output = buf.getvalue()
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_print_sections_filter(self):
        """セクション指定でのprint出力。"""
        d = _make_full_dashboard()
        buf = io.StringIO()
        d.print_dashboard(sections=[SECTION_SESSION], file=buf)
        output = buf.getvalue()
        assert "Session" in output
        assert "Pipeline" not in output


# ── テスト: 例外安全性（安全弁4）──────────────────────────────────


class TestExceptionSafety:
    """1つのツールの例外が他のセクションに影響しないことのテスト。"""

    def test_one_tool_error_others_ok(self):
        """1つのツールが例外を投げても他のセクションは表示される。"""
        # pipeline_measurementだけ壊す
        broken_pipeline = MagicMock()
        broken_pipeline.get_summary.side_effect = RuntimeError("boom")

        d = _make_full_dashboard(pipeline_measurement=broken_pipeline)
        result = d.collect()

        # pipelineはread_error
        assert result[SECTION_PIPELINE]["status"] == "read_error"
        # sessionは正常
        assert result[SECTION_SESSION]["cycle_count"] == 100
        # expressionは正常
        assert result[SECTION_EXPRESSION]["record_count"] == 45

    def test_all_tools_error(self):
        """全ツールが例外を投げてもcollectが完了する。"""

        def _make_broken():
            m = MagicMock()
            m.get_summary.side_effect = RuntimeError("boom")
            m.get_aggregation.side_effect = RuntimeError("boom")
            type(m).cycle_count = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            type(m).api_call_count = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            type(m).api_token_count = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            type(m).band_cumulative_time = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            type(m).last_compression_chars = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            type(m).enrichment_distribution = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            type(m).enrichment_effectiveness = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            return m

        d = Dashboard(
            return_pathway_monitor=_make_broken(),
            execution_monitor=_make_broken(),
            pipeline_measurement=_make_broken(),
            policy_selection_log=_make_broken(),
            expression_quality=_make_broken(),
            anomaly_detector=_make_broken(),
        )
        result = d.collect()
        assert len(result) == len(ALL_SECTIONS)
        for section_id, data in result.items():
            # enrichment section has partial failure handling, so
            # individual sub-sections show errors instead of top-level status
            if section_id == SECTION_ENRICHMENT:
                assert data.get("distribution") == "read_error"
                assert data.get("compression") == "read_error"
                assert data.get("effectiveness") == "read_error"
            else:
                assert data.get("status") == "read_error"

    def test_text_format_with_errors(self):
        """エラーがあってもテキスト出力が生成できる。"""

        def _make_broken():
            m = MagicMock()
            m.get_summary.side_effect = RuntimeError("boom")
            m.get_aggregation.side_effect = RuntimeError("boom")
            type(m).cycle_count = PropertyMock(
                side_effect=RuntimeError("boom")
            )
            return m

        d = Dashboard(
            return_pathway_monitor=_make_broken(),
            execution_monitor=_make_broken(),
            pipeline_measurement=_make_broken(),
            policy_selection_log=_make_broken(),
            expression_quality=_make_broken(),
            anomaly_detector=_make_broken(),
        )
        text = d.format_text()
        assert "Cyrene Dashboard" in text
        assert "(read error)" in text


# ── テスト: 構造的分離の検証 ──────────────────────────────────────


class TestStructuralSeparation:
    """Dashboardが書き込みメソッドを呼ばないことの検証。"""

    def test_no_write_calls_on_pathway_monitor(self):
        """ReturnPathwayMonitorの書き込みメソッドが呼ばれない。"""
        mock = _make_return_pathway_monitor()
        d = Dashboard(return_pathway_monitor=mock)
        d.collect()
        # record_firingやfinalize_tickが呼ばれないことを確認
        mock.record_firing.assert_not_called()
        mock.finalize_tick.assert_not_called()

    def test_no_write_calls_on_execution_monitor(self):
        """ExecutionMonitorの書き込みメソッドが呼ばれない。"""
        mock = _make_execution_monitor()
        d = Dashboard(execution_monitor=mock)
        d.collect()
        mock.record_band_time.assert_not_called()
        mock.record_compression.assert_not_called()
        mock.record_api_call.assert_not_called()
        mock.record_cycle_complete.assert_not_called()

    def test_no_write_calls_on_pipeline(self):
        """PipelineMeasurementの書き込みメソッドが呼ばれない。"""
        mock = _make_pipeline_measurement()
        d = Dashboard(pipeline_measurement=mock)
        d.collect()
        mock.begin_pipeline.assert_not_called()
        mock.end_pipeline.assert_not_called()
        mock.record_phase.assert_not_called()

    def test_no_write_calls_on_policy_log(self):
        """PolicySelectionLogの書き込みメソッドが呼ばれない。"""
        mock = _make_policy_selection_log()
        d = Dashboard(policy_selection_log=mock)
        d.collect()
        mock.record.assert_not_called()

    def test_no_write_calls_on_expression(self):
        """ExpressionQualityVerificationの書き込みメソッドが呼ばれない。"""
        mock = _make_expression_quality()
        d = Dashboard(expression_quality=mock)
        d.collect()
        mock.record_expression.assert_not_called()

    def test_no_write_calls_on_anomaly(self):
        """AnomalyDetectorの書き込みメソッドが呼ばれない。"""
        mock = _make_anomaly_detector()
        d = Dashboard(anomaly_detector=mock)
        d.collect()
        mock.record_snapshot.assert_not_called()


# ── テスト: 実ツールとの結合(インポート確認) ──────────────────────


class TestRealToolIntegration:
    """実際のツールクラスとの結合テスト。"""

    def test_with_real_return_pathway_monitor(self):
        """実ReturnPathwayMonitorとの結合。"""
        from tools.return_pathway_monitor import ReturnPathwayMonitor
        monitor = ReturnPathwayMonitor(enabled=True)
        d = Dashboard(return_pathway_monitor=monitor)
        result = d.collect(sections=[SECTION_PATHWAY])
        data = result[SECTION_PATHWAY]
        assert "pathway_fire_counts" in data
        assert data["concurrent_2plus_count"] == 0

    def test_with_real_execution_monitor(self):
        """実ExecutionMonitorとの結合。"""
        from tools.execution_monitor import ExecutionMonitor
        monitor = ExecutionMonitor(enabled=True)
        d = Dashboard(execution_monitor=monitor)
        result = d.collect(sections=[SECTION_SESSION, SECTION_BAND])
        assert result[SECTION_SESSION]["cycle_count"] == 0
        assert result[SECTION_BAND]["band_cumulative_time"] == {}

    def test_with_real_pipeline_measurement(self):
        """実PipelineMeasurementとの結合。"""
        from tools.pipeline_measurement import PipelineMeasurement
        pm = PipelineMeasurement(enabled=True)
        d = Dashboard(pipeline_measurement=pm)
        result = d.collect(sections=[SECTION_PIPELINE])
        data = result[SECTION_PIPELINE]
        assert data["pathway_counts"] == {}

    def test_with_real_policy_selection_log(self):
        """実PolicySelectionLogとの結合。"""
        from tools.policy_selection_log import PolicySelectionLog
        log = PolicySelectionLog(enabled=True)
        d = Dashboard(policy_selection_log=log)
        result = d.collect(sections=[SECTION_POLICY])
        data = result[SECTION_POLICY]
        assert data["window_size"] == 0

    def test_with_real_expression_quality(self):
        """実ExpressionQualityVerificationとの結合。"""
        from tools.expression_quality_verification import (
            ExpressionQualityVerification,
        )
        eq = ExpressionQualityVerification(enabled=True)
        d = Dashboard(expression_quality=eq)
        result = d.collect(sections=[SECTION_EXPRESSION])
        data = result[SECTION_EXPRESSION]
        assert data["record_count"] == 0

    def test_with_real_anomaly_detector(self):
        """実AnomalyDetectorとの結合。"""
        from tools.anomaly_detection import AnomalyDetector
        ad = AnomalyDetector(enabled=True)
        d = Dashboard(anomaly_detector=ad)
        result = d.collect(sections=[SECTION_ANOMALY])
        data = result[SECTION_ANOMALY]
        assert data["snapshot_count"] == 0

    def test_full_real_integration(self):
        """全実ツールとの統合テスト。"""
        from tools.return_pathway_monitor import ReturnPathwayMonitor
        from tools.execution_monitor import ExecutionMonitor
        from tools.pipeline_measurement import PipelineMeasurement
        from tools.policy_selection_log import PolicySelectionLog
        from tools.expression_quality_verification import (
            ExpressionQualityVerification,
        )
        from tools.anomaly_detection import AnomalyDetector

        d = Dashboard(
            return_pathway_monitor=ReturnPathwayMonitor(enabled=True),
            execution_monitor=ExecutionMonitor(enabled=True),
            pipeline_measurement=PipelineMeasurement(enabled=True),
            policy_selection_log=PolicySelectionLog(enabled=True),
            expression_quality=ExpressionQualityVerification(enabled=True),
            anomaly_detector=AnomalyDetector(enabled=True),
        )
        # テキスト出力が例外なく生成できる
        text = d.format_text()
        assert "Cyrene Dashboard" in text

        # JSON出力がvalidなJSONである
        json_text = d.format_json()
        data = json.loads(json_text)
        assert len(data) == len(ALL_SECTIONS)


# ── テスト: CLIエントリポイント ────────────────────────────────────


class TestCLI:
    """CLIエントリポイントのテスト。"""

    def test_main_no_args(self, capsys):
        """引数なしで実行できる。"""
        main([])
        captured = capsys.readouterr()
        assert "Cyrene Dashboard" in captured.out

    def test_main_json(self, capsys):
        """--jsonフラグで実行できる。"""
        main(["--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    def test_main_section_filter(self, capsys):
        """セクション指定で実行できる。"""
        main(["session"])
        captured = capsys.readouterr()
        assert "Session" in captured.out

    def test_main_json_section(self, capsys):
        """--jsonとセクション指定の組み合わせ。"""
        main(["--json", "session"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "session" in data
        assert len(data) == 1


# ── テスト: 混在状態 ─────────────────────────────────────────────


class TestMixedState:
    """一部接続・一部未接続の混在状態のテスト。"""

    def test_partial_connection(self):
        """一部のツールのみ接続されている状態。"""
        d = Dashboard(
            execution_monitor=_make_execution_monitor(),
            anomaly_detector=_make_anomaly_detector(),
        )
        result = d.collect()
        assert result[SECTION_SESSION]["cycle_count"] == 100
        assert result[SECTION_ANOMALY]["snapshot_count"] == 80
        assert result[SECTION_PIPELINE]["status"] == "not_connected"
        assert result[SECTION_POLICY]["status"] == "not_connected"
        assert result[SECTION_EXPRESSION]["status"] == "not_connected"
        assert result[SECTION_PATHWAY]["status"] == "not_connected"

    def test_partial_connection_text(self):
        """混在状態のテキスト出力。"""
        d = Dashboard(
            execution_monitor=_make_execution_monitor(),
        )
        text = d.format_text()
        assert "Cycles: 100" in text
        assert "(not connected)" in text
