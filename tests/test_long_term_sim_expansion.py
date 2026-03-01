"""
tests/test_long_term_sim_expansion.py - 長期シミュレータ拡張機能のテスト

新規シナリオ(5種)、拡張ターンレコード(enrichment文字数)、
統計サマリー、シナリオ間差分レポートのテスト。
"""

from __future__ import annotations

import math

import pytest

from psyche.state import Percept
from tools.long_term_sim import (
    ENRICHMENT_SECTION_HEADERS,
    INPUT_PATTERNS,
    SCENARIO_USER_IDS,
    SCENARIOS,
    _measure_enrichment_sections,
    _pattern_to_outcome,
    compute_statistics,
    generate_diff_report,
    run_simulation,
)


# ══════════════════════════════════════════════════════════════
# 第1段: 新規シナリオ定義のテスト
# ══════════════════════════════════════════════════════════════


class TestNewScenarioDefinitions:
    """新規5シナリオの定義が正しいことを検証する。"""

    NEW_SCENARIOS = ["stable", "high_variation", "long_silence", "multi_person", "gradual_shift"]

    def test_new_scenarios_exist_in_dict(self):
        """5つの新規シナリオがSCENARIOS辞書に存在する。"""
        for name in self.NEW_SCENARIOS:
            assert name in SCENARIOS, f"Scenario '{name}' not found in SCENARIOS"

    def test_new_scenarios_use_valid_patterns(self):
        """新規シナリオの全パターンキーがINPUT_PATTERNSに存在する。"""
        for name in self.NEW_SCENARIOS:
            for key in SCENARIOS[name]:
                assert key in INPUT_PATTERNS, (
                    f"Scenario '{name}' uses unknown pattern '{key}'"
                )

    def test_new_scenarios_have_positive_length(self):
        """新規シナリオは全て1ターン以上である。"""
        for name in self.NEW_SCENARIOS:
            assert len(SCENARIOS[name]) > 0, (
                f"Scenario '{name}' has 0 turns"
            )

    def test_stable_scenario_content(self):
        """stable: positiveとneutralの繰り返し。"""
        seq = SCENARIOS["stable"]
        unique_keys = set(seq)
        assert unique_keys == {"positive", "neutral"}

    def test_high_variation_scenario_has_many_pattern_types(self):
        """high_variation: 少なくとも4種以上のパターンを含む。"""
        seq = SCENARIOS["high_variation"]
        unique_keys = set(seq)
        assert len(unique_keys) >= 4

    def test_long_silence_scenario_dominated_by_neutral(self):
        """long_silence: neutralが過半数を占める。"""
        seq = SCENARIOS["long_silence"]
        neutral_count = sum(1 for k in seq if k == "neutral")
        assert neutral_count > len(seq) / 2

    def test_multi_person_scenario_has_user_id_mapping(self):
        """multi_person: SCENARIO_USER_IDSにuser_id切替情報がある。"""
        assert "multi_person" in SCENARIO_USER_IDS
        ids = SCENARIO_USER_IDS["multi_person"]
        assert len(ids) == len(SCENARIOS["multi_person"])
        # 少なくとも2種類のuser_idが含まれる
        assert len(set(ids)) >= 2

    def test_gradual_shift_scenario_structure(self):
        """gradual_shift: positiveからangryへの段階的遷移。"""
        seq = SCENARIOS["gradual_shift"]
        # 先頭はpositive
        assert seq[0] == "positive"
        # 末尾はangry
        assert seq[-1] == "angry"

    def test_existing_scenarios_unchanged(self):
        """既存シナリオが変更されていないことを確認する。"""
        assert SCENARIOS["repeated_failure"] == ["negative", "rejected"] * 25
        assert SCENARIOS["smoke"] == ["positive", "negative", "confused", "neutral", "angry"]
        assert len(SCENARIOS["neutral_baseline"]) == 60


# ══════════════════════════════════════════════════════════════
# 第2段: enrichment文字数計測のテスト
# ══════════════════════════════════════════════════════════════


class TestEnrichmentMeasurement:
    """enrichmentテキストの文字数計測をテストする。"""

    def test_measure_empty_text(self):
        """空テキストではtotalが0、各セクションも0。"""
        result = _measure_enrichment_sections("")
        assert result["total"] == 0
        for header in ENRICHMENT_SECTION_HEADERS:
            assert result[header] == 0

    def test_measure_text_with_all_sections(self):
        """全5セクションを含むテキストの計測。"""
        text = (
            "【心理状態（内面）】\n感情: happy=0.5\n"
            "【自己認識】\n自己モデル: 安定\n"
            "【動機・目標】\nドライブ: social=0.3\n"
            "【記憶・内省】\nエピソード: 3件\n"
            "【判断傾向】\n慎重度: 中\n"
        )
        result = _measure_enrichment_sections(text)
        assert result["total"] == len(text)
        # 各セクションが0より大きい
        for header in ENRICHMENT_SECTION_HEADERS:
            assert result[header] > 0

    def test_measure_text_with_partial_sections(self):
        """一部セクションのみのテキスト。"""
        text = "【心理状態（内面）】\n感情: happy=0.5\n【自己認識】\n自己モデル: 安定\n"
        result = _measure_enrichment_sections(text)
        assert result["total"] == len(text)
        assert result["【心理状態（内面）】"] > 0
        assert result["【自己認識】"] > 0
        assert result["【動機・目標】"] == 0

    def test_total_is_always_text_length(self):
        """totalは常にテキスト全体の文字数と一致する。"""
        text = "some random text without sections"
        result = _measure_enrichment_sections(text)
        assert result["total"] == len(text)

    def test_enrichment_section_headers_has_5_entries(self):
        """ENRICHMENT_SECTION_HEADERSが5セクション分ある。"""
        assert len(ENRICHMENT_SECTION_HEADERS) == 5


# ══════════════════════════════════════════════════════════════
# 第3段: 拡張ターンレコードのテスト
# ══════════════════════════════════════════════════════════════


class TestExtendedTurnRecord:
    """拡張ターンレコード（enrichment_charsフィールド）のテスト。"""

    @pytest.fixture(scope="class")
    def smoke_result(self) -> dict:
        """smokeシナリオの結果。"""
        return run_simulation(scenario_name="smoke")

    def test_enrichment_chars_present_in_all_turns(self, smoke_result):
        """全ターンにenrichment_charsフィールドが存在する。"""
        for rec in smoke_result["turns"]:
            assert "enrichment_chars" in rec, (
                f"Turn {rec['turn']}: missing enrichment_chars"
            )

    def test_enrichment_chars_has_total(self, smoke_result):
        """enrichment_charsにtotalキーが存在する。"""
        for rec in smoke_result["turns"]:
            ec = rec["enrichment_chars"]
            assert "total" in ec
            assert isinstance(ec["total"], int)
            assert ec["total"] >= 0

    def test_enrichment_chars_has_section_keys(self, smoke_result):
        """enrichment_charsに全5セクションのキーが存在する。"""
        for rec in smoke_result["turns"]:
            ec = rec["enrichment_chars"]
            for header in ENRICHMENT_SECTION_HEADERS:
                assert header in ec, (
                    f"Turn {rec['turn']}: missing section '{header}'"
                )

    def test_existing_fields_still_present(self, smoke_result):
        """既存フィールドが全て維持されている。"""
        required = {
            "turn", "tick", "input_pattern", "input",
            "psyche_state", "responsibility", "responsibility_influence",
            "policy", "outcome_applied",
        }
        for rec in smoke_result["turns"]:
            missing = required - set(rec.keys())
            assert not missing, f"Turn {rec.get('turn')}: missing {missing}"

    def test_version_is_3(self, smoke_result):
        """バージョンが3に更新されている。"""
        assert smoke_result["metadata"]["version"] == 3


# ══════════════════════════════════════════════════════════════
# 第4段: 統計サマリーのテスト
# ══════════════════════════════════════════════════════════════


class TestStatisticsSummary:
    """compute_statistics()の検証。"""

    @pytest.fixture(scope="class")
    def smoke_result(self) -> dict:
        return run_simulation(scenario_name="smoke")

    @pytest.fixture(scope="class")
    def smoke_stats(self, smoke_result) -> dict:
        return compute_statistics(smoke_result)

    def test_stats_has_scenario_name(self, smoke_stats):
        """統計にシナリオ名が含まれる。"""
        assert smoke_stats["scenario"] == "smoke"

    def test_stats_has_total_turns(self, smoke_stats):
        """統計にターン数が含まれる。"""
        assert smoke_stats["total_turns"] == len(SCENARIOS["smoke"])

    def test_stats_has_emotions(self, smoke_stats):
        """感情統計が含まれる。"""
        assert "emotions" in smoke_stats
        for emo_key, emo_st in smoke_stats["emotions"].items():
            assert "min" in emo_st
            assert "max" in emo_st
            assert "mean" in emo_st
            assert "stddev" in emo_st
            assert emo_st["min"] <= emo_st["max"]
            assert emo_st["min"] <= emo_st["mean"] <= emo_st["max"]
            assert emo_st["stddev"] >= 0.0

    def test_stats_has_mood(self, smoke_stats):
        """ムード統計が含まれる。"""
        assert "mood" in smoke_stats
        for mood_key in ["valence", "arousal"]:
            assert mood_key in smoke_stats["mood"]
            ms = smoke_stats["mood"][mood_key]
            assert ms["min"] <= ms["max"]

    def test_stats_has_drives(self, smoke_stats):
        """ドライブ統計が含まれる。"""
        assert "drives" in smoke_stats
        for drv_key, drv_st in smoke_stats["drives"].items():
            assert "min" in drv_st
            assert "max" in drv_st
            assert "mean" in drv_st
            assert "stddev" in drv_st

    def test_stats_has_fear_level(self, smoke_stats):
        """恐怖指数統計が含まれる。"""
        assert "fear_level" in smoke_stats
        fl = smoke_stats["fear_level"]
        assert fl["min"] <= fl["max"]
        assert fl["stddev"] >= 0.0

    def test_stats_has_policy_distribution(self, smoke_stats):
        """ポリシー分布統計が含まれる。"""
        assert "policy_distribution" in smoke_stats
        pd = smoke_stats["policy_distribution"]
        # 全countの合計がtotal_turnsに等しい
        total_count = sum(v["count"] for v in pd.values())
        assert total_count == smoke_stats["total_turns"]
        # 全ratioの合計がほぼ1.0
        total_ratio = sum(v["ratio"] for v in pd.values())
        assert abs(total_ratio - 1.0) < 0.01

    def test_stats_has_enrichment_total_chars(self, smoke_stats):
        """enrichment総文字数統計が含まれる。"""
        assert "enrichment_total_chars" in smoke_stats
        ec = smoke_stats["enrichment_total_chars"]
        assert ec["min"] <= ec["max"]
        assert ec["mean"] >= 0

    def test_stats_has_enrichment_sections(self, smoke_stats):
        """enrichmentセクション別統計が含まれる。"""
        assert "enrichment_sections" in smoke_stats

    def test_stats_empty_turns(self):
        """空ターンリストの場合にtotal_turns=0を返す。"""
        empty_result = {
            "metadata": {"scenario": "test", "total_turns": 0},
            "turns": [],
        }
        stats = compute_statistics(empty_result)
        assert stats["total_turns"] == 0

    def test_stddev_single_value(self):
        """単一ターンの場合にstddev=0.0。"""
        result = run_simulation(custom_sequence=["neutral"])
        stats = compute_statistics(result)
        assert stats["fear_level"]["stddev"] == 0.0


# ══════════════════════════════════════════════════════════════
# 第5段: シナリオ間差分レポートのテスト
# ══════════════════════════════════════════════════════════════


class TestDiffReport:
    """generate_diff_report()の検証。"""

    @pytest.fixture(scope="class")
    def two_scenario_results(self) -> dict[str, dict]:
        """2シナリオ（smoke + stable短縮版）の結果。"""
        return {
            "smoke": run_simulation(scenario_name="smoke"),
            "custom_a": run_simulation(
                custom_sequence=["neutral", "neutral", "positive"]
            ),
        }

    @pytest.fixture(scope="class")
    def diff_report(self, two_scenario_results) -> dict:
        return generate_diff_report(two_scenario_results)

    def test_report_has_scenarios_compared(self, diff_report):
        """比較シナリオ名のリストが含まれる。"""
        assert "scenarios_compared" in diff_report
        assert len(diff_report["scenarios_compared"]) == 2

    def test_report_has_generated_at(self, diff_report):
        """生成日時が含まれる。"""
        assert "generated_at" in diff_report

    def test_report_has_final_emotions(self, diff_report):
        """終了時点の感情分布比較が含まれる。"""
        assert "final_emotions" in diff_report
        for sc_name, emotions in diff_report["final_emotions"].items():
            assert isinstance(emotions, dict)

    def test_report_has_mood_ranges(self, diff_report):
        """ムード推移範囲の比較が含まれる。"""
        assert "mood_ranges" in diff_report

    def test_report_has_policy_distributions(self, diff_report):
        """ポリシー分布の比較が含まれる。"""
        assert "policy_distributions" in diff_report

    def test_report_has_enrichment_ranges(self, diff_report):
        """enrichment文字数範囲の比較が含まれる。"""
        assert "enrichment_total_char_ranges" in diff_report

    def test_report_has_fear_level_ranges(self, diff_report):
        """恐怖指数範囲の比較が含まれる。"""
        assert "fear_level_ranges" in diff_report

    def test_diff_report_requires_at_least_2(self):
        """1件のみではValueErrorになる。"""
        single_result = {
            "only_one": run_simulation(custom_sequence=["neutral"]),
        }
        with pytest.raises(ValueError, match="At least 2"):
            generate_diff_report(single_result)


# ══════════════════════════════════════════════════════════════
# 第6段: 新規シナリオの実行テスト
# ══════════════════════════════════════════════════════════════


class TestNewScenarioExecution:
    """新規シナリオが実際にシミュレーション実行できることを検証する。

    各シナリオは短縮版ではなく定義全体を実行するが、
    テスト時間のため一部のみ実行可能版も用意する。
    """

    def test_stable_short_run(self):
        """stableの短縮版（先頭6ターン）が実行可能。"""
        seq = SCENARIOS["stable"][:6]
        result = run_simulation(custom_sequence=seq)
        assert len(result["turns"]) == 6

    def test_high_variation_short_run(self):
        """high_variationの短縮版が実行可能。"""
        seq = SCENARIOS["high_variation"][:6]
        result = run_simulation(custom_sequence=seq)
        assert len(result["turns"]) == 6

    def test_long_silence_short_run(self):
        """long_silenceの短縮版が実行可能。"""
        seq = SCENARIOS["long_silence"][:6]
        result = run_simulation(custom_sequence=seq)
        assert len(result["turns"]) == 6

    def test_multi_person_executes(self):
        """multi_personシナリオが実行可能。user_id切替が動作する。"""
        result = run_simulation(scenario_name="multi_person")
        turns = result["turns"]
        assert len(turns) == len(SCENARIOS["multi_person"])
        # multi_personの場合、enrichment_charsが全ターンに存在する
        for rec in turns:
            assert "enrichment_chars" in rec

    def test_gradual_shift_short_run(self):
        """gradual_shiftの短縮版が実行可能。"""
        seq = SCENARIOS["gradual_shift"][:6]
        result = run_simulation(custom_sequence=seq)
        assert len(result["turns"]) == 6


# ══════════════════════════════════════════════════════════════
# 第7段: CLI引数のテスト
# ══════════════════════════════════════════════════════════════


class TestCLIArgs:
    """CLIの新規引数のテスト（_build_parserの検証）。"""

    def test_stats_flag_exists(self):
        """--statsフラグがパーサーに存在する。"""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--scenario", "smoke", "--stats"])
        assert args.stats is True

    def test_compare_flag_exists(self):
        """--compareフラグがパーサーに存在する。"""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--compare", "smoke", "stable"])
        assert args.compare == ["smoke", "stable"]

    def test_existing_flags_unchanged(self):
        """既存フラグが変更されていない。"""
        from tools.long_term_sim import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--scenario", "smoke",
            "--output", "test.json",
            "--delta-time", "3.0",
            "--user-id", "test_user",
        ])
        assert args.scenario == "smoke"
        assert args.output == "test.json"
        assert args.delta_time == 3.0
        assert args.user_id == "test_user"


# ══════════════════════════════════════════════════════════════
# 第8段: 後方互換性テスト
# ══════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """既存インターフェースの後方互換性を検証する。"""

    @pytest.fixture(scope="class")
    def smoke_result(self) -> dict:
        return run_simulation(scenario_name="smoke")

    def test_run_simulation_signature(self):
        """run_simulationの既存引数が動作する。"""
        result = run_simulation(
            scenario_name="smoke",
            delta_time=2.0,
            user_id="sim_user",
        )
        assert "metadata" in result
        assert "turns" in result

    def test_custom_sequence_still_works(self):
        """カスタムシーケンスが既存通り動作する。"""
        result = run_simulation(custom_sequence=["positive", "negative"])
        assert len(result["turns"]) == 2
        assert result["metadata"]["scenario"] == "custom"

    def test_invalid_scenario_still_raises(self):
        """不正シナリオ名でValueErrorが出る。"""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_simulation(scenario_name="nonexistent")

    def test_invalid_pattern_still_raises(self):
        """不正パターンキーでValueErrorが出る。"""
        with pytest.raises(ValueError, match="Invalid pattern key"):
            run_simulation(custom_sequence=["nonexistent_pattern"])

    def test_existing_scenario_results_unchanged(self, smoke_result):
        """既存のsmokeシナリオの基本構造が維持されている。"""
        assert smoke_result["metadata"]["scenario"] == "smoke"
        assert len(smoke_result["turns"]) == len(SCENARIOS["smoke"])
        # 既存の全フィールドが存在
        rec = smoke_result["turns"][0]
        assert "psyche_state" in rec
        assert "responsibility" in rec
        assert "policy" in rec

    def test_pattern_to_outcome_unchanged(self):
        """_pattern_to_outcomeの戻り値が既存通り。"""
        outcome = _pattern_to_outcome("positive")
        assert outcome["user_reaction"] == "positive"
        assert outcome["relationship_delta"] == 0.1


# ══════════════════════════════════════════════════════════════
# 第9段: 統計値の数学的正当性テスト
# ══════════════════════════════════════════════════════════════


class TestStatisticalCorrectness:
    """統計値の数学的正当性を検証する。"""

    def test_stddev_known_values(self):
        """既知の値リストでstddevの正しさを確認する。"""
        from tools.long_term_sim import _safe_stddev
        # [2, 4, 4, 4, 5, 5, 7, 9] -> mean=5, variance=4, stddev=2
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = _safe_stddev(values)
        assert abs(result - 2.0) < 0.0001

    def test_stddev_empty_list(self):
        """空リストでは0.0を返す。"""
        from tools.long_term_sim import _safe_stddev
        assert _safe_stddev([]) == 0.0

    def test_stddev_single_value(self):
        """1個の値では0.0を返す。"""
        from tools.long_term_sim import _safe_stddev
        assert _safe_stddev([5.0]) == 0.0

    def test_stddev_identical_values(self):
        """全て同じ値では0.0を返す。"""
        from tools.long_term_sim import _safe_stddev
        assert _safe_stddev([3.0, 3.0, 3.0]) == 0.0

    def test_policy_ratio_sum_is_one(self):
        """ポリシー比率の合計が1.0になる。"""
        result = run_simulation(scenario_name="smoke")
        stats = compute_statistics(result)
        total_ratio = sum(
            v["ratio"] for v in stats["policy_distribution"].values()
        )
        assert abs(total_ratio - 1.0) < 0.01

    def test_min_le_mean_le_max(self):
        """全ての統計量でmin <= mean <= maxが成立する。"""
        result = run_simulation(scenario_name="smoke")
        stats = compute_statistics(result)
        # 感情
        for emo_st in stats["emotions"].values():
            assert emo_st["min"] <= emo_st["mean"] <= emo_st["max"]
        # ムード
        for mood_st in stats["mood"].values():
            assert mood_st["min"] <= mood_st["mean"] <= mood_st["max"]
        # ドライブ
        for drv_st in stats["drives"].values():
            assert drv_st["min"] <= drv_st["mean"] <= drv_st["max"]
        # 恐怖指数
        fl = stats["fear_level"]
        assert fl["min"] <= fl["mean"] <= fl["max"]
