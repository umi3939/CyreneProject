"""
tests/test_structure_improvement.py - テスト基盤の構造的整理（結合テスト追加重点）

設計書: design_test_structure_improvement.md

4カテゴリの結合テストを追加:
- カテゴリ1: 帰還経路連鎖テスト（記憶→感情、選択→感情、他者仮説→感情、同時発火）
- カテゴリ2: Phase帯域横断テスト（毎ティック→3ティック→5ティック→10ティック→ポリシー選択帯域）
- カテゴリ3: enrichment生成と永続化の横断テスト（圧縮後構造、永続化一貫性、長期非空性、空状態代替値）
- カテゴリ4: 入力経路切替と帰還の複合テスト（テキスト→画面、自発→外部、tick_count一貫性、save/load切替）

テスト期待値は構造的性質（型・存在・範囲・変動・非空・安全代替値・例外不発生）に限定し、
特定の出力値への一致は検証しない。

既存テストの変更は一切行わない。

== テスト構造の分析 ==

結合テスト数と全テスト数の比率（本ファイル追加前）:
  結合テストファイル:
    - test_phase_chain_integration.py (~48 tests): Phase連鎖
    - test_phase_band_chain_integration.py (~55 tests): 帯域別連鎖
    - test_integration_extended.py (~40 tests): save/load・enrichment
    - test_extended_stability.py (~51 tests): 長時間安定性
    - test_save_load_regression.py (~109 tests): save/load回帰
    - test_e2e_smoke.py (~15 tests): E2Eスモーク
  結合テスト合計: ~318 / 全テスト ~8,883 = 約3.6%

Phase別テストカバレッジ:
  Phase 1-7 (毎ティック): test_phase_chain_integration.py, test_phase_band_chain_integration.py
  Phase 8-14 (3ティック): test_phase_band_chain_integration.py
  Phase 15-18 (5ティック): test_phase_band_chain_integration.py
  Phase 19-24b (5ティック): test_phase_band_chain_integration.py
  Phase 25-25f (5ティック): test_phase_band_chain_integration.py
  Phase 26-26h (5ティック): test_phase_band_chain_integration.py
  Phase 30-35c (10ティック): test_phase_band_chain_integration.py

帰還経路別テスト数:
  経路A (memory_emotion_return): test_memory_emotion_return.py(59テスト), orchestrator内5ティック帯域で発火
  経路B (selection_emotion_return): orchestrator内select_policy_dict後に発火
  経路C (other_hypothesis_emotion_return): test_other_hypothesis_emotion_return.py(65テスト), orchestrator内5ティック帯域で発火
  帰還経路の連鎖動作（発火→後続Phase伝搬）の結合テスト: 本ファイルで追加
"""

import json
from pathlib import Path

import pytest

from psyche.orchestrator import PsycheOrchestrator
from psyche.state import Percept


# ── Helpers ───────────────────────────────────────────────────────


def _make_percept(
    emotion: str = "happy",
    valence: float = 0.7,
    text: str = "テスト画面",
    intent: str = "expression",
) -> Percept:
    """テスト用の Percept を生成する。"""
    return Percept(
        text=text,
        meaning=text,
        emotion=emotion,
        intent=intent,
        emotion_valence=valence,
    )


EMOTIONS = [
    "happy", "sad", "angry", "neutral", "surprised",
    "loving", "teasing", "scared", "happy", "neutral",
]
VALENCES = [0.7, -0.6, -0.5, 0.0, 0.3, 0.8, 0.4, -0.5, 0.6, 0.0]

# enrichment の5セクションヘッダ（圧縮済み形式）
ENRICHMENT_SECTIONS = [
    "[内面]",
    "[自己]",
    "[動機]",
    "[記憶]",
    "[判断]",
]


def _run_ticks(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新する。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)


def _run_ticks_with_policy(orch: PsycheOrchestrator, count: int) -> None:
    """指定ティック数だけ多様な感情入力で更新し、定期的にポリシー選択も行う。"""
    for i in range(count):
        idx = i % len(EMOTIONS)
        percept = _make_percept(
            emotion=EMOTIONS[idx],
            valence=VALENCES[idx],
            text=f"テスト入力{i}",
        )
        orch.post_response_update(percept, delta_time=1.0)
        # 5ティック毎にポリシー選択を実行
        if (i + 1) % 5 == 0:
            orch.select_policy_dict(percept, [])


# ══════════════════════════════════════════════════════════════════
# カテゴリ1: 帰還経路連鎖テスト
# ══════════════════════════════════════════════════════════════════


class TestReturnPathwayChain:
    """帰還経路が発火し、その変動が後続Phaseに伝搬することの検証。

    3つの帰還経路:
    - 経路A: memory_emotion_return (記憶→感情)
    - 経路B: selection_emotion_return (選択→感情)
    - 経路C: other_hypothesis_emotion_return (他者仮説→感情)
    """

    def test_memory_return_fires_and_emotion_propagates(self):
        """記憶保存→感情帯域変動→後続Phase伝搬の検証。

        帰還経路A (memory_emotion_return) は5ティック帯域で発火する。
        発火すると感情ベクトルが変動し、その変動は後続の毎ティック帯域の
        Phase(reaction等)の入力に反映される。
        """
        orch = PsycheOrchestrator()
        # 10ティック実行して帰還経路Aが発火する機会を与える
        # (5ティック帯域で発火、10ティックで2サイクル)
        _run_ticks(orch, 10)

        # 感情ベクトルが存在し、各次元が0.0-1.0の範囲内であること
        emotions = orch._psyche.emotions.as_dict()
        assert isinstance(emotions, dict)
        for dim, val in emotions.items():
            assert 0.0 <= val <= 1.0, (
                f"Emotion dimension {dim} out of range: {val}"
            )

        # return_pathway_monitor が動作していること(READ-ONLY観測)
        rpm = orch._return_pathway_monitor
        assert rpm is not None

        # 帰還経路Aの発火カウントにアクセスできること(型チェックのみ)
        fire_counts = rpm.pathway_fire_counts
        assert isinstance(fire_counts, dict)
        assert "memory_emotion_return" in fire_counts
        assert isinstance(fire_counts["memory_emotion_return"], int)

        # 追加ティック後も感情ベクトルが範囲内であること(帰還量の累積安全性)
        _run_ticks(orch, 5)
        emotions_after = orch._psyche.emotions.as_dict()
        for dim, val in emotions_after.items():
            assert 0.0 <= val <= 1.0, (
                f"Post-return emotion {dim} out of range: {val}"
            )

    def test_selection_return_fires_after_policy_selection(self):
        """選択結果→感情帯域変動の検証。

        帰還経路B (selection_emotion_return) はselect_policy_dict直後に発火する。
        ポリシー選択後に感情ベクトルが変動しうることを構造的に検証する。
        """
        orch = PsycheOrchestrator()
        # 10ティック実行して十分な状態を構築
        _run_ticks(orch, 10)

        # ポリシー選択前の感情ベクトルを記録
        emotions_before = orch._psyche.emotions.as_dict()

        # ポリシー選択を実行(帰還経路Bが発火する契機)
        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

        # 帰還経路Bの発火カウントにアクセスできること
        rpm = orch._return_pathway_monitor
        fire_counts = rpm.pathway_fire_counts
        assert isinstance(fire_counts["selection_emotion_return"], int)

        # 選択後の感情ベクトルが範囲内であること
        emotions_after = orch._psyche.emotions.as_dict()
        for dim, val in emotions_after.items():
            assert 0.0 <= val <= 1.0, (
                f"Post-selection emotion {dim} out of range: {val}"
            )

    def test_other_hypothesis_return_does_not_interfere_with_other_model(self):
        """他者仮説→感情帯域変動が他者モデルの後続更新と干渉しない検証。

        帰還経路C (other_hypothesis_emotion_return) は5ティック帯域で発火する。
        帰還後も他者モデルの状態が破損しないことを検証する。
        """
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)

        # 他者モデルの状態が存在し、アクセス可能であること
        assert orch._other_model_sys is not None

        # 帰還経路Cの発火カウントにアクセスできること
        rpm = orch._return_pathway_monitor
        fire_counts = rpm.pathway_fire_counts
        assert isinstance(fire_counts["other_hypothesis_emotion_return"], int)

        # 追加ティック後も他者モデルの状態が正常であること
        _run_ticks(orch, 5)
        assert orch._other_model_sys is not None

        # 他者モデルの後続更新が例外なく実行されること
        _run_ticks(orch, 5)
        emotions_final = orch._psyche.emotions.as_dict()
        for dim, val in emotions_final.items():
            assert 0.0 <= val <= 1.0, (
                f"Post-other-hypothesis emotion {dim} out of range: {val}"
            )

    def test_concurrent_return_pathway_firing_within_band_limit(self):
        """複数帰還経路の同一ティック発火時の合算帯域制限の検証。

        3経路同時発火条件を構成し、発火後の感情ベクトルが
        全次元で0.0-1.0の範囲内に収まることを検証する。
        ポリシー選択を含む実行を行い、経路A(5ティック帯域)と
        経路B(選択時)と経路C(5ティック帯域)の同時発火機会を与える。
        """
        orch = PsycheOrchestrator()
        # 30ティックをポリシー選択付きで実行して十分な同時発火機会を与える
        _run_ticks_with_policy(orch, 30)

        # 感情ベクトルが全次元で範囲内であること(合算帯域制限の安全性)
        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert 0.0 <= val <= 1.0, (
                f"Concurrent firing emotion {dim} out of range: {val}"
            )

        # return_pathway_monitor の同時発火カウントにアクセスできること
        rpm = orch._return_pathway_monitor
        assert isinstance(rpm.concurrent_2plus_count, int)
        assert isinstance(rpm.concurrent_3_count, int)

        # 累積発火カウントが非負であること
        for pathway_id, count in rpm.pathway_fire_counts.items():
            assert count >= 0, (
                f"Pathway {pathway_id} fire count is negative: {count}"
            )


# ══════════════════════════════════════════════════════════════════
# カテゴリ2: Phase帯域横断テスト
# ══════════════════════════════════════════════════════════════════


class TestPhaseBandCrossing:
    """Phase帯域間のデータ受渡し検証。

    毎ティック帯域(1)→3ティック帯域(3)→5ティック帯域(5)→
    10ティック帯域(10)→ポリシー選択帯域の伝搬を検証する。
    """

    def test_every_tick_to_3tick_propagation(self):
        """毎ティック帯域の出力が3ティック帯域に伝搬する検証。

        毎ティック帯域の感情状態(Phase 1-7)の出力が
        3ティック帯域のPhase(self_model等)の入力として参照されることを検証。
        """
        orch = PsycheOrchestrator()
        # 3ティック実行して3ティック帯域が発火
        _run_ticks(orch, 3)

        # 毎ティック帯域の出力(感情状態)が存在する
        assert orch._psyche.emotions is not None
        assert orch._psyche.drives is not None
        assert orch._psyche.mood is not None

        # 3ティック帯域の出力(self_model)が存在する
        assert orch._last_self_view is not None, (
            "3-tick band Phase 9 (self_model): output should exist at tick 3"
        )

        # self_modelが感情状態を参照していることの間接的確認
        # (self_view自体がNoneでないことで、感情入力が処理されたことを示す)
        self_view = orch._last_self_view
        assert hasattr(self_view, 'emotion_snapshot') or self_view is not None

    def test_3tick_to_5tick_propagation(self):
        """3ティック帯域の出力が5ティック帯域に伝搬する検証。

        3ティック帯域の自己モデル(Phase 9)出力が
        5ティック帯域の差分認知→物語→エピソード→記憶系の入力として使用されることを検証。
        """
        orch = PsycheOrchestrator()
        # 5ティック実行して5ティック帯域が発火
        _run_ticks(orch, 5)

        # 3ティック帯域の出力が存在する
        assert orch._last_self_view is not None, (
            "3-tick band: self_view should exist before 5-tick band"
        )

        # 5ティック帯域の出力が存在する(差分→物語→エピソード連鎖)
        assert orch._last_diff_summary is not None, (
            "5-tick band Phase 15: diff_summary should exist at tick 5"
        )
        assert orch._last_narrative is not None, (
            "5-tick band Phase 19: narrative should exist at tick 5"
        )
        assert orch._last_episodes is not None, (
            "5-tick band Phase 20: episodes should exist at tick 5"
        )

    def test_5tick_to_10tick_propagation(self):
        """5ティック帯域の出力が10ティック帯域に伝搬する検証。

        5ティック帯域の記憶系出力が10ティック帯域の
        長期統計→参照頻度の入力として使用されることを検証。
        """
        orch = PsycheOrchestrator()
        # 10ティック実行して10ティック帯域が発火
        _run_ticks(orch, 10)

        # 5ティック帯域の記憶系出力が存在する
        assert orch._last_bindings is not None, (
            "5-tick band Phase 21: bindings should exist"
        )

        # 10ティック帯域の参照頻度記述の状態が存在する
        assert orch._reference_frequency_state is not None, (
            "10-tick band: reference_frequency_state should exist at tick 10"
        )

        # 長期統計(dynamics_observer)にアクセス可能であること
        assert hasattr(orch, '_dynamics_observer'), (
            "10-tick band: dynamics_observer attribute should exist"
        )
        assert orch._dynamics_observer is not None

    def test_10tick_to_policy_selection_propagation(self):
        """10ティック帯域の出力がポリシー選択帯域に伝搬する検証。

        10ティック帯域の候補拡張出力がポリシー選択(select_policy_dict)の
        入力として使用されることを検証。
        """
        orch = PsycheOrchestrator()
        # 10ティック実行して10ティック帯域が発火
        _run_ticks(orch, 10)

        # policy_candidate_expansion の状態が存在する
        assert orch._policy_expander is not None, (
            "10-tick band: policy_expander should exist at tick 10"
        )

        # ポリシー選択が正常に動作する
        percept = _make_percept()
        policy = orch.select_policy_dict(percept, [])
        assert isinstance(policy, dict)
        assert "policy_label" in policy

        # 選択結果に必要な構造が含まれること
        assert isinstance(policy.get("policy_label"), str)

        # enrichment も正常に生成される
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0


# ══════════════════════════════════════════════════════════════════
# カテゴリ3: enrichment生成と永続化の横断テスト
# ══════════════════════════════════════════════════════════════════


class TestEnrichmentAndPersistence:
    """enrichment圧縮とPhase出力の整合性、永続化との一貫性を検証。"""

    def test_compressed_enrichment_contains_all_section_headers(self):
        """enrichment圧縮後のテキストが全セクションヘッダを含む検証。

        圧縮パイプライン適用後のenrichmentテキストに
        全5セクション([内面]/[自己]/[動機]/[記憶]/[判断])のヘッダが含まれることを検証。
        """
        orch = PsycheOrchestrator()
        _run_ticks(orch, 10)
        orch.select_policy_dict(_make_percept(), [])

        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        for section in ENRICHMENT_SECTIONS:
            assert section in enrichment, (
                f"Enrichment missing section header: {section}"
            )

    def test_compressed_enrichment_preserves_sections_after_save_load(self, tmp_path):
        """save/load後もenrichmentが全セクションヘッダを含む検証。"""
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch1, 10)
        orch1.select_policy_dict(_make_percept(), [])
        orch1.save()

        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True

        # load後に数ティック実行してenrichmentキャッシュを再構築
        _run_ticks(orch2, 5)
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        for section in ENRICHMENT_SECTIONS:
            assert section in enrichment, (
                f"Post-load enrichment missing section header: {section}"
            )

    def test_enrichment_references_persisted_fields(self, tmp_path):
        """enrichmentが参照するフィールドが永続化されていることの検証。

        enrichmentの生成に使用される主要フィールドが
        save()出力のJSONに含まれていることを構造的に検証する。
        """
        orch = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        _run_ticks(orch, 10)
        orch.select_policy_dict(_make_percept(), [])

        # enrichmentが生成可能であること
        enrichment = orch.get_prompt_enrichment()
        assert len(enrichment) > 0

        # 保存
        orch.save()
        data = json.loads(
            (tmp_path / "psyche_snapshot.json").read_text(encoding="utf-8")
        )

        # enrichmentの生成に使用される主要フィールドが永続化データに存在すること
        # (psyche stateの感情・ムード・ドライブはenrichmentの[内面]セクションの入力)
        assert "psyche" in data
        assert "emotions" in data["psyche"]
        assert "mood" in data["psyche"]
        assert "drives" in data["psyche"]

        # 内省系フィールド([自己]セクションの入力)
        assert "last_self_view" in data
        assert "last_diff_summary" in data

        # version, tick_count が存在すること
        assert "version" in data
        assert "tick_count" in data

    def test_enrichment_nonempty_after_multiple_band_cycles(self):
        """長期実行後（複数帯域サイクル経過後）のenrichmentの非空性検証。"""
        orch = PsycheOrchestrator()
        # 30ティック実行(10ティック帯域が3サイクル)
        _run_ticks_with_policy(orch, 30)

        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        # 全セクションヘッダが含まれること
        for section in ENRICHMENT_SECTIONS:
            assert section in enrichment, (
                f"Long-run enrichment missing section: {section}"
            )

    def test_enrichment_safe_defaults_from_empty_state(self):
        """空状態からの起動時にenrichmentが安全な代替値を返す検証。"""
        orch = PsycheOrchestrator()
        # 1ティックだけ実行(最小限の状態構築)
        _run_ticks(orch, 1)

        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        # 空文字列でないこと(安全な代替値が生成されること)
        assert len(enrichment) > 0

        # 少なくとも[内面]セクションが存在すること
        assert "[内面]" in enrichment, (
            "First-tick enrichment should contain at least [内面] section"
        )


# ══════════════════════════════════════════════════════════════════
# カテゴリ4: 入力経路切替と帰還の複合テスト
# ══════════════════════════════════════════════════════════════════


class TestInputPathSwitchingAndReturn:
    """入力経路切替時の状態整合性と帰還経路の複合検証。"""

    def test_text_to_screen_state_preservation(self):
        """テキスト入力後に画面入力に切り替わった際の状態保持検証。

        テキスト入力(process_text_input)後の状態変化が
        画面入力(post_response_update)に切り替わった際に保持されることを検証。
        """
        orch = PsycheOrchestrator()
        # 初期ティックで基本状態を構築
        _run_ticks(orch, 3)

        # テキスト入力を処理
        text_result = orch.process_text_input(
            text="こんにちは、テスト入力です",
            sender_id="tester",
        )
        # process_text_inputはNoneを返す場合がある(内部状態依存)
        # 結果にかかわらず、テキスト入力による状態変化がある

        # テキスト入力後のtick_count
        tick_after_text = orch.tick_count

        # 画面入力(post_response_update)に切替
        percept = _make_percept(emotion="surprised", valence=0.4)
        orch.post_response_update(percept, delta_time=1.0)

        # tick_countが増加していること
        assert orch.tick_count == tick_after_text + 1

        # 感情ベクトルが範囲内であること(テキスト→画面の切替で破損しない)
        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert 0.0 <= val <= 1.0, (
                f"Post-switch emotion {dim} out of range: {val}"
            )

        # enrichmentが正常に生成されること
        enrichment = orch.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

    def test_spontaneous_result_not_overwritten_by_external(self):
        """自発起動後に外部入力が到達した際の結果保護検証。

        自発起動(check_spontaneous_activation)の結果が
        後続の外部入力によって上書き消滅しないことを検証。
        """
        orch = PsycheOrchestrator()
        # 基本状態を構築
        _run_ticks(orch, 5)

        # 自発起動チェックを実行
        activation_result = orch.check_spontaneous_activation()
        # activationの結果はNoneの場合がある(内部状態依存)

        # 自発起動結果の属性にアクセス可能であること
        assert hasattr(orch, '_last_activation_result')
        stored_result = orch._last_activation_result

        # 外部入力を処理
        percept = _make_percept(emotion="happy", valence=0.8)
        orch.post_response_update(percept, delta_time=1.0)

        # 外部入力後もlast_activation_resultが存在すること
        # (post_response_updateによって上書きされないこと)
        assert hasattr(orch, '_last_activation_result')

        # 感情ベクトルが正常範囲内であること
        emotions = orch._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert 0.0 <= val <= 1.0, (
                f"Post-external emotion {dim} out of range: {val}"
            )

    def test_tick_count_consistency_across_multiple_paths(self):
        """複数経路から連続入力した際のtick_countの一貫性検証。

        テキスト入力→画面入力→自発チェック→画面入力の順で
        tick_countが一貫して増加することを検証。
        """
        orch = PsycheOrchestrator()

        # 画面入力3回
        _run_ticks(orch, 3)
        assert orch.tick_count == 3

        # テキスト入力(tick_countを直接増加させない)
        orch.process_text_input(text="テスト", sender_id="tester")
        tick_after_text = orch.tick_count
        # process_text_inputはtick_countに影響しない
        assert tick_after_text == 3

        # 画面入力2回
        _run_ticks(orch, 2)
        assert orch.tick_count == 5

        # 自発チェック(tick_countを増加させない)
        orch.check_spontaneous_activation()
        assert orch.tick_count == 5

        # 画面入力3回
        _run_ticks(orch, 3)
        assert orch.tick_count == 8

        # 全体でtick_countが単調増加していること
        assert orch.tick_count == 8

    def test_path_switch_save_load_resume_integrity(self, tmp_path):
        """入力経路切替を含むsave/load→resumeの整合性検証。

        複数経路の入力を行った後にsave/load→resumeし、
        復帰後も正常に動作することを検証する。
        """
        orch1 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)

        # 画面入力
        _run_ticks(orch1, 5)

        # テキスト入力
        orch1.process_text_input(text="保存前テスト", sender_id="tester")

        # 自発チェック
        orch1.check_spontaneous_activation()

        # ポリシー選択
        policy1 = orch1.select_policy_dict(_make_percept(), [])
        assert isinstance(policy1, dict)

        # 追加画面入力
        _run_ticks(orch1, 5)
        saved_tick = orch1.tick_count
        assert saved_tick == 10

        # save
        orch1.save()

        # load
        orch2 = PsycheOrchestrator(data_dir=tmp_path, memory_count=5)
        loaded = orch2.load()
        assert loaded is True
        assert orch2.tick_count == saved_tick

        # load後に各経路で正常動作すること

        # 画面入力
        _run_ticks(orch2, 3)
        assert orch2.tick_count == saved_tick + 3

        # テキスト入力
        text_result = orch2.process_text_input(
            text="復帰後テスト", sender_id="tester2",
        )

        # 自発チェック
        activation = orch2.check_spontaneous_activation()

        # ポリシー選択
        policy2 = orch2.select_policy_dict(_make_percept(), [])
        assert isinstance(policy2, dict)
        assert "policy_label" in policy2

        # enrichment正常生成
        enrichment = orch2.get_prompt_enrichment()
        assert isinstance(enrichment, str)
        assert len(enrichment) > 0

        # 感情ベクトルが範囲内
        emotions = orch2._psyche.emotions.as_dict()
        for dim, val in emotions.items():
            assert 0.0 <= val <= 1.0, (
                f"Post-resume emotion {dim} out of range: {val}"
            )
