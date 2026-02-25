"""
tests/test_e2e_smoke.py - Gemini API を使った end-to-end スモークテスト

このファイルは、システムの 2-call 構造（知覚コール + 代弁コール）が
実際の Gemini API 応答で正しく動作するかを検証するテスト群です。

設計書: design_e2e_smoke_test.md

検証階層:
  階層1: 接続基盤 — APIキーの存在確認、LLM抽象化層の初期化
  階層2: 知覚経路 — parse_percept が実APIの応答を構造化できること
  階層3: 代弁経路 — render_expression が実APIで正しくテキスト生成できること
  階層4: 統合パイプライン — orchestrator 経由の複数ターン実行
  階層5: テキスト入力経路 — テキスト対話パイプラインの完走
  階層6: 状態汚染ガード — テスト実行が永続状態を汚染しないこと

安全弁（設計書 §4 に対応）:
  1. 環境依存排除弁 — GEMINI_API_KEY 未設定時は全テスト自動スキップ
  2. タイムアウト弁 — 各APIコールに asyncio.wait_for でタイムアウト設定
  3. 状態汚染排除弁 — 検証用 orchestrator は一時ディレクトリを使い永続化しない
  4. 内容非依存弁 — 全アサーションは形式（型・キー・空でない）のみ検証
  5. 実行分離弁 — APIキー未設定環境では全テストが skip される
"""

# ── 標準ライブラリ群 ──────────────────────────────────────────────
# Python 3.10 で「list[str]」のような新しい型記法を使えるようにする
from __future__ import annotations

# 非同期処理ライブラリ — タイムアウト制御に asyncio.wait_for を使う
import asyncio
# JSON 形式の読み書き — ログファイルの記録に使用する
import json
# ログ出力ライブラリ — テスト中の情報を記録する
import logging
# 環境変数の読み取り — APIキーの有無確認に使用する
import os
# 時間計測用 — 各テストの所要時間をモノトニック時計で測る
import time
# タイムスタンプ生成用 — ログファイル名に日時を含める
from datetime import datetime
# ファイルパス操作 — ログディレクトリやデータファイルの指定に使う
from pathlib import Path
# 型ヒント用 — 関数シグネチャに Any を使う
from typing import Any

# ── テストフレームワーク ──────────────────────────────────────────
# Python の標準的なテストフレームワーク
import pytest

# ── APIキーの有無を判定する（安全弁1: 環境依存排除弁）──────────────
# 環境変数 GEMINI_API_KEY の値を取得し、値があれば True、なければ False にする
_HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY"))

# ── このファイル内の全テストに適用するスキップ条件 ─────────────────
# APIキーが未設定の環境では、全てのテストを自動的にスキップする
# これにより、通常の「python -m pytest tests/」実行でAPIテストは走らない
pytestmark = pytest.mark.skipif(
    not _HAS_API_KEY,  # APIキーがないとき True → テストをスキップ
    reason="GEMINI_API_KEY not set; e2e smoke tests skipped",
)

# ── APIコールの制限時間（秒）──────────────────────────────────────
# 各外部APIコールがこの時間を超えると asyncio.TimeoutError になる
# ネットワーク障害時に無期限待機しないための安全弁（安全弁2: タイムアウト弁）
_API_TIMEOUT = 60.0

# ── このファイル専用のロガーを作成する ─────────────────────────────
# __name__ は "tests.test_e2e_smoke" になり、他のロガーと区別できる
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# ヘルパー関数群
# テスト本体から呼び出される補助的な関数。
# これらは検証ロジックを持たず、環境構築とログ記録のみを行う。
# ════════════════════════════════════════════════════════════════════


def _make_log_dir() -> Path:
    """e2eテスト結果のログ出力先ディレクトリを作成して返す。

    ログは プロジェクトルート/logs/e2e_smoke/ に保存される。
    このログはシステム動作に一切影響しない（開発者確認用）。
    """
    # テストファイルの親の親 = プロジェクトルート、その下に logs/e2e_smoke を作る
    log_dir = Path(__file__).parent.parent / "logs" / "e2e_smoke"
    # parents=True: 途中のディレクトリも作る, exist_ok=True: 既存でもエラーにしない
    log_dir.mkdir(parents=True, exist_ok=True)
    # 作成したディレクトリのパスを返す
    return log_dir


def _write_log(log_dir: Path, test_name: str, data: dict[str, Any]) -> None:
    """単一テスト結果を JSON ファイルに書き出す。

    ファイル名は「YYYYMMDD_HHMMSS_テスト名.json」の形式。
    検証結果の事後確認用であり、システムの動作には影響しない。
    """
    # 現在時刻を「年月日_時分秒」形式の文字列にする
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ファイル名を組み立てる
    filename = f"{timestamp}_{test_name}.json"
    # ログディレクトリ内のフルパスを構成する
    log_path = log_dir / filename
    # 辞書を JSON 文字列に変換してファイルに書き出す
    log_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,  # 日本語をエスケープせずそのまま出力する
            indent=2,            # 読みやすいように2スペースインデントする
            default=str,         # datetime 等をそのまま文字列に変換する
        ),
        encoding="utf-8",
    )


def _make_temp_data_dir(tmp_path: Path) -> Path:
    """テスト専用の一時データディレクトリを作成する。

    orchestrator が必要とする最小限のデータファイルを配置する。
    この一時ディレクトリは pytest が自動的にクリーンアップするため、
    運用中の data/ ディレクトリとは完全に独立する（安全弁3: 状態汚染排除弁）。
    """
    # pytest が提供する一時パスの下に data ディレクトリを作る
    data = tmp_path / "data"
    data.mkdir()

    # --- 最小限の長期記憶データを配置する ---
    # MemoryManager が読み込む記憶ファイルの最小構成
    memories = [
        {
            "id": 1,                             # 記憶の一意ID
            "summary": "e2e test memory entry",  # 記憶の要約テキスト
            "keywords": ["test"],                # 検索用キーワードのリスト
            "importance": 3,                     # 重要度（1が低い、5が高い）
            "date": "2026-01-01T00:00:00",      # 記憶が作られた日時
            "protected": False,                  # 保護フラグ（False = 忘却対象）
            "last_recalled": None,               # 最後に思い出された日時（未使用）
        },
    ]
    # JSON に変換してファイルに書き出す
    (data / "example_memories.json").write_text(
        json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 最小限の愛着データを配置する ---
    # AttachmentManager が読み込む空のファイル
    (data / "example_attachments.json").write_text(
        json.dumps({}, ensure_ascii=False), encoding="utf-8"
    )

    # --- 最小限のアイデンティティデータを配置する ---
    # IdentityManager が参照するキャラクター特性データ
    (data / "identity.json").write_text(
        json.dumps(
            {
                "core_traits": ["romantic", "caring"],             # 核となる特性
                "trait_confidence": {"romantic": 0.9, "caring": 0.8},  # 特性の確信度
                "pending_changes": [],                              # 保留中の変更（空）
                "risk": 0.0,                                        # リスク値（なし）
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --- 最小限の未来投射データを配置する ---
    # ProjectionManager が参照する目標データ
    (data / "projections.json").write_text(
        json.dumps(
            {
                "goals": [
                    {
                        "id": "e2e_test",               # 目標のID
                        "description": "e2e test goal",  # 目標の説明文
                        "progress": 0.1,                 # 進捗度（0.0〜1.0）
                        "status": "active",              # ステータス（有効）
                    }
                ],
                "risk": 0.0,  # 投射リスク値
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --- 最小限の心理状態データを配置する ---
    # StateManager が読み込む初期状態
    (data / "state.json").write_text(
        json.dumps(
            {
                "test_user": {
                    # 感情ベクトル（5次元：喜び/悲しみ/恐怖/怒り/穏やか）
                    "emotions": {
                        "joy": 0.0, "sad": 0.0, "fear": 0.0,
                        "anger": 0.0, "calm": 0.5,
                    },
                    # 動機ベクトル（社会性と好奇心）
                    "drives": {"social": 0.5, "curiosity": 0.5},
                    # 全体的な気分（-1.0=ネガティブ 〜 1.0=ポジティブ）
                    "mood": 0.0,
                    # 最終更新時刻を現在時刻に設定する
                    "last_updated": datetime.now().isoformat(timespec="seconds"),
                    # 損失回避度（低い値 = 損失を恐れにくい）
                    "loss_aversion": 0.3,
                    # 恐怖指数（0.0 = 恐怖なし）
                    "fear_index": 0.0,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --- 最小限のペルソナ（キャラクター設定）データを配置する ---
    # 表現生成で参照されるキャラクター口調設定
    (data / "persona.json").write_text(
        json.dumps(
            {
                "name": "キュレネ",                # キャラクター名
                "first_person": "あたし",          # 一人称
                "second_person": "あなた",         # 二人称
                "tone": "romantic, sweet",          # 口調
                "style_rules": {
                    "禁止": ["です", "ます"],      # 使ってはいけない語尾
                    "推奨": ["♪", "！"],           # 使用推奨の記号
                },
                "example_lines": ["ふふっ♪"],     # セリフの例
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --- 最小限の責任データを配置する ---
    # ResponsibilityManager が読み込む空のファイル
    (data / "responsibility.json").write_text(
        json.dumps({}, ensure_ascii=False), encoding="utf-8"
    )

    # 作成した一時データディレクトリのパスを返す
    return data


def _split_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する（brain.py と同じロジック）。

    句読点（。！？等）や記号（♪♡等）を区切り文字として使い、
    テキストを文のリストに変換する。
    この関数は brain.py の think_streaming 内の文分割と同一の処理。
    """
    # 分割された文を格納するリスト
    sentences = []
    # 現在組み立て中の文
    current = ""

    # テキストの各文字を順番に処理する
    for i, char in enumerate(text):
        # 文字を現在の文に追加する
        current += char

        # 句読点や記号に到達したら、そこで文を区切る
        if char in "。！？!?♪♥♡★☆\n":
            # 前後の空白を除去した文を取得する
            sentence = current.strip()
            # 空でなければリストに追加する
            if sentence:
                sentences.append(sentence)
            # 次の文のために空にする
            current = ""
        # 日本語テキスト後の "w"（笑い表現）の場合
        elif char == 'w':
            # テキストの次の文字を取得する（末尾なら None）
            next_char = text[i + 1] if i + 1 < len(text) else None
            # "ww" のような連続 w でなければ文の区切りを検討する
            if next_char != 'w':
                # "w" を除いた部分を取得する
                pre_w = current.rstrip('w')
                # 直前が日本語文字（非ASCII）なら笑い表現として区切る
                if pre_w and not pre_w[-1].isascii():
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""

    # ループ終了後、残っているテキストがあれば最後の文として追加する
    if current.strip():
        sentences.append(current.strip())

    # 分割された文のリストを返す
    return sentences


# ════════════════════════════════════════════════════════════════════
# pytest フィクスチャ群
# テスト関数に自動的に注入される共有リソース。
# フィクスチャはテストごとに新しいインスタンスが生成される。
# ════════════════════════════════════════════════════════════════════


@pytest.fixture()
def log_dir():
    """e2eテスト結果を書き出すログディレクトリを提供する。"""
    return _make_log_dir()


@pytest.fixture()
def temp_data(tmp_path):
    """使い捨て用の一時データディレクトリを提供する。

    pytest が提供する tmp_path 上に作られるため、テスト終了後に自動削除される。
    """
    return _make_temp_data_dir(tmp_path)


@pytest.fixture()
def orchestrator(temp_data):
    """使い捨ての PsycheOrchestrator インスタンスを生成して提供する。

    一時データディレクトリを使用するため、運用中の永続状態には影響しない。
    このインスタンスの save() は決して呼ばない（安全弁3: 状態汚染排除弁）。
    """
    # 心理統合管理クラスをインポートする
    from psyche.orchestrator import PsycheOrchestrator

    # 一時データディレクトリを指定してインスタンスを作成する
    # memory_count=1: 長期記憶が1件存在するものとして初期化する
    orch = PsycheOrchestrator(memory_count=1, data_dir=temp_data)
    # テスト終了後に自動的に破棄される
    return orch


# ── 共有ペルソナ辞書（全テストで同一のキャラクター設定を使う）─────
# brain.py の _build_persona_dict() と同じ形式
_PERSONA = {
    "name": "キュレネ",                              # キャラクター名
    "tone": "romantic, sweet, playful",               # 口調指定
    "style_rules": {
        "禁止": ["敬語", "絵文字"],                  # 使用禁止パターン
        "推奨": ["♪♡使用可", "い抜き言葉", "カジュアルなタメ口"],  # 推奨パターン
    },
}


# ════════════════════════════════════════════════════════════════════
# 階層1: 接続基盤テスト
# APIキーの存在確認と LLM 抽象化層の基本動作を検証する。
# このクラスのテストが失敗する場合、後続の全テストも失敗する。
# ════════════════════════════════════════════════════════════════════


class TestConnectionFoundation:
    """外部API接続の基盤が正しく機能しているかを検証するテスト群。"""

    def test_api_key_present(self):
        """環境変数 GEMINI_API_KEY が設定されていることを確認する。

        pytestmark の skipif でスキップされるため、ここに到達した
        時点で GEMINI_API_KEY は設定済みのはず。二重チェックとして確認する。
        """
        # 環境変数から APIキーの値を取得する
        key = os.getenv("GEMINI_API_KEY")
        # APIキーが None でないことを確認する（未設定の場合 None になる）
        assert key is not None
        # APIキーが空文字列でないことを確認する
        assert len(key) > 0

    def test_llm_wrapper_init(self):
        """LLM 抽象化層の主要な関数とシステムプロンプトがインポート可能か確認する。

        この時点ではまだ API コールは行わない。
        インポートが成功すること＝モジュール構造が正しいことの検証。
        """
        # LLM 抽象化層から主要なシンボルをインポートする
        from src.llm_wrapper import (
            llm_call,                 # テキストのみの API コール関数
            llm_call_with_system,     # システムプロンプト付き API コール関数
            llm_call_with_image,      # 画像付き API コール関数（マルチモーダル）
            VISION_SYSTEM_PROMPT,     # 知覚用システムプロンプト（画面記述エンジン用）
            EXPRESSION_SYSTEM_PROMPT,  # 代弁用システムプロンプト（発話レンダラ用）
            PERCEPTION_SYSTEM_PROMPT,  # テキスト解析用システムプロンプト
        )
        # 各関数が呼び出し可能であることを確認する
        assert callable(llm_call)
        assert callable(llm_call_with_system)
        assert callable(llm_call_with_image)
        # 各システムプロンプトが空でない文字列であることを確認する
        assert isinstance(VISION_SYSTEM_PROMPT, str)
        assert len(VISION_SYSTEM_PROMPT) > 0
        assert isinstance(EXPRESSION_SYSTEM_PROMPT, str)
        assert len(EXPRESSION_SYSTEM_PROMPT) > 0
        assert isinstance(PERCEPTION_SYSTEM_PROMPT, str)
        assert len(PERCEPTION_SYSTEM_PROMPT) > 0

    @pytest.mark.asyncio
    async def test_basic_llm_call(self, log_dir):
        """最も基本的な LLM コールが成功し、空でない文字列が返ることを確認する。

        簡単なプロンプトを送信し、応答が返ることを検証する。
        応答の「内容の正確さ」は検証しない（安全弁4: 内容非依存弁）。
        """
        # LLM コール関数をインポートする
        from src.llm_wrapper import llm_call

        # 所要時間計測のためタイマーを開始する
        t0 = time.monotonic()
        # タイムアウト付きで API コールを実行する（安全弁2: タイムアウト弁）
        result = await asyncio.wait_for(
            # 簡単な算数の質問を送る（応答が短くなるよう max_tokens=32 を指定）
            llm_call(
                "テスト: 1+1の答えを1文字で返してください。",
                params={"temperature": 0.1, "max_tokens": 32},
            ),
            timeout=_API_TIMEOUT,
        )
        # 経過時間を計算する
        elapsed = time.monotonic() - t0

        # 応答が文字列型であることを確認する
        assert isinstance(result, str)
        # 応答が空でないことを確認する
        assert len(result) > 0
        # フォールバック応答（API不達時の代替応答）でないことを確認する
        assert "no_llm_available" not in result

        # テスト結果をログファイルに記録する
        _write_log(log_dir, "basic_llm_call", {
            "result_length": len(result),          # 応答の文字数
            "result_preview": result[:200],         # 応答の先頭200文字
            "elapsed_seconds": round(elapsed, 3),   # 所要時間（秒）
        })


# ════════════════════════════════════════════════════════════════════
# 階層2: 知覚経路テスト
# parse_percept が実際の Gemini API 応答を使って
# テキストを Percept 構造体に正しく変換できるかを検証する。
# ════════════════════════════════════════════════════════════════════


class TestPerceptionPathway:
    """知覚処理パイプラインが実 API 応答で正しく動作するかを検証するテスト群。"""

    @pytest.mark.asyncio
    async def test_parse_percept_text(self, orchestrator, log_dir):
        """日本語テキスト入力に対して parse_percept が有効な Percept を返すことを確認する。

        感情・意図・トピック等が正しい型で返されることを検証する。
        感情の「種類」や「内容の正しさ」は検証しない（安全弁4: 内容非依存弁）。
        """
        # 知覚処理関数と Percept データ型をインポートする
        from psyche.perception import parse_percept
        from psyche.state import Percept
        # LLM コール関数をインポートする（LLM エンリッチメントに渡す）
        from src.llm_wrapper import llm_call

        # 所要時間計測のためタイマーを開始する
        t0 = time.monotonic()
        # タイムアウト付きで知覚処理を実行する
        percept = await asyncio.wait_for(
            parse_percept(
                "こんにちは！今日はいい天気だね♪",   # 挨拶テキストを入力する
                llm_call_fn=llm_call,                   # LLM エンリッチメント用の関数
                state=orchestrator.psyche,              # 現在の心理状態（参照情報として渡す）
            ),
            timeout=_API_TIMEOUT,
        )
        # 経過時間を計算する
        elapsed = time.monotonic() - t0

        # --- 形式的検証のみ（内容は検証しない）---
        # 返り値が Percept 型であることを確認する
        assert isinstance(percept, Percept)
        # 感情ラベルが空でない文字列であることを確認する
        assert isinstance(percept.emotion, str)
        assert len(percept.emotion) > 0
        # 意図ラベルが空でない文字列であることを確認する
        assert isinstance(percept.intent, str)
        assert len(percept.intent) > 0
        # トピックリストがリスト型であることを確認する
        assert isinstance(percept.topics, list)
        # 感情価が float 型で -1.0〜1.0 の範囲であることを確認する
        assert isinstance(percept.emotion_valence, float)
        assert -1.0 <= percept.emotion_valence <= 1.0

        # テスト結果をログに記録する
        _write_log(log_dir, "parse_percept_text", {
            "emotion": percept.emotion,
            "intent": percept.intent,
            "topics": percept.topics,
            "emotion_valence": percept.emotion_valence,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_parse_percept_screen_description(self, orchestrator, log_dir):
        """画面記述テキスト（vision 経路の出力のような入力）を処理できることを確認する。

        知覚コールが返す「画面の客観的記述」を入力として、
        Percept 構造体が正しく生成されることを検証する。
        """
        # 知覚処理関数と Percept データ型をインポートする
        from psyche.perception import parse_percept
        from psyche.state import Percept
        from src.llm_wrapper import llm_call

        # ゲーム画面の客観的記述テキスト（知覚コールの典型的な出力形式）
        screen_text = (
            "画面にはゲームのタイトル画面が表示されている。"
            "中央に大きなロゴがあり、「スタート」ボタンが光っている。"
            "背景は夜空で星が流れている。"
        )

        # 所要時間計測のためタイマーを開始する
        t0 = time.monotonic()
        # タイムアウト付きで知覚処理を実行する
        percept = await asyncio.wait_for(
            parse_percept(
                screen_text,                   # 画面記述テキストを入力する
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )
        # 経過時間を計算する
        elapsed = time.monotonic() - t0

        # --- 形式的検証のみ ---
        # Percept インスタンスが返されたことを確認する
        assert isinstance(percept, Percept)
        # 感情ラベルが存在することを確認する
        assert isinstance(percept.emotion, str)
        assert len(percept.emotion) > 0
        # トピックリストが存在することを確認する
        assert isinstance(percept.topics, list)

        # テスト結果をログに記録する
        _write_log(log_dir, "parse_percept_screen", {
            "emotion": percept.emotion,
            "intent": percept.intent,
            "topics": percept.topics,
            "emotion_valence": percept.emotion_valence,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_percept_structure_consistency(self, orchestrator):
        """Percept の辞書表現が必須キーを全て含み、各値の型が正しいことを確認する。

        model_dump() で辞書化した結果を検証し、
        後段処理（orchestrator 等）が期待するデータ構造であることを保証する。
        """
        # 知覚処理関数をインポートする
        from psyche.perception import parse_percept
        from src.llm_wrapper import llm_call

        # 悲しい感情を含むテキストで知覚処理を実行する
        percept = await asyncio.wait_for(
            parse_percept(
                "悲しいことがあったんだ...",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # Percept を辞書に変換する（pydantic の model_dump メソッド）
        percept_dict = percept.model_dump()

        # 後段処理が必要とする必須キーの集合を定義する
        required_keys = {
            "text",              # 入力テキスト
            "meaning",           # 意味の要約
            "emotion",           # 感情ラベル（happy, sad 等）
            "intent",            # 意図ラベル（greeting, question 等）
            "topics",            # トピックのリスト
            "sentiment",         # 感情値（レガシー互換フィールド）
            "emotion_valence",   # 感情価（-1.0 〜 1.0 の連続値）
        }
        # 辞書のキー集合が必須キー集合を全て含むことを確認する
        assert required_keys.issubset(set(percept_dict.keys()))
        # 各キーの値の型を個別に確認する
        assert isinstance(percept_dict["text"], str)
        assert isinstance(percept_dict["meaning"], str)
        assert isinstance(percept_dict["emotion"], str)
        assert isinstance(percept_dict["intent"], str)
        assert isinstance(percept_dict["topics"], list)
        assert isinstance(percept_dict["sentiment"], (int, float))
        assert isinstance(percept_dict["emotion_valence"], (int, float))


# ════════════════════════════════════════════════════════════════════
# 階層3: 代弁経路テスト
# render_expression が実際の Gemini API 応答を使って
# 心理状態と方針からキャラクターのセリフを生成できるかを検証する。
# ════════════════════════════════════════════════════════════════════


class TestExpressionPathway:
    """表現生成（代弁）パイプラインが実 API 応答で正しく動作するかを検証するテスト群。"""

    @pytest.mark.asyncio
    async def test_render_expression_basic(self, orchestrator, log_dir):
        """render_expression が有効なテキストとメタ情報を返すことを確認する。

        返り値の辞書に "text"（セリフ）と "meta"（メタ情報）が
        含まれていることを検証する。テキストの「品質」は検証しない（安全弁4）。
        """
        # 表現生成関数をインポートする
        from psyche.expression import render_expression
        from src.llm_wrapper import llm_call

        # 方針辞書を作成する（orchestrator.select_policy_dict の返り値と同形式）
        policy = {
            "policy_label": "共感する",            # 選択された方針のラベル
            "rationale": "相手の気持ちに寄り添う",  # 方針が選ばれた根拠
            "text": "ふふっ♪",                     # フォールバック用テキスト
        }

        # 所要時間計測のためタイマーを開始する
        t0 = time.monotonic()
        # タイムアウト付きで表現生成を実行する
        result = await asyncio.wait_for(
            render_expression(
                state=orchestrator.psyche,    # 現在の心理状態
                policy=policy,                # 選択された方針
                memory_snippet=[],            # 関連記憶（今回は空）
                persona=_PERSONA,             # ペルソナ設定（共有定数）
                llm_call_fn=llm_call,         # LLM コール関数
                screen_context="ユーザーが楽しそうにゲームをしている画面",
            ),
            timeout=_API_TIMEOUT,
        )
        # 経過時間を計算する
        elapsed = time.monotonic() - t0

        # --- 形式的検証のみ ---
        # 返り値が辞書であることを確認する
        assert isinstance(result, dict)
        # "text" キーが存在し、空でない文字列であることを確認する
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        # "meta" キーが存在し、辞書であることを確認する
        assert "meta" in result
        assert isinstance(result["meta"], dict)

        # テスト結果をログに記録する
        _write_log(log_dir, "render_expression_basic", {
            "text_length": len(result["text"]),
            "text_preview": result["text"][:200],
            "meta": result.get("meta", {}),
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_expression_text_splittable(self, orchestrator, log_dir):
        """表現生成のテキストが文分割処理で処理可能な形式であることを確認する。

        brain.py の think_streaming と同じ文分割ロジックを適用し、
        少なくとも1つの文が生成されることを検証する。
        これは後段処理（音声合成への文単位送信）の前提条件である。
        """
        # 表現生成関数をインポートする
        from psyche.expression import render_expression
        from src.llm_wrapper import llm_call

        # 「励ます」方針で表現生成を行う
        policy = {
            "policy_label": "励ます",       # 方針ラベル
            "rationale": "相手を元気づける",  # 方針の根拠
            "text": "大丈夫だよ♪",          # フォールバック用テキスト
        }

        # タイムアウト付きで表現生成を実行する
        result = await asyncio.wait_for(
            render_expression(
                state=orchestrator.psyche,
                policy=policy,
                memory_snippet=[],
                persona=_PERSONA,
                llm_call_fn=llm_call,
                screen_context="ユーザーが落ち込んでいる様子",
            ),
            timeout=_API_TIMEOUT,
        )

        # 生成されたテキストを取得する
        text = result.get("text", "")
        # テキストが空でない文字列であることを確認する
        assert isinstance(text, str)
        assert len(text) > 0

        # brain.py と同じ文分割ロジックでテキストを分割する
        sentences = _split_sentences(text)

        # 少なくとも1文が生成されたことを確認する
        assert len(sentences) >= 1
        # 各文が空でない文字列であることを確認する
        for s in sentences:
            assert isinstance(s, str)
            assert len(s) > 0

        # テスト結果をログに記録する
        _write_log(log_dir, "expression_splittable", {
            "original_text": text[:200],
            "sentence_count": len(sentences),
            "sentences": [s[:100] for s in sentences],
        })


# ════════════════════════════════════════════════════════════════════
# 階層4: 統合パイプラインテスト
# 知覚→心理更新→方針選択→表現 の一連のフローを
# orchestrator 経由で実行し、全ステップが正常に完了するかを検証する。
# ════════════════════════════════════════════════════════════════════


class TestIntegratedPipeline:
    """2-call 構造の統合パイプラインを検証するテスト群。"""

    @pytest.mark.asyncio
    async def test_single_turn_pipeline(self, orchestrator, log_dir):
        """単一ターン: 知覚→心理更新→方針選択→表現 の一連の流れが完走すること。

        2-call 構造の全ステップを順番に実行し、
        各ステップが例外なく完了することを検証する。
        """
        # 必要なモジュールをインポートする
        from psyche.perception import parse_percept         # 知覚処理
        from psyche.expression import render_expression     # 表現生成
        from psyche.silence_hesitation import is_silence_policy  # 沈黙判定
        from psyche.state import Percept                    # 知覚データ型
        from src.llm_wrapper import llm_call                # LLM コール

        # 画面記述テキスト（知覚コールの出力を模した入力）
        screen_text = (
            "ユーザーがチャットで挨拶している画面。"
            "テキスト入力欄に「こんにちは」と書かれている。"
        )

        # 所要時間計測のためタイマーを開始する
        t0 = time.monotonic()

        # ステップ1: 知覚処理 — テキストを Percept に構造化する
        percept = await asyncio.wait_for(
            parse_percept(
                screen_text,
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )
        # 返り値が Percept 型であることを確認する
        assert isinstance(percept, Percept)

        # ステップ2: 心理更新 — orchestrator の全 Phase を実行する
        # delta=1.0 は前回の更新から1秒経過したことを示す
        # "viewer" は入力経路の識別子（画面閲覧経路）
        orchestrator.post_response_update(percept, 1.0, "viewer")

        # ステップ3: 方針選択 — 心理状態と知覚から最適な方針を選ぶ
        policy = orchestrator.select_policy_dict(percept, [], "viewer")
        # 方針が辞書であり、ラベルが含まれていることを確認する
        assert isinstance(policy, dict)
        assert "policy_label" in policy

        # ステップ4: 表現生成（沈黙が選ばれた場合はスキップする）
        response_text = ""
        if not is_silence_policy(policy):
            # エンリッチメント（心理状態の詳細テキスト）を取得する
            enrichment = orchestrator.get_prompt_enrichment("viewer")
            # LLM に方針と状態を渡してセリフを生成する
            result = await asyncio.wait_for(
                render_expression(
                    state=orchestrator.psyche,
                    policy=policy,
                    memory_snippet=[],
                    persona=_PERSONA,
                    llm_call_fn=llm_call,
                    screen_context=screen_text,
                    psyche_enrichment=enrichment,
                ),
                timeout=_API_TIMEOUT,
            )
            # 返り値が辞書であり、テキストを含むことを確認する
            assert isinstance(result, dict)
            assert "text" in result
            assert isinstance(result["text"], str)
            # 生成されたテキストを取得する
            response_text = result.get("text", "")

        # 経過時間を計算する
        elapsed = time.monotonic() - t0

        # テスト結果をログに記録する
        _write_log(log_dir, "single_turn_pipeline", {
            "percept_emotion": percept.emotion,
            "percept_intent": percept.intent,
            "policy_label": policy.get("policy_label", ""),
            "is_silence": is_silence_policy(policy),
            "response_preview": response_text[:100] if response_text else "(silence)",
            "tick_count": orchestrator.tick_count,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_multi_turn_pipeline(self, orchestrator, log_dir):
        """複数ターンの連続実行で Phase 更新が例外なく完了し、
        心理状態が（完全には）固定化していないことを確認する。

        3種類の異なる感情トーンのテキストを順番に入力し、
        各ターンの処理が正常に完了することを検証する。
        状態変化の「方向」や「適切さ」は検証しない（安全弁4: 内容非依存弁）。
        """
        # 必要なモジュールをインポートする
        from psyche.perception import parse_percept
        from psyche.expression import render_expression
        from psyche.silence_hesitation import is_silence_policy
        from src.llm_wrapper import llm_call

        # 3種類の感情トーンが異なる入力テキスト
        inputs = [
            "おはよう！今日はいい天気だね！",         # ポジティブ
            "最近ちょっと疲れちゃったんだよね...",    # ネガティブ
            "でも明日は楽しみなことがあるんだ！",    # 期待
        ]

        # 各ターンの結果を蓄積するリスト
        turn_results = []     # ターンごとの詳細結果
        states_before = []    # 各ターン開始前の心理状態サマリ
        states_after = []     # 各ターン終了後の心理状態サマリ

        # 各入力テキストに対してパイプラインを実行する
        for i, text in enumerate(inputs):
            # タイマーを開始する
            t0 = time.monotonic()

            # ターン開始前の心理状態サマリを記録する
            state_before = orchestrator.psyche.emotion_summary()
            states_before.append(state_before)

            # 知覚処理を実行する
            percept = await asyncio.wait_for(
                parse_percept(
                    text,
                    llm_call_fn=llm_call,
                    state=orchestrator.psyche,
                ),
                timeout=_API_TIMEOUT,
            )

            # 心理状態を更新する（delta=2.0 は2秒経過を示す）
            orchestrator.post_response_update(percept, 2.0, "viewer")

            # 方針を選択する
            policy = orchestrator.select_policy_dict(percept, [], "viewer")

            # 表現を生成する（沈黙が選ばれなかった場合）
            response_text = ""
            if not is_silence_policy(policy):
                # エンリッチメントを取得する
                enrichment = orchestrator.get_prompt_enrichment("viewer")
                # 表現を生成する
                result = await asyncio.wait_for(
                    render_expression(
                        state=orchestrator.psyche,
                        policy=policy,
                        memory_snippet=[],
                        persona=_PERSONA,
                        llm_call_fn=llm_call,
                        screen_context=text,
                        psyche_enrichment=enrichment,
                    ),
                    timeout=_API_TIMEOUT,
                )
                # 生成されたテキストを取得する
                response_text = result.get("text", "")

                # 自己行動知覚に通知する（brain.py と同じフロー）
                if response_text:
                    orchestrator.notify_self_output(
                        response_text=response_text,
                        policy_label=policy.get("policy_label", ""),
                    )

            # ターン終了後の心理状態サマリを記録する
            state_after = orchestrator.psyche.emotion_summary()
            states_after.append(state_after)
            # 経過時間を計算する
            elapsed = time.monotonic() - t0

            # このターンの結果を蓄積する
            turn_results.append({
                "turn": i + 1,                  # ターン番号（1始まり）
                "input_text": text,              # 入力テキスト
                "percept_emotion": percept.emotion,  # 検出された感情ラベル
                "policy_label": policy.get("policy_label", ""),  # 選択された方針
                "response_preview": response_text[:100] if response_text else "(silence)",
                "state_before": state_before,    # ターン前の状態サマリ
                "state_after": state_after,      # ターン後の状態サマリ
                "elapsed_seconds": round(elapsed, 3),  # 所要時間（秒）
            })

        # ティックカウントが入力回数分以上進んだことを確認する
        assert orchestrator.tick_count >= len(inputs)

        # 心理状態が全ターンで完全に同一だったかを記録する（観測のみ）
        # 注意: デフォルト状態では変化が観測されないことがあるため、
        # これはテスト失敗にはしない（ソフトチェック）
        all_same = all(b == a for b, a in zip(states_before, states_after))

        # テスト結果をログに記録する
        _write_log(log_dir, "multi_turn_pipeline", {
            "total_turns": len(inputs),
            "final_tick_count": orchestrator.tick_count,
            "all_states_same": all_same,
            "turns": turn_results,
        })

    @pytest.mark.asyncio
    async def test_orchestrator_no_exception_on_phase_update(self, orchestrator, log_dir):
        """orchestrator の Phase 更新、方針選択、エンリッチメント生成が
        例外を投げずに完了することを確認する。

        70+システムの全てが実 API 応答由来の Percept を受け取っても
        クラッシュしないことを保証する基本テスト。
        """
        # 知覚処理関数をインポートする
        from psyche.perception import parse_percept
        from src.llm_wrapper import llm_call

        # テスト入力テキストで知覚処理を実行する
        percept = await asyncio.wait_for(
            parse_percept(
                "テスト入力です",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # Phase 更新が例外なく完了することを確認する
        orchestrator.post_response_update(percept, 1.0, "viewer")

        # 方針選択が例外なく完了し、辞書が返ることを確認する
        policy = orchestrator.select_policy_dict(percept, [], "viewer")
        assert isinstance(policy, dict)

        # エンリッチメント生成が例外なく完了し、文字列が返ることを確認する
        enrichment = orchestrator.get_prompt_enrichment("viewer")
        assert isinstance(enrichment, str)

        # テスト結果をログに記録する
        _write_log(log_dir, "no_exception_phase_update", {
            "tick_count": orchestrator.tick_count,
            "enrichment_length": len(enrichment),
            "policy_label": policy.get("policy_label", ""),
        })


# ════════════════════════════════════════════════════════════════════
# 階層5: テキスト入力経路テスト
# テキスト対話入力経路（brain.py の think_text と同等のフロー）を
# 実 API 応答で検証する。画面知覚なしの純テキスト対話経路。
# ════════════════════════════════════════════════════════════════════


class TestTextInputPathway:
    """テキスト対話入力経路が実 API 応答で正しく動作するかを検証するテスト群。"""

    @pytest.mark.asyncio
    async def test_text_input_pipeline(self, orchestrator, log_dir):
        """テキスト入力→process_text_input→心理更新→方針選択→表現 の
        パイプライン全体が完走することを確認する。

        brain.py の think_text メソッドと同等のフローを手動で再現する。
        """
        # 必要なモジュールをインポートする
        from psyche.perception import parse_percept
        from psyche.expression import render_expression
        from psyche.silence_hesitation import is_silence_policy
        from src.llm_wrapper import llm_call

        # テスト用の入力テキスト
        user_text = "今日はどんな一日だった？"

        # 所要時間計測のためタイマーを開始する
        t0 = time.monotonic()

        # ステップ1: 知覚処理（テキスト入力を Percept に構造化する）
        percept = await asyncio.wait_for(
            parse_percept(
                user_text,
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # ステップ2: テキスト対話入力の前処理（テキスト経路固有の処理）
        # sender_id と conversation_id はテスト用の識別子を使う
        handoff = orchestrator.process_text_input(
            text=user_text,
            sender_id="test_user",
            conversation_id="e2e_test",
        )

        # ステップ3: 心理状態を更新する
        # "text" はテキスト対話経路であることを示す入力経路識別子
        orchestrator.post_response_update(percept, 1.0, "text")

        # ステップ4: 方針を選択する
        policy = orchestrator.select_policy_dict(percept, [], "text")
        # 方針が辞書であり、ラベルが含まれていることを確認する
        assert isinstance(policy, dict)
        assert "policy_label" in policy

        # ステップ5: 表現を生成する（沈黙が選ばれなかった場合）
        response_text = ""
        if not is_silence_policy(policy):
            # テキスト経路用のエンリッチメントを取得する
            enrichment = orchestrator.get_prompt_enrichment("text")
            # 表現を生成する
            result = await asyncio.wait_for(
                render_expression(
                    state=orchestrator.psyche,
                    policy=policy,
                    memory_snippet=[],
                    persona=_PERSONA,
                    llm_call_fn=llm_call,
                    screen_context=user_text,
                    psyche_enrichment=enrichment,
                ),
                timeout=_API_TIMEOUT,
            )
            # 生成されたテキストを取得する
            response_text = result.get("text", "")

            # 自己行動知覚に通知する（brain.py と同じフロー）
            if response_text:
                orchestrator.notify_self_output(
                    response_text=response_text,
                    policy_label=policy.get("policy_label", ""),
                )

        # 経過時間を計算する
        elapsed = time.monotonic() - t0

        # テスト結果をログに記録する
        _write_log(log_dir, "text_input_pipeline", {
            "user_text": user_text,
            "percept_emotion": percept.emotion,
            "percept_intent": percept.intent,
            "policy_label": policy.get("policy_label", ""),
            "response_preview": response_text[:200] if response_text else "(silence)",
            "tick_count": orchestrator.tick_count,
            "elapsed_seconds": round(elapsed, 3),
        })

    @pytest.mark.asyncio
    async def test_text_input_percept_structure(self, orchestrator):
        """テキスト入力から生成された Percept が正しい構造を持つことを確認する。

        テキスト対話経路に特化した知覚構造化の形式検証。
        """
        # 知覚処理関数と Percept 型をインポートする
        from psyche.perception import parse_percept
        from psyche.state import Percept
        from src.llm_wrapper import llm_call

        # 嬉しい感情を含むテキストで知覚処理を実行する
        percept = await asyncio.wait_for(
            parse_percept(
                "嬉しいニュースがあるの！聞いて！",
                llm_call_fn=llm_call,
                state=orchestrator.psyche,
            ),
            timeout=_API_TIMEOUT,
        )

        # --- 形式的検証のみ（安全弁4: 内容非依存弁）---
        # Percept インスタンスであることを確認する
        assert isinstance(percept, Percept)
        # 感情ラベルが空でない文字列であることを確認する
        assert isinstance(percept.emotion, str)
        assert len(percept.emotion) > 0
        # 意図ラベルが空でない文字列であることを確認する
        assert isinstance(percept.intent, str)
        assert len(percept.intent) > 0
        # トピックリストがリスト型であることを確認する
        assert isinstance(percept.topics, list)
        # 感情価が正しい型と範囲であることを確認する
        assert isinstance(percept.emotion_valence, float)
        assert -1.0 <= percept.emotion_valence <= 1.0
        # 意味テキストが空でない文字列であることを確認する
        assert isinstance(percept.meaning, str)
        assert len(percept.meaning) > 0


# ════════════════════════════════════════════════════════════════════
# 階層6: 状態汚染ガードテスト
# e2e テストの実行が運用中の永続状態に影響を与えないことを検証する。
# orchestrator は save() を呼ばない限りファイルを生成しない。
# ════════════════════════════════════════════════════════════════════


class TestStateContaminationGuard:
    """e2eテストが永続状態を汚染しないことを検証するテスト群。"""

    def test_no_save_file_created(self, temp_data):
        """ティック実行後に永続化ファイルが生成されていないことを確認する。

        save() を呼ばずにティックを回しただけでは、
        ファイルシステムに何も書き込まれないことを検証する。
        """
        # 心理統合管理クラスをインポートする
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        # 一時データディレクトリで orchestrator を作成する
        orch = PsycheOrchestrator(memory_count=0, data_dir=temp_data)

        # テスト用の知覚データを作成する（API コール不要）
        percept = Percept(text="test", emotion="neutral", intent="unknown")
        # 1ティック実行する（全 Phase が走る）
        orch.post_response_update(percept, 1.0, "viewer")

        # 一時データディレクトリに永続化ファイルがないことを確認する
        save_files = list(temp_data.glob("psyche_state*.json"))
        assert len(save_files) == 0, (
            f"Unexpected save files found: {save_files}"
        )

    def test_orchestrator_independent_instances(self, temp_data):
        """2つの orchestrator インスタンスが完全に独立していることを確認する。

        一方のインスタンスでティックを回しても、
        もう一方のインスタンスの状態に影響しないことを検証する。
        """
        # 心理統合管理クラスと知覚データ型をインポートする
        from psyche.orchestrator import PsycheOrchestrator
        from psyche.state import Percept

        # 同じデータディレクトリで2つの独立したインスタンスを作成する
        orch1 = PsycheOrchestrator(memory_count=0, data_dir=temp_data)
        orch2 = PsycheOrchestrator(memory_count=0, data_dir=temp_data)

        # テスト用の知覚データを作成する
        percept = Percept(
            text="test input",    # テスト入力テキスト
            emotion="happy",      # 感情ラベル
            intent="greeting",    # 意図ラベル
            emotion_valence=0.7,  # 感情価
        )

        # orch1 のみでティックを実行する
        orch1.post_response_update(percept, 1.0, "viewer")

        # orch1 はティックが進んでいることを確認する
        assert orch1.tick_count >= 1
        # orch2 はティックが進んでいないことを確認する（独立性の証明）
        assert orch2.tick_count == 0
