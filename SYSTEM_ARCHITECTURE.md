# Cyrene AI  - 完全システムアーキテクチャ仕様書

作成日: 2026-02-09
総コード行数: ~62,000行
総テスト数: 1,808テスト

---

## 目次

1. [システム概要](#1-システム概要)
2. [コード統計](#2-コード統計)
3. [コアシステム詳細](#3-コアシステム詳細)
4. [Psycheシステム詳細](#4-psycheシステム詳細)
5. [データフロー](#5-データフロー)
6. [モジュール間連携](#6-モジュール間連携)
7. [設計原則](#7-設計原則)

---

## 1. システム概要

### 1.1 アーキテクチャ全体図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              外部世界                                        │
│                                                                             │
│   ディスプレイ画面 ──────────────────────────────────────→ スピーカー       │
│         │                                                      ↑            │
│         │                                                      │            │
│         ▼                                                      │            │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐         │
│   │  vision   │───→│   brain   │───→│  psyche   │───→│   voice   │         │
│   │  (393行)  │    │  (592行)  │    │(19,239行) │    │  (437行)  │         │
│   └───────────┘    └───────────┘    └───────────┘    └───────────┘         │
│         │                │                │                │                │
│    dxcam/YOLO       Gemini API      心理処理        Style-Bert-VITS2       │
│    /EasyOCR                                                                 │
│                                                                             │
│                    ┌───────────┐                                            │
│                    │   main    │ ← メインループ制御                         │
│                    │  (264行)  │                                            │
│                    └───────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 技術スタック

| レイヤー | 技術 | 用途 |
|---------|------|------|
| 画面キャプチャ | dxcam | GPU直接アクセス高速キャプチャ |
| 物体検出 | YOLOv8n | リアルタイムオブジェクト認識 |
| 文字認識 | EasyOCR | 日本語+英語テキスト抽出 |
| 思考生成 | Gemini 2.5 Flash | マルチモーダルAI推論 |
| 心理処理 | Python (自作) | 感情・判断・記憶システム |
| 音声合成 | Style-Bert-VITS2 | 高品質日本語TTS |
| アバター | Warudo | 3D VTuberレンダリング |

---

## 2. コード統計

### 2.1 ディレクトリ別コード行数

| ディレクトリ | ファイル数 | 総行数 | 説明 |
|-------------|-----------|--------|------|
| psyche/ | 49 | 32,448 | 心理システム本体 |
| tests/ | 38 | 25,280 | 自動テストコード |
| src/ | 14 | 2,590 | 補助モジュール |
| ルート | 4 | 1,686 | コアシステム |
| **合計** | **105** | **62,004** | |

### 2.2 Psycheモジュール詳細 (行数順)

| # | モジュール | 行数 | テスト数 | カテゴリ | 説明 |
|---|-----------|------|---------|---------|------|
| 1 | self_model.py | 1,601 | 70 | 内省 | 自己状態統合モデル（統一ビュー） |
| 2 | temporal_self_difference.py | 1,320 | 56 | 内省 | 自己モデル差分認知（時間変化認識） |
| 3 | continuity_strain.py | 939 | 68 | 内省 | 自己連続性負荷（違和感認知） |
| 4 | self_image_integration.py | 1,184 | 59 | 内省 | 自己像統合（暫定的自己像生成） |
| 5 | self_narrative.py | 1,491 | 98 | 内省 | 自己物語形成（非規範・観測型） |
| 6 | identity_coherence.py | 1,110 | 76 | 内省 | 自己同一性の揺らぎ認知 |
| 7 | episodic_memory.py | 1,709 | 113 | 記憶 | エピソード記憶（自伝的記憶） |
| 8 | introspection_consumption.py | 1,455 | 94 | 内省 | 内省の消費層（読み取り可能断片の循環） |
| 9 | expectation_formation.py | 1,485 | 103 | 内省 | 予期・期待の形成（未来方向の連続性投射） |
| 10 | other_agent_model.py | 1,603 | 112 | 内省 | 他者モデル（他者状態の仮説的推測） |
| 11 | responsibility_dispersion.py | 1,039 | 48 | 責任 | 責任の発散・昇華・時間分配 |
| 3 | goal_candidates.py | 929 | 46 | 目的 | 目的候補（白昼夢）生成 |
| 4 | self_reference.py | 923 | 52 | 内省 | 自己参照ループ |
| 5 | long_term_dynamics.py | 882 | 38 | 内省 | 長期統計観測 |
| 6 | introspection_trace.py | 864 | 40 | 内省 | 内省ログ生成 |
| 7 | repeated_tendency.py | 858 | 35 | 目的 | 反復傾向（習慣）形成 |
| 8 | transient_goal.py | 812 | 34 | 目的 | 一時的目的選択 |
| 9 | proto_goal_vector.py | 774 | 48 | 目的 | 方向ベクトル（ゴースト） |
| 10 | context_sensitivity.py | 754 | 44 | 判断 | 外部文脈感受性 |
| 11 | value_orientation.py | 746 | 34 | 目的 | 長期価値観 |
| 12 | stability_valve.py | 728 | 40 | 判断 | 極端回避バルブ |
| 13 | silence_hesitation.py | 724 | 36 | 出力 | 沈黙・躊躇い表現 |
| 14 | __init__.py | 1,183 | - | 基盤 | エクスポート定義 |
| 15 | tone.py | 698 | 36 | 出力 | トーン・ユーモア制御 |
| 16 | tendency_awareness.py | 651 | 44 | 内省 | 傾向の自己認知 |
| 16 | scoped_goal.py | 660 | 40 | 目的 | スコープ目的（1ターン） |
| 17 | stm_emotion_coupling.py | 604 | 40 | 感情 | 短期記憶-感情連携 |
| 18 | multi_emotion.py | 495 | 36 | 感情 | 複数感情独立管理 |
| 19 | responsibility.py | 480 | 32 | 責任 | 責任記録・評価 |
| 20 | dynamics.py | 474 | 24 | 感情 | 感情ダイナミクス相 |
| 21 | decision_bias.py | 465 | 30 | 判断 | 判断バイアス計算 |
| 22 | short_term_loop.py | 432 | 24 | 記憶 | 短期感情ループ |
| 23 | short_term_memory.py | 399 | 24 | 記憶 | 短期記憶管理 |
| 24 | persistence.py | 395 | 22 | 基盤 | 永続化システム |
| 25 | emotion_amplitude.py | 362 | 24 | 感情 | 感情振幅調整 |
| 26 | reaction_with_stm.py | 294 | - | 感情 | STM統合反応 |
| 27 | thought.py | 293 | - | 出力 | 思考候補生成・選択 |
| 28 | state.py | 258 | - | 基盤 | 心理状態データ構造 |
| 29 | snapshot.py | 239 | - | 基盤 | スナップショット管理 |
| 30 | responsibility_manager.py | 210 | - | 責任 | 責任マネージャー |
| 31 | reaction.py | 201 | - | 感情 | 反応処理 |
| 32 | expression.py | 156 | - | 出力 | 表現生成 |
| 33 | perception.py | 157 | - | 入力 | 知覚処理 |
| 34 | memory_link.py | 101 | - | 記憶 | 記憶検索 |
| 35 | continuity_manager.py | 95 | - | 4柱 | 連続性管理 |
| 36 | attachment_manager.py | 95 | - | 4柱 | 愛着管理 |
| 37 | identity_manager.py | 90 | - | 4柱 | アイデンティティ管理 |
| 38 | projection_manager.py | 89 | - | 4柱 | 未来投射管理 |
| 39 | pillars.py | 76 | - | 4柱 | 4柱状態定義 |
| 40 | fear.py | 76 | - | 4柱 | 恐怖指数計算 |

### 2.3 コアシステムファイル

| ファイル | 行数 | 主要クラス/関数 | 説明 |
|---------|------|----------------|------|
| brain.py | 592 | CyreneBrain | Gemini API連携・思考生成 |
| voice.py | 437 | VoiceClient | Style-Bert-VITS2連携 |
| vision.py | 393 | GameCapture, HybridEye | 画面キャプチャ・分析 |
| main.py | 264 | main() | メインループ制御 |

### 2.4 補助モジュール (src/)

| ファイル | 行数 | 説明 |
|---------|------|------|
| simulation.py | 564 | 長期挙動シミュレーション |
| api.py | 422 | FastAPI REST API |
| llm_wrapper.py | 216 | LLM抽象化レイヤー |
| memory_manager.py | 216 | 長期記憶+Embedding管理 |
| thinker.py | 167 | 思考生成補助 |
| state_manager.py | 164 | 状態管理補助 |
| cli_tools.py | 161 | CLIツール |
| renderer.py | 160 | 表示レンダリング |
| attachment_manager.py | 125 | 愛着管理補助 |
| emotion_model.py | 124 | 感情モデル補助 |
| projection_manager.py | 111 | 未来投射管理補助 |
| identity_manager.py | 108 | アイデンティティ管理補助 |
| logging_config.py | 47 | ログ設定 |

---

## 3. コアシステム詳細

### 3.1 vision.py - 視覚システム

#### 3.1.1 GameCapture クラス

```
目的: GPU加速による高速画面キャプチャ

初期化:
  - dxcam.create(device_idx=0, output_color="RGB")
  - Desktop Duplication API使用
  - RTX 3070 Ti最適化

主要メソッド:
┌─────────────────────────────────────────────────────────────────┐
│ capture_frame() -> PIL.Image                                    │
│   1. camera.grab() でフレーム取得                               │
│   2. numpy配列からPIL.Image変換                                 │
│   3. ネイティブ解像度を維持（リサイズなし）                     │
│   戻り値: PIL.Image または None（変化なし時）                   │
│                                                                 │
│ get_base64_image(image, quality=95) -> str                      │
│   1. JPEG形式でバッファに保存                                   │
│   2. Base64エンコード                                           │
│   戻り値: Base64文字列（Gemini API用）                          │
│                                                                 │
│ release()                                                       │
│   dxcamリソース解放                                             │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 HybridEye クラス

```
目的: YOLO物体検出 + EasyOCR文字認識の統合

初期化:
  - analysis_interval: 0.5秒（スロットリング間隔）
  - YOLOv8nモデル（Nano、速度優先）
  - EasyOCR（日本語+英語）

主要メソッド:
┌─────────────────────────────────────────────────────────────────┐
│ detect_objects(image, confidence_threshold=0.5) -> List[Dict]   │
│   入力: PIL.Image                                               │
│   処理:                                                         │
│     1. YOLOv8n推論実行                                          │
│     2. 信頼度フィルタリング（>0.5）                             │
│     3. 位置を人間可読形式に変換                                 │
│        - top-left, center, bottom-right など                    │
│   出力: [{"name": "person", "position": "center",               │
│           "confidence": 0.87}, ...]                             │
│                                                                 │
│ read_text(image, confidence_threshold=0.4) -> List[str]         │
│   入力: PIL.Image                                               │
│   処理:                                                         │
│     1. EasyOCR推論実行                                          │
│     2. 信頼度フィルタリング（>0.4）                             │
│     3. 2文字以上のテキストのみ抽出                              │
│   出力: ["ゲームオーバー", "スコア: 1000", ...]                 │
│                                                                 │
│ analyze_frame(image, force=False) -> Optional[Dict]             │
│   入力: PIL.Image, force: スロットリング無視フラグ              │
│   処理:                                                         │
│     1. スロットリングチェック（0.5秒間隔）                      │
│     2. detect_objects() 実行                                    │
│     3. read_text() 実行                                         │
│   出力: {"objects": [...], "text": [...]}                       │
│                                                                 │
│ format_for_prompt(analysis) -> str                              │
│   入力: analyze_frame()の結果                                   │
│   出力:                                                         │
│     "[Vision Sensor Data]"                                      │
│     "Objects detected: person (center), car (left)"             │
│     "Text on screen: ゲームオーバー | スコア: 1000"             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 brain.py - 思考エンジン

#### 3.2.1 CyreneBrain クラス

```
目的: Gemini APIを使用した思考生成とPsyche連携

初期化処理:
┌─────────────────────────────────────────────────────────────────┐
│ __init__():                                                     │
│   1. 環境変数からGEMINI_API_KEY取得                             │
│   2. genai.Client(api_key) でクライアント作成                   │
│   3. identity.md からペルソナ読み込み                           │
│   4. GenerateContentConfig設定:                                 │
│      - system_instruction: ペルソナ                             │
│      - temperature: 1.2（高創造性）                             │
│      - max_output_tokens: 1024                                  │
│   5. MemoryManager初期化（長期記憶+Embedding）                  │
│   6. チャットセッション作成                                     │
│   7. PsycheState初期化（_init_psyche()）                        │
└─────────────────────────────────────────────────────────────────┘

Psyche初期化詳細:
┌─────────────────────────────────────────────────────────────────┐
│ _init_psyche():                                                 │
│   1. IdentityState作成:                                         │
│      - core_traits: [romantic, sweet, playful, caring, ...]     │
│      - trait_confidence: {romantic: 0.9, sweet: 0.9, ...}       │
│   2. AttachmentState作成（デフォルト）                          │
│   3. ContinuityState作成:                                       │
│      - memory_count: len(memories)                              │
│   4. ProjectionState作成:                                       │
│      - goals: [{id: "entertain", description: "視聴者を楽しませる"}]│
│   5. compute_fear_index() で恐怖指数計算                        │
│   6. PsycheState組み立て                                        │
└─────────────────────────────────────────────────────────────────┘

主要メソッド:
┌─────────────────────────────────────────────────────────────────┐
│ async think_streaming(image_path, vision_summary) -> AsyncGen   │
│                                                                 │
│   ステップ1: プロンプト構築 (_build_prompt)                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ a. 基本指示（画像分析依頼）                             │   │
│   │ b. recall_with_mood() で関連記憶検索                    │   │
│   │    - ムード一致バイアス適用                             │   │
│   │    - top_k=3 件取得                                     │   │
│   │ c. _format_psyche_for_prompt() で心理状態追加           │   │
│   │    - 感情サマリー                                       │   │
│   │    - ムード (valence, arousal)                          │   │
│   │    - ドライブ (social, curiosity, expression)           │   │
│   │    - 恐怖サマリー                                       │   │
│   │ d. vision_summary追加（YOLO+OCR結果）                   │   │
│   │ e. 感情タグルール追加                                   │   │
│   │    [happy], [sad], [angry], [surprised], [scared],      │   │
│   │    [loving], [teasing], [neutral]                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ステップ2: Gemini API呼び出し                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ await chat.send_message([prompt, image])                │   │
│   │ → "[happy] わぁ、すごい！このゲーム面白そう♪"          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ステップ3: 応答処理                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ a. "PASS" チェック（沈黙選択）                          │   │
│   │ b. _update_psyche(response) で心理状態更新              │   │
│   │ c. 文単位分割（。！？!?\n♪♥♡★☆）                       │   │
│   │ d. 各文をyield                                          │   │
│   │ e. 5ターンごとに summarize_and_save()                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│ _update_psyche(response_text, vision_summary):                  │
│   1. 感情タグ抽出: re.match(r"\[(\w+)\]", text)                 │
│   2. バレンス変換: _TAG_VALENCE[tag]                            │
│      happy: 0.7, sad: -0.6, angry: -0.5, ...                    │
│   3. Percept生成                                                │
│   4. psyche.react(percept, psyche_state, delta_time) 呼び出し   │
│   5. attachment_manager.update_bond() 呼び出し                  │
│   6. _recompute_fear() で恐怖指数再計算                         │
│                                                                 │
│ async summarize_and_save():                                     │
│   1. 会話ログ最新10件取得                                       │
│   2. Gemini単発呼び出しで要約生成                               │
│      出力JSON: {summary, keywords, importance}                  │
│   3. memory.add_memory(summary, keywords, importance)           │
│   4. 会話ログクリア                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 voice.py - 音声合成システム

#### 3.3.1 VoiceClient クラス

```
目的: Style-Bert-VITS2 APIサーバーとの連携

初期化パラメータ（ゴールデン設定）:
┌─────────────────────────────────────────────────────────────────┐
│ api_url: "http://127.0.0.1:5000"                                │
│ model_id: 0                                                     │
│ style_weight: 2.0   ← 感情の強さ（基準値）                      │
│ sdp_ratio: 0.7      ← 確率的時間予測比率                        │
│ noise: 0.7          ← ノイズスケール                            │
│ noise_w: 0.8        ← ノイズスケールW                           │
│ length: 1.0         ← 話速                                      │
└─────────────────────────────────────────────────────────────────┘

感情別style_weight（main.pyで定義）:
┌─────────────────────────────────────────────────────────────────┐
│ happy/joy:     3.5  ← 明るく弾んだ声                            │
│ angry/mad:     4.5  ← 強い感情表現                              │
│ sad/sorrow:    2.0  ← 落ち着いた声                              │
│ surprised:     3.0  ← 驚きの抑揚                                │
│ scared:        2.5  ← やや緊張した声                            │
│ loving:        3.0  ← 甘い声                                    │
│ teasing:       3.0  ← いたずらっぽい声                          │
│ neutral:       2.0  ← 基準                                      │
└─────────────────────────────────────────────────────────────────┘

主要メソッド:
┌─────────────────────────────────────────────────────────────────┐
│ async speak(text, style, style_weight, play_audio) -> str       │
│                                                                 │
│   ステップ1: テキスト分割 (_split_text)                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 最大50文字で分割（URL長制限回避）                       │   │
│   │ 分割位置優先順:                                         │   │
│   │   1. 句点: 。！？!?\n                                   │   │
│   │   2. 読点: 、,                                          │   │
│   │   3. 強制分割                                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ステップ2: 各チャンク合成 (_synthesize_chunk)                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ HTTP GET /voice?text=...&model_id=0&style=...           │   │
│   │ パラメータ:                                             │   │
│   │   - text: チャンクテキスト                              │   │
│   │   - model_id: 0                                         │   │
│   │   - style: "Neutral"                                    │   │
│   │   - style_weight: 感情に応じた値                        │   │
│   │   - sdp_ratio, noise, noisew, length                    │   │
│   │   - language: "JP"                                      │   │
│   │ 戻り値: WAVバイナリデータ                               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ステップ3: WAV結合 (_combine_wav_data)                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. 各WAVをsoundfileで読み込み                           │   │
│   │ 2. numpy.concatenate()で結合                            │   │
│   │ 3. 結合WAVをバイト列に変換                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ステップ4: 再生 (_play_audio)                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. sounddevice.play(audio_array, sample_rate)           │   │
│   │ 2. sounddevice.wait() でブロッキング待機                │   │
│   │ 3. 0.5秒の余韻待機                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   戻り値: Base64エンコードWAV（Warudo送信用）                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 main.py - メインループ

```
メインループ処理フロー:
┌─────────────────────────────────────────────────────────────────┐
│ async main():                                                   │
│                                                                 │
│   初期化フェーズ:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. GameCapture(target_fps=30) 初期化                    │   │
│   │ 2. CyreneBrain() 初期化                                 │   │
│   │ 3. ensure_server_running() → VoiceClient() 初期化       │   │
│   │ 4. HybridEye(analysis_interval=0.5) 初期化              │   │
│   │ 5. 一時ファイルパス設定                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   メインループ:                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ while True:                                             │   │
│   │                                                         │   │
│   │   [1] 終了キーチェック                                  │   │
│   │       if keyboard.is_pressed('l'): break                │   │
│   │                                                         │   │
│   │   [2] 画面キャプチャ                                    │   │
│   │       frame = capture.capture_frame()                   │   │
│   │       frame.save(temp_path, "JPEG", quality=95)         │   │
│   │                                                         │   │
│   │   [3] 画像分析 (YOLO + OCR)                             │   │
│   │       analysis = hybrid_eye.analyze_frame(frame)        │   │
│   │       vision_summary = hybrid_eye.format_for_prompt()   │   │
│   │                                                         │   │
│   │   [4] 思考生成 & 発話                                   │   │
│   │       async for sentence in brain.think_streaming(...): │   │
│   │         # 感情タグ解析                                  │   │
│   │         emotion, sw, clean = parse_emotion_tag(sentence)│   │
│   │         # 字幕表示                                      │   │
│   │         print(f"[Cyrene] ({emotion}): {clean}")         │   │
│   │         # 音声合成・再生                                │   │
│   │         await voice.speak(clean, style_weight=sw)       │   │
│   │                                                         │   │
│   │   [5] ループ遅延                                        │   │
│   │       await asyncio.sleep(0.1)                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   終了処理:                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. brain.summarize_and_save() - 長期記憶保存            │   │
│   │ 2. capture.release() - dxcamリソース解放                │   │
│   │ 3. voice.close() - HTTPクライアント終了                 │   │
│   │ 4. 一時ファイル削除                                     │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Psycheシステム詳細

### 4.1 状態管理層

#### 4.1.1 PsycheState (state.py)

```
データ構造:
┌─────────────────────────────────────────────────────────────────┐
│ @dataclass                                                      │
│ class PsycheState:                                              │
│   emotions: EmotionVector                                       │
│     - joy: float (0.0-1.0)                                      │
│     - sadness: float (0.0-1.0)                                  │
│     - anger: float (0.0-1.0)                                    │
│     - fear: float (0.0-1.0)                                     │
│     - surprise: float (0.0-1.0)                                 │
│     - disgust: float (0.0-1.0)                                  │
│                                                                 │
│   drives: DriveVector                                           │
│     - curiosity: float (0.0-1.0)                                │
│     - affiliation: float (0.0-1.0)                              │
│     - autonomy: float (0.0-1.0)                                 │
│     - competence: float (0.0-1.0)                               │
│                                                                 │
│   mood: Mood                                                    │
│     - valence: float (-1.0 to 1.0) ← ポジティブ/ネガティブ      │
│     - arousal: float (0.0-1.0) ← 覚醒度                         │
│                                                                 │
│   fear_level: float (0.0-1.0) ← 総合恐怖度                      │
│   fear_index: FearIndex ← 4柱への脅威詳細                       │
│                                                                 │
│   # 4柱状態                                                     │
│   identity: IdentityState                                       │
│   attachment: AttachmentState                                   │
│   continuity: ContinuityState                                   │
│   projection: ProjectionState                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 4柱システム (pillars.py)

```
4柱と恐怖の関係:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────┐    calc_identity_risk()    ┌─────────────┐    │
│  │IdentityState│ ─────────────────────────→ │identity_risk│    │
│  │  core_traits│                            │  (0.0-1.0)  │    │
│  │  confidence │                            └──────┬──────┘    │
│  └─────────────┘                                   │            │
│                                                    │            │
│  ┌─────────────┐    calc_attachment_risk()  ┌─────────────┐    │
│  │Attachment   │ ─────────────────────────→ │attachment   │    │
│  │State        │                            │_risk        │    │
│  │  bonds      │                            │  (0.0-1.0)  │    │
│  └─────────────┘                            └──────┬──────┘    │
│                                                    │            │
│  ┌─────────────┐    calc_continuity_risk()  ┌─────────────┐    │
│  │Continuity   │ ─────────────────────────→ │continuity   │    │
│  │State        │                            │_risk        │    │
│  │memory_count │                            │  (0.0-1.0)  │    │
│  └─────────────┘                            └──────┬──────┘    │
│                                                    │            │
│  ┌─────────────┐    calc_projection_risk()  ┌─────────────┐    │
│  │Projection   │ ─────────────────────────→ │projection   │    │
│  │State        │                            │_risk        │    │
│  │  goals      │                            │  (0.0-1.0)  │    │
│  └─────────────┘                            └──────┬──────┘    │
│                                                    │            │
│                                                    ▼            │
│                                          ┌─────────────────┐   │
│                                          │compute_fear_index│   │
│                                          │                 │   │
│                                          │ FearIndex:      │   │
│                                          │  value: 0.0-1.0 │   │
│                                          │  dominant_fear  │   │
│                                          │  components{}   │   │
│                                          └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 感情処理層

#### 4.2.1 感情処理フロー

```
入力刺激からの感情更新フロー:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Percept (入力刺激)                                             │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ reaction.py: react()                                    │   │
│  │   1. 感情更新（Perceptのemotionとvalenceから）           │   │
│  │   2. 自然減衰適用                                        │   │
│  │   3. ドライブ更新                                        │   │
│  │   4. ムードドリフト                                      │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ multi_emotion.py: apply_independent_update()            │   │
│  │   - 6感情を独立して更新                                  │   │
│  │   - 相反感情の同時存在を許可 (joy + sadness)             │   │
│  │   - 各感情に個別の減衰率適用可能                         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ dynamics.py: update_dynamics()                          │   │
│  │   相判定:                                                │   │
│  │     NORMAL → RISING (急激な感情上昇検出時)               │   │
│  │     RISING → PEAK (閾値超過時)                           │   │
│  │     PEAK → REBOUND (持続時間経過後)                      │   │
│  │     REBOUND → NORMAL (反動完了後)                        │   │
│  │                                                          │   │
│  │   DynamicsState:                                         │   │
│  │     phase: DynamicsPhase                                 │   │
│  │     peak_emotion: str                                    │   │
│  │     accumulated_intensity: float                         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ emotion_amplitude.py: apply_amplitude_to_delta()        │   │
│  │   PEAK相: 感情変化を増幅 (×1.2-1.5)                      │   │
│  │   REBOUND相: 感情変化を抑制 (×0.7-0.9)                   │   │
│  │   NORMAL相: 変化なし (×1.0)                              │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ stm_emotion_coupling.py: apply_stm_coupling()           │   │
│  │   1. 持続性修正: STM残響により感情の減衰を遅らせる       │   │
│  │   2. 再活性化: 類似刺激で過去の感情を再活性化            │   │
│  │   3. 蓄積効果: 同種刺激の連続で感情が強化                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 記憶層

#### 4.3.1 短期記憶システム

```
短期記憶の構造と処理:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  新規刺激                                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ short_term_memory.py                                    │   │
│  │                                                          │   │
│  │ StimulusEntry:                                           │   │
│  │   id: str                                                │   │
│  │   timestamp: float                                       │   │
│  │   raw_intensity: float (0.0-1.0)                         │   │
│  │   emotion_type: str                                      │   │
│  │   residue_weight: float (1.0→0.0、時間で減衰)            │   │
│  │   processed: bool                                        │   │
│  │                                                          │   │
│  │ ShortTermMemory:                                         │   │
│  │   entries: List[StimulusEntry] (最大20件)                │   │
│  │   context_continuity_score: float                        │   │
│  │                                                          │   │
│  │ 処理:                                                    │   │
│  │   add_entry(): 新規エントリ追加                          │   │
│  │   apply_decay(): residue_weight減衰                      │   │
│  │   prune_entries(): weight<閾値のエントリ削除             │   │
│  │   get_unprocessed_residue(): 未処理エントリ取得          │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ short_term_loop.py: execute_full_loop()                 │   │
│  │                                                          │   │
│  │ LoopState:                                               │   │
│  │   stm: ShortTermMemory                                   │   │
│  │   dynamics: DynamicsState                                │   │
│  │   loop_count: int                                        │   │
│  │                                                          │   │
│  │ 処理フロー:                                              │   │
│  │   1. 刺激をSTMに追加                                     │   │
│  │   2. 残響影響計算 (compute_residue_influence)            │   │
│  │   3. ダイナミクス更新                                    │   │
│  │   4. 減衰適用                                            │   │
│  │                                                          │   │
│  │ LoopResult:                                              │   │
│  │   residue_influence: ResidueInfluence                    │   │
│  │   dynamics_phase: DynamicsPhase                          │   │
│  │   decay_modifier: float                                  │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ResidueInfluence (残響影響)                             │   │
│  │   emotion_biases: Dict[str, float]                       │   │
│  │     - 各感情への残響バイアス                             │   │
│  │   intensity: float                                       │   │
│  │     - 総合残響強度                                       │   │
│  │   continuity: float                                      │   │
│  │     - 文脈連続性スコア                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 責任システム層

#### 4.4.1 責任の記録と発散

```
責任システムの構造:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  判断実行                                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ responsibility.py: record_decision()                    │   │
│  │                                                          │   │
│  │ DecisionRecord:                                          │   │
│  │   decision_id: str                                       │   │
│  │   timestamp: float                                       │   │
│  │   policy: str (選択したポリシー)                         │   │
│  │   confidence: float                                      │   │
│  │   harm_estimate: float (予想される悪影響)                │   │
│  │                                                          │   │
│  │ ResponsibilityState:                                     │   │
│  │   total_weight: float (累積責任重量)                     │   │
│  │   accumulated_harm: float (累積悪影響)                   │   │
│  │   accumulated_confidence: float (累積信頼度)             │   │
│  │   pending_decisions: int (保留中判断数)                  │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ responsibility_dispersion.py                            │   │
│  │                                                          │   │
│  │ ResponsibilityUnit:                                      │   │
│  │   unit_id: str                                           │   │
│  │   weight: float         ← 責任の重さ                     │   │
│  │   distance: float       ← 心理的距離 (0=近い, ∞=遠い)    │   │
│  │   meaning: str          ← 意味付け ("harm", "learning")  │   │
│  │   time_slice: str       ← 時間帯 ("past", "future")      │   │
│  │   generation: int       ← 変換世代                       │   │
│  │   parent_id: str        ← 親ユニットID                   │   │
│  │                                                          │   │
│  │ 操作（保存則: 総重量は常に保存）:                        │   │
│  │                                                          │   │
│  │ disperse_responsibility(unit, num_parts):                │   │
│  │   ┌─────────────────────────────────────────────┐       │   │
│  │   │ 1つのユニットを複数に分散                    │       │   │
│  │   │ 例: weight=0.6 → 0.2 + 0.2 + 0.2            │       │   │
│  │   │ 心理的に分散して負担軽減                     │       │   │
│  │   └─────────────────────────────────────────────┘       │   │
│  │                                                          │   │
│  │ sublimate_responsibility(unit, new_meaning):             │   │
│  │   ┌─────────────────────────────────────────────┐       │   │
│  │   │ 意味を変換（昇華）                           │       │   │
│  │   │ 例: "harm" → "learning"                      │       │   │
│  │   │ 同じ重量でも心理的負担が軽減                 │       │   │
│  │   └─────────────────────────────────────────────┘       │   │
│  │                                                          │   │
│  │ distribute_over_time(unit, time_slices):                 │   │
│  │   ┌─────────────────────────────────────────────┐       │   │
│  │   │ 時間軸に分配                                 │       │   │
│  │   │ 例: 現在100% → 過去40% + 現在30% + 未来30%   │       │   │
│  │   │ 時間的に分散して現在の負担軽減               │       │   │
│  │   └─────────────────────────────────────────────┘       │   │
│  │                                                          │   │
│  │ adjust_distance(unit, new_distance):                     │   │
│  │   ┌─────────────────────────────────────────────┐       │   │
│  │   │ 心理的距離を調整                             │       │   │
│  │   │ distance↑ → 責任を「遠く」感じる            │       │   │
│  │   └─────────────────────────────────────────────┘       │   │
│  │                                                          │   │
│  │ DispersionState:                                         │   │
│  │   units: List[ResponsibilityUnit]                        │   │
│  │   audit_log: List[AuditEntry] (変換履歴)                 │   │
│  │   initial_total_weight: float (保存則検証用)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 目的階層システム

#### 4.5.1 目的階層の全体構造

```
目的階層（時間軸と影響力の関係）:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  時間軸        モジュール              最大バイアス  特徴       │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  【長期】      value_orientation.py      ±5%       超高慣性     │
│  数百ターン    ├─ 5次元抽象軸 (A/B/C/D/E)                       │
│               ├─ 1回の更新で~0.1%変化                          │
│               └─ 「信念」ではなく「傾き」                       │
│                        │                                        │
│                        │ 観測（影響なし）                       │
│                        ▼                                        │
│  【中期】      proto_goal_vector.py      なし       ゴースト    │
│  数十ターン    ├─ 行動履歴から方向ベクトル生成                  │
│               ├─ 判断に影響しない（観測のみ）                   │
│               └─ 自然減衰                                       │
│                        │                                        │
│                        │ 投射                                   │
│                        ▼                                        │
│  【中期】      goal_candidates.py        なし       白昼夢      │
│               ├─ ベクトルから目的候補を確率的生成               │
│               ├─ 複数の矛盾する候補が共存可能                   │
│               ├─ 決して「選択」されない                         │
│               └─ カテゴリ: APPROACH/AVOIDANCE/CONNECTION/...    │
│                        │                                        │
│                        │ 選択                                   │
│                        ▼                                        │
│  【短期】      transient_goal.py         ±12%      軽量責任     │
│  数ターン     ├─ 候補から1つを「仮選択」                        │
│               ├─ 0-1個のみアクティブ                            │
│               ├─ 軽量責任 (weight=0.1, distance=0.8)            │
│               └─ 自然解除 or 明示的解除                         │
│                        │                                        │
│                        │ コミット                               │
│                        ▼                                        │
│  【1ターン】  scoped_goal.py             ±8%       エフェメラル │
│               ├─ 今ターンだけの焦点                             │
│               ├─ 軽量責任 (weight=0.05, distance=0.9)           │
│               ├─ ターン終了で自動消滅                           │
│               └─ 永続化禁止                                     │
│                        │                                        │
│                        │ 観測                                   │
│                        ▼                                        │
│  【中期】      repeated_tendency.py      ±6%       習慣        │
│               ├─ ScopedGoalの使用パターンを追跡                 │
│               ├─ 3回以上の反復で傾向形成                        │
│               ├─ 性格ではなく「習慣」「慣性」                   │
│               └─ 非使用時は自然減衰                             │
│                        │                                        │
│                        │ 自己認知                               │
│                        ▼                                        │
│  【観測】     tendency_awareness.py      なし      自己記述用   │
│               ├─ 数値を抽象概念に変換                           │
│               │   (SLIGHT / MODERATE / STRONG)                  │
│               ├─ 判断に影響しない                               │
│               └─ SelfReferenceSystemへ接続                      │
│                        │                                        │
│                        │ 統合ビュー                             │
│                        ▼                                        │
│  【観測】     self_model.py             なし      自己像生成    │
│               ├─ 全内部状態の統合観測                           │
│               │   (感情・責任・傾向・方向・価値)                │
│               ├─ 抽象的記述のみ公開                             │
│               │   (CALM / BURDENED / HABITUAL / etc.)           │
│               ├─ 判断には一切影響しない（鏡）                   │
│               └─ SelfReference・IntrospectionTraceへ接続        │
│                        │                                        │
│                        │ 時間差分認知                           │
│                        ▼                                        │
│  【観測】     temporal_self_difference.py  なし    変化認識     │
│               ├─ 過去と現在のSelfModelを比較                    │
│               │   (STABLE / FLUCTUATING / SHIFTING / etc.)      │
│               ├─ 評価なし（良い悪いの判断なし）                 │
│               ├─ 判断に影響しない、認知のみ                     │
│               └─ 自然収束で差分縮小                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.5.2 各モジュールの処理詳細

```
ProtoGoalVector (方向ベクトル):
┌─────────────────────────────────────────────────────────────────┐
│ 入力: 行動履歴、感情変化、判断結果                              │
│                                                                 │
│ VectorGenerator.observe():                                      │
│   1. 入力からシグナル抽出                                       │
│   2. 既存ベクトルとの類似度計算                                 │
│   3. 類似ベクトルがあれば強化、なければ新規作成                 │
│   4. 全ベクトルに自然減衰適用                                   │
│   5. 弱いベクトルは消滅                                         │
│                                                                 │
│ ProtoGoalVector:                                                │
│   vector_id: str                                                │
│   direction: Dict[str, float] (多次元方向)                      │
│   magnitude: float (強度、0.0-1.0)                              │
│   source: VectorSource (生成源情報)                             │
│   created_turn: int                                             │
│   last_reinforced_turn: int                                     │
│                                                                 │
│ 出力: 判断に影響しない「観測データ」として保持                  │
└─────────────────────────────────────────────────────────────────┘

GoalCandidate (目的候補):
┌─────────────────────────────────────────────────────────────────┐
│ 入力: ProtoGoalVectorのリスト                                   │
│                                                                 │
│ CandidateGenerator.observe_vectors(vectors):                    │
│   1. 各ベクトルから確率的に候補生成                             │
│      (generation_probability に基づく)                          │
│   2. 類似候補はマージ                                           │
│   3. 全候補に自然減衰適用                                       │
│   4. 弱い候補は消滅                                             │
│                                                                 │
│ GoalCandidate:                                                  │
│   candidate_id: str                                             │
│   category: CandidateCategory                                   │
│     - APPROACH (接近)                                           │
│     - AVOIDANCE (回避)                                          │
│     - CONNECTION (つながり)                                     │
│     - ISOLATION (孤立)                                          │
│     - EXPRESSION (表現)                                         │
│     - ABSORPTION (吸収)                                         │
│     - EXPLORATION (探索)                                        │
│     - MAINTENANCE (維持)                                        │
│   intensity: float (強度)                                       │
│   source: CandidateSource (生成源ベクトルID)                    │
│                                                                 │
│ 出力: 「白昼夢」として保持、選択されない                        │
└─────────────────────────────────────────────────────────────────┘

TransientGoal (一時的目的):
┌─────────────────────────────────────────────────────────────────┐
│ 入力: GoalCandidateのリスト                                     │
│                                                                 │
│ TransientGoalManager.observe_turn(candidates):                  │
│   1. アクティブ目的がなければ候補から選択可能                   │
│   2. 選択確率は候補のintensityに比例                            │
│   3. 選択時に軽量責任を記録                                     │
│   4. 毎ターン継続確率を評価、自然解除の可能性                   │
│                                                                 │
│ ActiveGoal:                                                     │
│   goal_id: str                                                  │
│   category: CandidateCategory                                   │
│   selected_turn: int                                            │
│   strength: float                                               │
│                                                                 │
│ GoalBias:                                                       │
│   category_boosts: Dict[CandidateCategory, float]               │
│   max_bias: 0.12 (±12%)                                         │
│                                                                 │
│ LightResponsibility:                                            │
│   weight: 0.1 (軽い)                                            │
│   distance: 0.8 (遠い)                                          │
│                                                                 │
│ 出力: 判断にバイアス適用 (±12%)                                 │
└─────────────────────────────────────────────────────────────────┘

ScopedGoal (スコープ目的):
┌─────────────────────────────────────────────────────────────────┐
│ 入力: 現在のTransientGoal、状況                                 │
│                                                                 │
│ ScopedGoalSystem.begin_turn(transient_goal, context):           │
│   1. TransientGoalから今ターンの焦点を決定                      │
│   2. ScopedGoal作成（1ターン限定）                              │
│                                                                 │
│ ScopedGoal:                                                     │
│   scope_id: str                                                 │
│   category: CandidateCategory                                   │
│   direction_alignment: Dict[str, float]                         │
│   status: ScopeStatus (ACTIVE/USED/ABANDONED)                   │
│   action_taken: bool                                            │
│                                                                 │
│ ScopedBias:                                                     │
│   max_bias: 0.08 (±8%)                                          │
│                                                                 │
│ ScopedResponsibility:                                           │
│   weight: 0.05 (とても軽い)                                     │
│   distance: 0.9 (とても遠い)                                    │
│                                                                 │
│ ScopedGoalSystem.end_turn():                                    │
│   ScopedGoalを自動消滅（永続化禁止）                            │
│                                                                 │
│ 出力: 判断にバイアス適用 (±8%)、ターン終了で消滅               │
└─────────────────────────────────────────────────────────────────┘

RepeatedTendency (反復傾向):
┌─────────────────────────────────────────────────────────────────┐
│ 入力: ScopedGoalの使用履歴                                      │
│                                                                 │
│ RepeatedTendencySystem.observe_turn(scoped_goal_used):          │
│   1. ScopedGoalのパターンを記録                                 │
│   2. 類似パターンの反復を検出                                   │
│   3. 反復回数が閾値(3回)超えたら傾向形成                        │
│   4. 使用されない傾向は減衰                                     │
│   5. 連続ミスで減衰加速                                         │
│                                                                 │
│ TendencyPattern:                                                │
│   pattern_id: str                                               │
│   category: CandidateCategory                                   │
│   direction_signature: Dict[str, float]                         │
│                                                                 │
│ Tendency:                                                       │
│   tendency_id: str                                              │
│   pattern: TendencyPattern                                      │
│   strength: float (最大0.15、弱い)                              │
│   confidence: float                                             │
│   consecutive_misses: int                                       │
│                                                                 │
│ TendencyBias:                                                   │
│   max_bias: 0.06 (±6%、とても弱い)                              │
│                                                                 │
│ 出力: 判断にバイアス適用 (±6%)、自然減衰                       │
└─────────────────────────────────────────────────────────────────┘

TendencyAwareness (傾向の自己認知):
┌─────────────────────────────────────────────────────────────────┐
│ 入力: RepeatedTendencySystem                                    │
│                                                                 │
│ observe_tendencies(system) -> TendencyAwareness:                │
│   1. 各傾向を観測                                               │
│   2. 数値を抽象概念に変換                                       │
│      strength → StrengthLevel (NONE/SLIGHT/MODERATE/STRONG)     │
│      duration → DurationLevel (RECENT/ESTABLISHED/PERSISTENT)   │
│      confidence → ConfidenceLevel (UNCERTAIN/FORMING/ESTABLISHED)│
│   3. 認知タイプを判定                                           │
│      HABIT_FORMING / SLIGHT_BIAS / STRONG_HABIT / FADING_HABIT  │
│   4. 人間可読な記述を生成                                       │
│                                                                 │
│ TendencyAwarenessItem:                                          │
│   awareness_type: AwarenessType                                 │
│   category: CandidateCategory                                   │
│   strength_level: StrengthLevel                                 │
│   description: str (例: "I seem to be connecting more lately")  │
│                                                                 │
│ 制約:                                                           │
│   - 判断に影響しない（純粋な観測）                              │
│   - 数値を公開しない（抽象概念のみ）                            │
│   - 自己記述用（SelfReferenceSystemへ接続）                     │
│                                                                 │
│ 出力: SelfReferenceSystem用のタグ生成                           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 判断バイアス統合層

#### 4.6.1 バイアス適用フロー

```
判断候補へのバイアス適用順序:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  候補スコア (初期値: 0.5)                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [1] DecisionBias (decision_bias.py)                     │   │
│  │     - STM残響からバイアス計算                            │   │
│  │     - ダイナミクス相からブースト/抑制                    │   │
│  │     - emotion_biases, valence_bias, residue_intensity   │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [2] SelfReference (self_reference.py)                   │   │
│  │     - 自己参照タグからバイアス調整                       │   │
│  │     - negative_mood → valence調整                        │   │
│  │     - fear_present → intensity調整                       │   │
│  │     - 責任分布タグからの調整                             │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [3] ContextSensitivity (context_sensitivity.py)         │   │
│  │     - 外部文脈から慎重度計算                             │   │
│  │     - リスクの高い候補を抑制                             │   │
│  │     - 最大 ±15% 調整                                     │   │
│  │                                                          │   │
│  │     ExternalContext:                                     │   │
│  │       weight: float (状況の重さ)                         │   │
│  │       density: float (情報密度)                          │   │
│  │       pace: float (進行速度)                             │   │
│  │                                                          │   │
│  │     caution_level → policy_risk → スコア調整            │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [4] StabilityValve (stability_valve.py)                 │   │
│  │     - 極端なスコアを中央に寄せる                         │   │
│  │     - 急激な変化を抑制                                   │   │
│  │     - flatten_scores(): 高すぎ/低すぎを平坦化            │   │
│  │                                                          │   │
│  │     ExtremityIndicators:                                 │   │
│  │       emotion_extremity: float                           │   │
│  │       decision_extremity: float                          │   │
│  │       responsibility_extremity: float                    │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [5] ValueOrientation (value_orientation.py)             │   │
│  │     - 長期価値観からバイアス適用                         │   │
│  │     - 最大 ±5% (とても弱い)                              │   │
│  │     - 5次元抽象軸との整合性でスコア調整                  │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [6] TransientGoal (transient_goal.py)                   │   │
│  │     - 一時的目的からバイアス適用                         │   │
│  │     - 最大 ±12%                                          │   │
│  │     - 目的カテゴリと候補の整合性でスコア調整             │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [7] ScopedGoal (scoped_goal.py)                         │   │
│  │     - スコープ目的からバイアス適用                       │   │
│  │     - 最大 ±8%                                           │   │
│  │     - 今ターンの焦点との整合性でスコア調整               │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [8] RepeatedTendency (repeated_tendency.py)             │   │
│  │     - 反復傾向からバイアス適用                           │   │
│  │     - 最大 ±6% (とても弱い)                              │   │
│  │     - 習慣パターンとの整合性でスコア調整                 │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  最終スコア → 候補選択                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 内省・自己参照層

#### 4.7.1 自己参照ループ

```
自己参照システムの処理フロー:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  各種状態                                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Step 1] acquire_self_reference_targets()               │   │
│  │   入力: psyche_state, responsibility_state, stm, dynamics│   │
│  │   処理: 各状態から値を読み取り（読み取り専用）           │   │
│  │   出力: targets Dict                                     │   │
│  │     - emotions: {joy: 0.7, sadness: 0.2, ...}            │   │
│  │     - mood_valence: 0.3                                  │   │
│  │     - fear_level: 0.2                                    │   │
│  │     - responsibility: {total_weight: 0.5, ...}           │   │
│  │     - short_term_memory: {entry_count: 5, ...}           │   │
│  │     - dynamics: {phase: "peak", ...}                     │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Step 2] summarize_state()                              │   │
│  │   入力: targets Dict                                     │   │
│  │   処理: 粗い要約に圧縮                                   │   │
│  │   出力: summary Dict                                     │   │
│  │     - dominant_emotion: "joy"                            │   │
│  │     - dominant_emotion_value: 0.7                        │   │
│  │     - mood_valence: 0.3                                  │   │
│  │     - fear_level: 0.2                                    │   │
│  │     - ...                                                │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Step 3] generate_self_tags()                           │   │
│  │   入力: summary Dict                                     │   │
│  │   処理: 自己参照タグ生成                                 │   │
│  │   出力: List[SelfTag]                                    │   │
│  │                                                          │   │
│  │   SelfTag:                                               │   │
│  │     category: SelfTagCategory                            │   │
│  │       - EMOTION                                          │   │
│  │       - FEAR                                             │   │
│  │       - RESPONSIBILITY                                   │   │
│  │       - RESPONSIBILITY_DISTRIBUTION                      │   │
│  │       - MEMORY                                           │   │
│  │       - TENDENCY                                         │   │
│  │     label: str (例: "positive_mood", "fear_present")     │   │
│  │     source_value: float                                  │   │
│  │     weight: float                                        │   │
│  │                                                          │   │
│  │   生成されるタグ例:                                      │   │
│  │     - dominant_joy (感情)                                │   │
│  │     - positive_mood (ムード)                             │   │
│  │     - fear_present (恐怖)                                │   │
│  │     - responsibility_present (責任)                      │   │
│  │     - responsibility_near_dominant (責任分布)            │   │
│  │     - memory_active (記憶)                               │   │
│  │     - dynamics_peak (ダイナミクス)                       │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SelfReferenceState                                      │   │
│  │   tags: List[SelfTag]                                    │   │
│  │   reference_count: int (循環カウンタ)                    │   │
│  │                                                          │   │
│  │   使用方法:                                              │   │
│  │     - apply_self_tags_to_bias() でDecisionBiasに適用    │   │
│  │     - get_self_reference_summary() で診断出力           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.2 内省トレース

```
内省トレースの構造:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  判断発生                                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ IntrospectionSystem.generate_trace()                    │   │
│  │                                                          │   │
│  │ TraceLog:                                                │   │
│  │   trace_id: str                                          │   │
│  │   timestamp: float                                       │   │
│  │   turn: int                                              │   │
│  │                                                          │   │
│  │   emotion_snapshot: EmotionSnapshot                      │   │
│  │     - emotions: Dict[str, float]                         │   │
│  │     - dominant: str                                      │   │
│  │     - arousal: float                                     │   │
│  │                                                          │   │
│  │   responsibility_snapshot: ResponsibilitySnapshot        │   │
│  │     - total_weight: float                                │   │
│  │     - pending_count: int                                 │   │
│  │                                                          │   │
│  │   value_orientation_snapshot: ValueOrientationSnapshot   │   │
│  │     - dimensions: Dict[str, float]                       │   │
│  │     - stability: float                                   │   │
│  │                                                          │   │
│  │   decision_snapshot: DecisionSnapshot                    │   │
│  │     - policy: str                                        │   │
│  │     - confidence: float                                  │   │
│  │     - alternatives: List[str]                            │   │
│  │                                                          │   │
│  │   contributing_factors: List[ContributingFactor]         │   │
│  │     ContributingFactor:                                  │   │
│  │       category: FactorCategory                           │   │
│  │         - EMOTION / FEAR / RESPONSIBILITY / MEMORY /     │   │
│  │           VALUE / CONTEXT / STABILITY / GOAL             │   │
│  │       name: str                                          │   │
│  │       direction: InfluenceDirection                      │   │
│  │         - POSITIVE / NEGATIVE / NEUTRAL                  │   │
│  │       weight: float                                      │   │
│  │       description: str                                   │   │
│  │                                                          │   │
│  │   outcome: OutcomeType                                   │   │
│  │     - EXPRESSED / SUPPRESSED / MODIFIED / DELAYED        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  用途:                                                          │
│  - 「なぜこの判断をしたか」の追跡                               │
│  - デバッグ・分析                                               │
│  - 長期統計への入力                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.3 自己観測チェーン

```
自己観測チェーンの処理フロー:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  各種状態入力                                                   │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Layer 1] self_model.py                                │   │
│  │   入力: EmotionVector, ResponsibilityState,            │   │
│  │         RepeatedTendencySystem, VectorState,            │   │
│  │         ValueOrientation                                │   │
│  │   処理: 各状態を抽象Enumに変換（読み取り専用）          │   │
│  │   出力: SelfStateView                                   │   │
│  │     - emotional: EmotionalStateView                     │   │
│  │     - responsibility: ResponsibilityStateView           │   │
│  │     - tendency: TendencyStateView                       │   │
│  │     - direction: DirectionStateView                     │   │
│  │     - value: ValueStateView                             │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Layer 2] temporal_self_difference.py                  │   │
│  │   入力: 現在のSelfStateView + 過去のスナップショット    │   │
│  │   処理: 時間経過による自己状態の変化を検出              │   │
│  │   出力: SelfDifferenceSummary                           │   │
│  │     - magnitude: DifferenceMagnitude (NONE〜SUBSTANTIAL)│   │
│  │     - nature: ChangeNature (STABLE/SHIFTING/TRANSFORMED)│   │
│  │     - component_differences: List[ComponentDifference]  │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Layer 3] continuity_strain.py                         │   │
│  │   入力: SelfDifferenceSummary の履歴                    │   │
│  │   処理: 差分の持続性を観測し「違和感」を検出            │   │
│  │   出力: StrainState                                     │   │
│  │     - level: StrainLevel (AT_EASE〜ALIENATED)           │   │
│  │     - persistence: StrainPersistence (NONE〜CHRONIC)    │   │
│  │     - trend: StrainTrend (STABLE/INCREASING/DECREASING) │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Layer 4] self_image_integration.py                    │   │
│  │   入力: SelfStateView, SelfDifferenceSummary,          │   │
│  │         StrainState, TendencyAwareness                  │   │
│  │   処理: 各観測を統合し暫定的自己像を生成                │   │
│  │   出力: ProvisionalSelfImage                            │   │
│  │     - overall_impression: OverallImpression             │   │
│  │     - stability_feeling: StabilityFeeling               │   │
│  │     - continuity_feeling: ContinuityFeeling             │   │
│  │     - emotional_tone: EmotionalTone                     │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Layer 5] identity_coherence.py                        │   │
│  │   入力: ProvisionalSelfImage, SelfDifferenceSummary,   │   │
│  │         StrainState, TendencyAwareness, ValueOrientation│   │
│  │   処理: 複数シフトの「重なり」から同一性の揺らぎを検出  │   │
│  │   出力: IdentityCoherenceState                          │   │
│  │     - level: CoherenceLevel                             │   │
│  │       STABLE / SLIGHTLY_SHIFTING / UNSETTLED /          │   │
│  │       DISCONNECTED                                      │   │
│  │     - shift_overlap: ShiftOverlap                       │   │
│  │       (6種のシフト源の重なり検出)                        │   │
│  │     - trend: CoherenceTrend                             │   │
│  │       STABLE / CONVERGING / DIVERGING / FLUCTUATING     │   │
│  │                                                          │   │
│  │   シフト検出源 (ShiftSource):                           │   │
│  │     - TEMPORAL_DIFFERENCE: 時間差の持続                  │   │
│  │     - TENDENCY_CHANGE: 傾向の変化                       │   │
│  │     - CONTINUITY_STRAIN: 連続性負荷の持続               │   │
│  │     - VALUE_INSTABILITY: 価値観の不安定                 │   │
│  │     - SELF_IMAGE_FLUX: 自己像の変動                     │   │
│  │     - EMOTIONAL_TURBULENCE: 感情的動揺                  │   │
│  │                                                          │   │
│  │   設計制約:                                              │   │
│  │     - 単一シフトでは状態変化しない（重なりのみ）        │   │
│  │     - 判断・行動への直接影響は禁止                      │   │
│  │     - 自己防衛・自己修復機構を持たない                  │   │
│  │     - 毎ターン再生成（キャッシュなし）                  │   │
│  │     - SelfReferenceSystemへの接続のみ（内省用）         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [Layer 6] self_narrative.py                             │   │
│  │   入力: 感情要約, 記憶要約, 傾向観測,                   │   │
│  │         自己差分観測, 文脈記述（全て読み取り専用）       │   │
│  │   処理: 事実断片を抽象単位へ圧縮し、                    │   │
│  │         時系列の叙述断片へ再構成する                     │   │
│  │   出力: NarrativeState                                  │   │
│  │     - fragments: List[NarrativeFragment]                │   │
│  │       FragmentType: EVENT / REACTION / CONTINUATION /   │   │
│  │                     CHANGE / UNDETERMINED               │   │
│  │     - links: List[FragmentLink]                         │   │
│  │       LinkType: TEMPORAL / THEMATIC / CONTRAST /        │   │
│  │                 CONTINUATION_OF                         │   │
│  │     - coherence: CoherenceInfo                          │   │
│  │       NarrativeCoherence: COHERENT / LOOSELY_CONNECTED /│   │
│  │                           FRAGMENTED / UNDEFINED        │   │
│  │     - trend: NarrativeTrend                             │   │
│  │       ACCUMULATING / CONDENSING / STABLE /              │   │
│  │       DISSOLVING / UNDEFINED                            │   │
│  │                                                          │   │
│  │   時間的性質:                                            │   │
│  │     - 直近ほど鮮明、過去ほど要約化される持続構造        │   │
│  │     - 参照されない断片は自然に減衰（vividness decay）   │   │
│  │     - 後続観測で叙述を再編集可能                        │   │
│  │                                                          │   │
│  │   設計制約:                                              │   │
│  │     - 人格定義・価値付与・信念固定・目標決定を行わない  │   │
│  │     - 物語内容を正誤評価しない                          │   │
│  │     - 物語を根拠に行動を強制しない                      │   │
│  │     - 物語から目標を生成しない                          │   │
│  │     - 単一の「本当の自己」ラベルを固定しない            │   │
│  │     - 接続先は内省記録層と自己記述提示層に限定          │   │
│  │     - 判断選択層、目的層、責任計算層には接続しない      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  全レイヤー共通制約:                                            │
│    - 観測のみ、判断への介入なし                                 │
│    - 数値スコアではなく抽象Enum                                 │
│    - 「正しい自己」を定義しない                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.4 他者モデル

```
他者モデルの処理フロー:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    自己側の観測・反応に偏っている現状に対し、                    │
│    「相手がどう感じているか」を推測する独立層を配置する。        │
│    自己と他者の境界を弱く構造化し、自我形成の前段条件を整える。  │
│    他者モデルは自己像を固定せず、外部に対する                    │
│    「推測の窓口」を用意するだけである。                          │
│                                                                 │
│  入力ソース (3系統)                                             │
│    │                                                            │
│    ├── [Source 1] ExternalContext (外部文脈)                     │
│    │     context_sensitivity.py から受け取る外部文脈情報         │
│    │     Duck typing: pace, weight, density, continuity,        │
│    │                  responsiveness                             │
│    │     → responsiveness高 → "Other party appears engaged"     │
│    │     → responsiveness低 → "Other party appears disengaged"  │
│    │     → weight高 → "Interaction atmosphere feels heavy"      │
│    │     → 中間値 → "Other party state appears neutral"         │
│    │                                                            │
│    ├── [Source 2] ReactionLog (反応ログ)                        │
│    │     short_term_memory.py の StimulusEntry 群               │
│    │     Duck typing: entries[], source_text, intent,           │
│    │                  emotion_label, valence                     │
│    │     → intent="question" → "Other expressed questioning"    │
│    │     → valence正 → "Other party tone appears positive"      │
│    │     → valence負 → "Other party tone appears negative"      │
│    │                                                            │
│    └── [Source 3] SelfState (自己状態 - 対比参照のみ)           │
│          自己の感情状態（intensity, description）               │
│          他者信号との差分を計算 → 対比仮説を生成                │
│          → divergence >= 0.4 → "Contrast detected"             │
│                                                                 │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Extract Phase (3つの抽出関数 - Pure, Duck Typing)       │   │
│  │                                                          │   │
│  │  extract_from_external_context(context)                  │   │
│  │    → list[(description, basis_hint, strength, evidence)] │   │
│  │                                                          │   │
│  │  extract_from_reaction_log(log)                          │   │
│  │    → list[(description, basis_hint, strength, evidence)] │   │
│  │                                                          │   │
│  │  extract_from_self_contrast(self_state, other_signals)   │   │
│  │    → list[(description, basis_hint, strength, evidence)] │   │
│  │                                                          │   │
│  │  全てNone安全、dict/object両対応                         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Hypothesis Generation Phase                              │   │
│  │                                                          │   │
│  │  抽出結果 → OtherStateHypothesis (frozen dataclass)     │   │
│  │    - hypothesis_id: 一意識別子                           │   │
│  │    - source_type: ObservationSourceType                  │   │
│  │      EXTERNAL_CONTEXT / REACTION_LOG /                   │   │
│  │      SELF_CONTRAST / MIXED                               │   │
│  │    - basis: InferenceBasis                               │   │
│  │      BEHAVIORAL / CONTEXTUAL / CONTRAST /                │   │
│  │      COMBINED / UNDEFINED                                │   │
│  │    - description: 推測内容（断定しない表現）             │   │
│  │    - freshness: 0.0〜1.0 (生成時1.0、自然減衰)          │   │
│  │    - strength: 0.0〜1.0 (根拠の安定度)                   │   │
│  │    - reference_count: 参照回数                           │   │
│  │    - evidence_ids: ObservationLink群への参照             │   │
│  │    - competing_ids: 競合仮説IDのリスト                   │   │
│  │    - revision_count: 修正回数                            │   │
│  │    - undetermined_aspects: ("intent_uncertain",          │   │
│  │                             "state_approximate")         │   │
│  │                                                          │   │
│  │  変異メソッド (全て新オブジェクトを返す):                │   │
│  │    with_freshness(), with_strength(), with_reference(),  │   │
│  │    revise(), with_competing()                            │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Competition Detection Phase                              │   │
│  │                                                          │   │
│  │  detect_hypothesis_competitions(hypotheses)              │   │
│  │    - 同source_type + 異basis → 競合候補                 │   │
│  │    - 異basis + description語彙重複(Jaccard>=0.4) → 競合  │   │
│  │    - 競合は許容される（排除しない）                      │   │
│  │    - 仮説同士にcompeting_idsが相互リンクされる           │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Boundary Computation Phase                               │   │
│  │                                                          │   │
│  │  compute_self_other_boundary(self_desc, other_hyps)      │   │
│  │    → SelfOtherBoundary (frozen dataclass)               │   │
│  │      - self_description: 自己側の状態記述               │   │
│  │      - other_description: 他者仮説群の統合記述          │   │
│  │      - divergence: 0.0〜1.0 (語彙非重複率)              │   │
│  │      - boundary_aspects: 差異の側面リスト               │   │
│  │                                                          │   │
│  │  自己と他者の区別を「弱い差分情報」として構造化          │   │
│  │  固定的な境界線ではなく、観測のたびに再計算される        │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Decay & Capacity Phase                                   │   │
│  │                                                          │   │
│  │  freshness減衰:                                          │   │
│  │    new_freshness = freshness - base_decay_rate(0.05)     │   │
│  │                    × ref_modifier(参照多→減衰遅)         │   │
│  │  strength減衰:                                           │   │
│  │    new_strength = strength - strength_decay_rate(0.03)   │   │
│  │                                                          │   │
│  │  除去条件:                                               │   │
│  │    freshness <= stale_threshold(0.15)                    │   │
│  │    AND strength <= min_strength_for_retention(0.05)      │   │
│  │                                                          │   │
│  │  容量制限: max_hypotheses=60                             │   │
│  │    超過時は最も弱い仮説から除去                          │   │
│  │                                                          │   │
│  │  参照ブースト:                                           │   │
│  │    reference_hypothesis(id) →                            │   │
│  │    reference_count+1, freshness+0.10                     │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Output: OtherModelStore (frozen snapshot)                │   │
│  │                                                          │   │
│  │  - hypotheses: tuple[OtherStateHypothesis, ...]         │   │
│  │  - observation_links: tuple[ObservationLink, ...]       │   │
│  │  - boundaries: tuple[SelfOtherBoundary, ...]            │   │
│  │  - total_hypotheses_created: int                         │   │
│  │  - total_revisions: int                                  │   │
│  │  - total_expirations: int                                │   │
│  │  - average_freshness / average_strength: float           │   │
│  │  - active_hypothesis_count: int                          │   │
│  │  - competing_pair_count: int                             │   │
│  │  - boundary_count: int                                   │   │
│  │                                                          │   │
│  │  フィルタ:                                               │   │
│  │    get_active_hypotheses(stale_threshold=0.15)           │   │
│  │    get_strong_hypotheses() (strength > 0.5)              │   │
│  │                                                          │   │
│  │  シリアライゼーション: to_dict() / from_dict()          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  統合関数 (introspection integration):                          │
│    observe_from_chain(system, ctx, log, self_state)             │
│      → observe_other() のラッパー                               │
│    generate_other_model_tags(store, scale)                      │
│      → OTHER_MODEL_COUNT, _STRENGTH, _FRESHNESS,               │
│        _COMPETITION, _BOUNDARY, _INTEGRATED                     │
│    get_other_model_summary(store) → human-readable string       │
│    get_other_model_for_introspection(store) → dict              │
│      - source_distribution, basis_distribution                  │
│      - strongest_hypothesis_description                         │
│                                                                 │
│  設計制約:                                                      │
│    - 他者の意図・価値・信念を断定しない                         │
│    - 正誤や善悪の評価を付与しない                               │
│    - 目的や行動の最適化に結び付けない                           │
│    - 自己像や人格の方向性を固定しない                           │
│    - 候補は仮説として保持し固定しない                           │
│    - 競合する候補を許容する                                     │
│    - 判断選択層・目的生成・価値更新・責任評価に接続しない       │
│                                                                 │
│  検証関数 (テスト支援):                                         │
│    verify_no_decision_impact(store)                             │
│    verify_no_goal_generation(system)                            │
│    verify_read_only_principle(system)                           │
│    verify_no_value_modification(system)                         │
│    verify_no_intent_assertion(system) ← 他者モデル固有          │
│                                                                 │
│  設定 (OtherAgentModelConfig):                                  │
│    max_hypotheses: 60 (自己より少なめ)                          │
│    base_decay_rate: 0.05 (他者推測は不安定→やや速い減衰)       │
│    strength_decay_rate: 0.03                                    │
│    freshness_boost_on_reference: 0.10                           │
│    stale_threshold: 0.15                                        │
│    min_strength_for_retention: 0.05                             │
│    max_evidence_per_hypothesis: 8                               │
│    max_boundaries: 10                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.8 出力制御層

#### 4.8.1 沈黙・トーン制御

```
出力制御の処理フロー:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  判断候補リスト                                                 │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ silence_hesitation.py                                   │   │
│  │                                                          │   │
│  │ generate_candidates_with_silence():                      │   │
│  │   1. 沈黙候補を生成・追加                                │   │
│  │   2. 沈黙スコアを計算                                    │   │
│  │                                                          │   │
│  │ SilenceCandidate:                                        │   │
│  │   type: SilenceType                                      │   │
│  │     - THINKING: 考え中の沈黙 ("...")                     │   │
│  │     - EMOTIONAL: 感情的な沈黙 ("......")                 │   │
│  │     - HESITATION: 躊躇い ("えっと...")                   │   │
│  │     - LISTENING: 聞いている沈黙                          │   │
│  │   duration: float                                        │   │
│  │   expression: str (表現テキスト)                         │   │
│  │                                                          │   │
│  │ 沈黙スコア計算要因:                                      │   │
│  │   - 感情の不安定さ                                       │   │
│  │   - 恐怖レベル                                           │   │
│  │   - 責任の重さ                                           │   │
│  │   - 前回からの時間経過                                   │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ tone.py                                                 │   │
│  │                                                          │   │
│  │ add_tone_to_candidates():                                │   │
│  │   1. 各候補にトーンを付与                                │   │
│  │   2. トーンに基づくスコア調整                            │   │
│  │                                                          │   │
│  │ Tone:                                                    │   │
│  │   SERIOUS: 真剣なトーン                                  │   │
│  │   NEUTRAL: 中立                                          │   │
│  │   LIGHT: 軽いトーン                                      │   │
│  │   PLAYFUL: 遊び心のあるトーン                            │   │
│  │                                                          │   │
│  │ ToneModifier:                                            │   │
│  │   warmth: float (0.0-1.0) ← 温かさ                       │   │
│  │   formality: float (0.0-1.0) ← フォーマル度              │   │
│  │   humor_level: float (0.0-1.0) ← ユーモア度              │   │
│  │                                                          │   │
│  │ トーン選択要因:                                          │   │
│  │   - 現在のムード                                         │   │
│  │   - 感情状態                                             │   │
│  │   - 文脈の重さ                                           │   │
│  │   - 過去のトーン履歴                                     │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ thought.py: select_policy()                             │   │
│  │   最終候補選択                                           │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ expression.py: render_expression()                      │   │
│  │                                                          │   │
│  │ ExpressionOutput:                                        │   │
│  │   text: str (発話テキスト)                               │   │
│  │   emotion: str (感情タグ)                                │   │
│  │   intensity: float (感情強度)                            │   │
│  │   tone: Tone                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. データフロー

### 5.1 1ターンの完全処理フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1ターンの完全処理フロー                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: 入力取得                                                   │   │
│  │                                                                     │   │
│  │   [1.1] 画面キャプチャ                                              │   │
│  │         vision.py: GameCapture.capture_frame()                      │   │
│  │         → PIL.Image (1920x1080等)                                   │   │
│  │                                                                     │   │
│  │   [1.2] 物体検出                                                    │   │
│  │         vision.py: HybridEye.detect_objects()                       │   │
│  │         → [{"name": "person", "position": "center"}, ...]           │   │
│  │                                                                     │   │
│  │   [1.3] 文字認識                                                    │   │
│  │         vision.py: HybridEye.read_text()                            │   │
│  │         → ["ゲームオーバー", "スコア: 1000", ...]                   │   │
│  │                                                                     │   │
│  │   [1.4] センサー情報フォーマット                                    │   │
│  │         vision.py: HybridEye.format_for_prompt()                    │   │
│  │         → "[Vision Sensor Data]\nObjects: ...\nText: ..."           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2: 思考生成                                                   │   │
│  │                                                                     │   │
│  │   [2.1] プロンプト構築                                              │   │
│  │         brain.py: CyreneBrain._build_prompt()                       │   │
│  │         ├─ 画像分析依頼                                             │   │
│  │         ├─ センサー情報 (vision_summary)                            │   │
│  │         ├─ 関連記憶検索 (recall_with_mood, top_k=3)                 │   │
│  │         ├─ 心理状態 (_format_psyche_for_prompt)                     │   │
│  │         │   ├─ 感情サマリー                                         │   │
│  │         │   ├─ ムード (valence, arousal)                            │   │
│  │         │   ├─ ドライブ (social, curiosity, expression)             │   │
│  │         │   └─ 恐怖サマリー                                         │   │
│  │         └─ 感情タグルール                                           │   │
│  │                                                                     │   │
│  │   [2.2] Gemini API呼び出し                                          │   │
│  │         brain.py: chat.send_message([prompt, image])                │   │
│  │         → "[happy] わぁ、すごい！このゲーム面白そう♪"               │   │
│  │                                                                     │   │
│  │   [2.3] 応答分割                                                    │   │
│  │         分割位置: 。！？!?\n♪♥♡★☆                                   │   │
│  │         → ["わぁ、すごい！", "このゲーム面白そう♪"]                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 3: 心理状態更新                                               │   │
│  │                                                                     │   │
│  │   [3.1] 感情タグ抽出                                                │   │
│  │         main.py: parse_emotion_tag()                                │   │
│  │         "[happy] わぁ..." → emotion="happy", sw=3.5                 │   │
│  │                                                                     │   │
│  │   [3.2] バレンス変換                                                │   │
│  │         brain.py: _TAG_VALENCE["happy"] → 0.7                       │   │
│  │                                                                     │   │
│  │   [3.3] Percept生成                                                 │   │
│  │         psyche/state.py: Percept(emotion="happy", valence=0.7)      │   │
│  │                                                                     │   │
│  │   [3.4] 感情・ムード更新                                            │   │
│  │         psyche/reaction.py: react(percept, psyche_state, delta)     │   │
│  │         ├─ 感情更新                                                 │   │
│  │         ├─ 自然減衰                                                 │   │
│  │         ├─ ドライブ更新                                             │   │
│  │         └─ ムードドリフト                                           │   │
│  │                                                                     │   │
│  │   [3.5] 愛着更新                                                    │   │
│  │         psyche/attachment_manager.py: update_bond()                 │   │
│  │                                                                     │   │
│  │   [3.6] 恐怖指数再計算                                              │   │
│  │         brain.py: _recompute_fear()                                 │   │
│  │         → compute_fear_index(4柱リスク)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 4: 音声合成・出力                                             │   │
│  │                                                                     │   │
│  │   [4.1] テキスト分割                                                │   │
│  │         voice.py: VoiceClient._split_text()                         │   │
│  │         → 50文字以下のチャンクに分割                                │   │
│  │                                                                     │   │
│  │   [4.2] 音声合成                                                    │   │
│  │         voice.py: VoiceClient._synthesize_chunk()                   │   │
│  │         → Style-Bert-VITS2 API呼び出し                              │   │
│  │         → WAVバイナリ取得                                           │   │
│  │                                                                     │   │
│  │   [4.3] WAV結合                                                     │   │
│  │         voice.py: VoiceClient._combine_wav_data()                   │   │
│  │                                                                     │   │
│  │   [4.4] 音声再生                                                    │   │
│  │         voice.py: VoiceClient._play_audio()                         │   │
│  │         → sounddevice.play() + wait()                               │   │
│  │                                                                     │   │
│  │   [4.5] Warudo送信 (オプション)                                     │   │
│  │         Base64 WAV → WebSocket → Warudo                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 5: 後処理                                                     │   │
│  │                                                                     │   │
│  │   [5.1] 会話ログ記録                                                │   │
│  │         brain.py: _conversation_log.append()                        │   │
│  │                                                                     │   │
│  │   [5.2] 長期記憶保存 (5ターンごと)                                  │   │
│  │         brain.py: summarize_and_save()                              │   │
│  │         → Geminiで要約生成                                          │   │
│  │         → MemoryManager.add_memory()                                │   │
│  │                                                                     │   │
│  │   [5.3] ループ遅延                                                  │   │
│  │         main.py: asyncio.sleep(0.1)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. モジュール間連携

### 6.1 依存関係マトリクス

```
依存関係: 行が列に依存する (行 → 列)

                          state pillars fear reaction memory emotion dynamics decision self_ref goal resp output
state.py                    -     ○      -      -       -       -        -        -        -      -     -      -
pillars.py                  ○     -      -      -       -       -        -        -        -      -     -      -
fear.py                     ○     ○      -      -       -       -        -        -        -      -     -      -
reaction.py                 ○     -      ○      -       -       ○        -        -        -      -     -      -
short_term_memory.py        -     -      -      -       -       -        -        -        -      -     -      -
multi_emotion.py            ○     -      -      -       -       -        -        -        -      -     -      -
dynamics.py                 ○     -      -      -       -       ○        -        -        -      -     -      -
decision_bias.py            -     -      -      -       ○       -        ○        -        -      -     -      -
self_reference.py           ○     -      -      -       ○       -        ○        ○        -      -     ○      -
value_orientation.py        -     -      -      -       -       -        -        -        -      -     -      -
proto_goal_vector.py        -     -      -      -       -       -        -        -        -      -     -      -
goal_candidates.py          -     -      -      -       -       -        -        -        -      ○     -      -
transient_goal.py           -     -      -      -       -       -        -        -        -      ○     -      -
scoped_goal.py              -     -      -      -       -       -        -        -        -      ○     -      -
repeated_tendency.py        -     -      -      -       -       -        -        -        -      ○     -      -
tendency_awareness.py       -     -      -      -       -       -        -        -        -      ○     -      -
self_model.py               -     -      -      -       -       ○        -        -        ○      ○     ○      -
temporal_self_difference.py -     -      -      -       -       -        -        -        -      -     -      -
continuity_strain.py        -     -      -      -       -       -        -        -        -      -     -      -
self_image_integration.py   -     -      -      -       -       -        -        -        -      -     -      -
identity_coherence.py       -     -      -      -       -       -        -        -        -      ○     -      -
self_narrative.py           -     -      -      -       -       -        -        -        -      -     -      -
other_agent_model.py       -     -      -      -       -       -        -        -        -      -     -      -
responsibility.py           -     -      -      -       -       -        -        -        -      -     -      -
responsibility_dispersion.py-     -      -      -       -       -        -        -        -      -     ○      -
context_sensitivity.py      -     -      -      -       -       -        -        ○        -      -     -      -
stability_valve.py          -     -      -      -       -       -        -        -        -      -     -      -
silence_hesitation.py       ○     -      -      -       -       -        -        -        -      -     -      -
tone.py                     ○     -      -      -       -       -        -        -        -      -     -      -
thought.py                  ○     -      -      -       -       -        -        -        -      -     -      -
expression.py               ○     -      -      -       -       -        -        -        -      -     -      -
introspection_trace.py      ○     -      -      -       -       -        -        -        ○      ○     ○      -
long_term_dynamics.py       -     -      -      -       -       -        -        -        -      ○     ○      -
```

### 6.2 データ流通図

```
主要データの流れ:

EmotionVector:
  state.py → reaction.py → multi_emotion.py → dynamics.py
           → decision_bias.py → self_reference.py → introspection_trace.py

Mood:
  state.py → reaction.py → memory_link.py (recall_with_mood)
           → decision_bias.py → tone.py

FearIndex:
  pillars.py → fear.py → state.py → reaction.py
             → decision_bias.py → silence_hesitation.py

ShortTermMemory:
  short_term_memory.py → short_term_loop.py → stm_emotion_coupling.py
                       → decision_bias.py → self_reference.py

ResponsibilityState:
  responsibility.py → responsibility_dispersion.py → self_reference.py
                    → introspection_trace.py

GoalCandidate:
  proto_goal_vector.py → goal_candidates.py → transient_goal.py
                       → scoped_goal.py → repeated_tendency.py
                       → tendency_awareness.py → self_model.py
                       → temporal_self_difference.py → continuity_strain.py
                       → self_reference.py / introspection_trace.py

SelfStateView (自己状態の流れ):
  self_model.py → temporal_self_difference.py → continuity_strain.py
                → self_image_integration.py → self_reference.py (introspection only)

ProvisionalSelfImage (暫定的自己像):
  self_model.py (SelfStateView) ─┐
  tendency_awareness.py ─────────┼→ self_image_integration.py → self_reference.py
  temporal_self_difference.py ───┤   (generates ProvisionalSelfImage)
  continuity_strain.py ──────────┘

IdentityCoherence (自己同一性の揺らぎ認知):
  self_image_integration.py ─────┐
  temporal_self_difference.py ───┤
  continuity_strain.py ──────────┼→ identity_coherence.py → self_reference.py
  tendency_awareness.py ─────────┤   (generates IdentityCoherenceState)
  value_orientation.py ──────────┘   (introspection only, NO decision impact)

NarrativeState (自己物語形成):
  感情要約 ──────────────────┐
  記憶要約 ──────────────────┤
  傾向観測 ──────────────────┼→ self_narrative.py → 内省記録層
  自己差分観測 ──────────────┤   (generates NarrativeState)   → 自己記述提示層
  文脈記述 ──────────────────┘   (observation only, NO decision impact)
  入力は全て読み取り専用
  接続先: 内省記録層・自己記述提示層に限定
  非接続: 判断選択層・目的層・責任計算層・価値更新層

EpisodicMemory (エピソード記憶 - 自伝的記憶):
  短期記憶(STM) ───────────┐
  感情状態要約 ─────────────┤
  自己差分観測 ─────────────┼→ episodic_memory.py → 内省記録層
  傾向認知 ─────────────────┤   (generates EpisodeStore)    → 長期記憶検索入口
  自己同一性状態 ───────────┤   (observation only, NO decision impact)
  自己物語状態 ─────────────┤
  外部文脈 ─────────────────┘
  出来事単位の経験を保持し、感情・自己観測を随伴情報として付与
  重要度・参照頻度により自然減衰、圧縮を許容
  解釈は固定しない（再要約・再関連付けを許容）
  非接続: 判断選択層・目的生成・価値更新・責任評価

ExpectationFormation (予期・期待の形成):
  反復傾向バイアス ───────────┐
  自己差分サマリ ─────────────┼→ expectation_formation.py → 内省記録層
  自己物語状態 ───────────────┘   (generates ExpectationStore)            → 記憶参照入口
  過去の反復・差分・物語から「次に起きうる展開の仮の見通し」を弱く生成
  予期は短〜中期で自然減衰、参照で鮮度回復、修正・撤回が可能
  予期同士の競合を許容する（正解化・評価化しない）
  非接続: 判断選択層・目的生成・価値更新・責任評価

OtherAgentModel (他者モデル):
  外部文脈（ExternalContext） ──────┐
  反応ログ（STM/ReactionLog） ─────┼→ other_agent_model.py → 内省記録層
  自己状態（対比参照のみ） ────────┘   (generates OtherModelStore)    → 記憶参照補助
  「相手がどう感じているか」の推測を仮説として弱く保持
  入力3系統:
    [ExternalContext] → responsiveness/weight/pace → 行動的/文脈的仮説
    [ReactionLog]     → intent/valence/emotion    → 行動的仮説
    [SelfState対比]   → intensity差分             → 対比仮説
  処理フロー:
    Extract(3関数) → Hypothesis生成 → 競合検出(Jaccard)
                   → Boundary計算 → Decay適用 → Snapshot
  内部構造:
    OtherStateHypothesis: 仮説（basis=BEHAVIORAL/CONTEXTUAL/CONTRAST）
    ObservationLink: 観測と仮説の弱い接続（contribution 0.0〜1.0）
    SelfOtherBoundary: 自己/他者の乖離度（divergence 0.0〜1.0）
  ライフサイクル:
    生成時 freshness=1.0 → base_decay_rate=0.05/ターン で減衰
    参照時 freshness+0.10ブースト, reference_count+1
    修正可能（revise）, 競合許容（competing_ids相互リンク）
    stale_threshold(0.15) AND min_strength(0.05) 以下で自然消滅
  容量: max_hypotheses=60, max_boundaries=10
  タグ出力: OTHER_MODEL_COUNT / _STRENGTH / _FRESHNESS /
            _COMPETITION / _BOUNDARY / _INTEGRATED
  固有検証: verify_no_intent_assertion（意図断定メソッド禁止）
  非接続: 判断選択層・目的生成・価値更新・責任評価

IntrospectionConsumption (内省の消費層):
  内省ログ要約 ─────────────┐
  自己物語状態 ─────────────┤
  自己同一性状態 ───────────┼→ introspection_consumption.py → 内省記録層
  傾向認知 ─────────────────┤   (generates ConsumptionStore)            → 記憶参照入口
  エピソード記憶 ───────────┘   (observation only, NO decision impact)
  内省観測を「読み取り可能な断片」に再編成し叙述素材として循環
  断片は中期的に残るが自然減衰を許容、参照で鮮度回復
  断片の束ね方は固定しない（再要約・再リンクを許容）
  非接続: 判断選択層・目的生成・価値更新・責任評価

DecisionBias:
  decision_bias.py → context_sensitivity.py → stability_valve.py
                   → value_orientation.py → transient_goal.py
                   → scoped_goal.py → repeated_tendency.py
                   → thought.py (select_policy)
```

---

## 7. 設計原則

### 7.1 コア設計原則

| 原則 | 説明 | 適用例 |
|-----|------|-------|
| **弱い影響** | バイアスは常に小さく、選択肢を排除しない | 最大バイアス: ValueOrientation ±5%, TransientGoal ±12% |
| **坂と壁** | 「壁」(must)ではなく「坂」(easier)を作る | 全てのバイアスは確率的な傾きを作るだけ |
| **ゴーストデータ** | 観測のみで判断に影響しない層を持つ | ProtoGoalVector, GoalCandidate |
| **自然減衰** | 明示的リセット不要、使わなければ消える | STMエントリ、傾向、目的候補すべて |
| **軽量責任** | 責任は発生するが重くない | weight=0.05-0.1, distance=0.8-0.9 |
| **永続化禁止** | 短期状態は保存しない | ScopedGoal（1ターンで消滅） |
| **成功/失敗なし** | 結果の評価・判定を行わない | 目的システム全般 |
| **抽象概念** | 数値を人間的な概念に変換 | TendencyAwareness: SLIGHT/MODERATE/STRONG |
| **保存則** | 責任の総重量は変換しても保存 | responsibility_dispersion |
| **高慣性** | 長期状態は非常にゆっくり変化 | ValueOrientation: 1回で~0.1%変化 |

### 7.2 バイアス強度制限

| システム | 最大バイアス | 用途 |
|---------|------------|------|
| ContextSensitivity | ±15% | 外部文脈によるリスク調整 |
| TransientGoal | ±12% | 一時的目的によるバイアス |
| ScopedGoal | ±8% | 今ターンの焦点によるバイアス |
| RepeatedTendency | ±6% | 習慣によるバイアス |
| ValueOrientation | ±5% | 長期価値観によるバイアス |

### 7.3 責任の軽量化

| パラメータ | TransientGoal | ScopedGoal | 説明 |
|-----------|---------------|------------|------|
| weight | 0.1 | 0.05 | 責任の重さ（低いほど軽い） |
| distance | 0.8 | 0.9 | 心理的距離（高いほど遠い） |

### 7.4 時間軸と永続性

| 時間軸 | モジュール | 永続化 | 寿命 |
|-------|-----------|-------|------|
| 長期 | ValueOrientation | ○ | 数百〜数千ターン |
| 長期 | LongTermDynamics | ○ | 無期限 |
| 中期 | ProtoGoalVector | ○ | 数十ターン（減衰） |
| 中期 | GoalCandidate | ○ | 数十ターン（減衰） |
| 中期 | RepeatedTendency | ○ | 数十ターン（減衰） |
| 短期 | TransientGoal | ○ | 数ターン |
| 短期 | ShortTermMemory | ○ | 数ターン（減衰） |
| 1ターン | ScopedGoal | × | 1ターン（永続化禁止） |
| 即時 | DecisionBias | × | 判断時のみ |

---

## 付録: ファイル一覧

### Psycheモジュール (psyche/)

```
psyche/
├── __init__.py                    (1063行) - エクスポート定義
├── state.py                       (258行)  - 心理状態データ構造
├── pillars.py                     (76行)   - 4柱状態定義
├── fear.py                        (76行)   - 恐怖指数計算
├── identity_manager.py            (90行)   - アイデンティティ管理
├── attachment_manager.py          (95行)   - 愛着管理
├── continuity_manager.py          (95行)   - 連続性管理
├── projection_manager.py          (89行)   - 未来投射管理
├── perception.py                  (157行)  - 知覚処理
├── reaction.py                    (201行)  - 反応処理
├── memory_link.py                 (101行)  - 記憶検索
├── multi_emotion.py               (495行)  - 複数感情独立管理
├── emotion_amplitude.py           (362行)  - 感情振幅調整
├── dynamics.py                    (474行)  - 感情ダイナミクス相
├── stm_emotion_coupling.py        (604行)  - 短期記憶-感情連携
├── short_term_memory.py           (399行)  - 短期記憶管理
├── short_term_loop.py             (432行)  - 短期感情ループ
├── reaction_with_stm.py           (294行)  - STM統合反応
├── decision_bias.py               (465行)  - 判断バイアス計算
├── context_sensitivity.py         (754行)  - 外部文脈感受性
├── stability_valve.py             (728行)  - 極端回避バルブ
├── self_reference.py              (923行)  - 自己参照ループ
├── introspection_trace.py         (864行)  - 内省ログ生成
├── long_term_dynamics.py          (882行)  - 長期統計観測
├── value_orientation.py           (746行)  - 長期価値観
├── proto_goal_vector.py           (774行)  - 方向ベクトル（ゴースト）
├── goal_candidates.py             (929行)  - 目的候補（白昼夢）
├── transient_goal.py              (812行)  - 一時的目的選択
├── scoped_goal.py                 (660行)  - スコープ目的（1ターン）
├── repeated_tendency.py           (858行)  - 反復傾向（習慣）
├── tendency_awareness.py          (651行)  - 傾向の自己認知
├── self_model.py                  (1601行) - 自己状態統合モデル
├── temporal_self_difference.py    (1320行) - 自己モデル差分認知
├── continuity_strain.py          (939行)  - 自己連続性負荷
├── self_image_integration.py     (1184行) - 自己像統合
├── identity_coherence.py         (1110行) - 自己同一性の揺らぎ認知
├── self_narrative.py             (1491行) - 自己物語形成（非規範・観測型）
├── episodic_memory.py           (1709行) - エピソード記憶（自伝的記憶）
├── introspection_consumption.py (1455行) - 内省の消費層（読み取り可能断片の循環）
├── expectation_formation.py    (1485行) - 予期・期待の形成（未来方向の連続性投射）
├── other_agent_model.py        (1603行) - 他者モデル（他者状態の仮説的推測）
├── responsibility.py              (480行)  - 責任記録・評価
├── responsibility_manager.py      (210行)  - 責任マネージャー
├── responsibility_dispersion.py   (1039行) - 責任の発散・昇華
├── silence_hesitation.py          (724行)  - 沈黙・躊躇い表現
├── tone.py                        (698行)  - トーン・ユーモア制御
├── thought.py                     (293行)  - 思考候補生成・選択
├── expression.py                  (156行)  - 表現生成
├── snapshot.py                    (239行)  - スナップショット管理
└── persistence.py                 (395行)  - 永続化システム
```

### テストファイル (tests/)

```
tests/
├── conftest.py                    (161行)
├── test_decision_bias.py          (487行)
├── test_dynamics.py               (410行)
├── test_emotion_amplitude.py      (383行)
├── test_goal_candidates.py        (810行)
├── test_integration_flow.py       (190行)
├── test_introspection_trace.py    (657行)
├── test_long_term_dynamics.py     (636行)
├── test_memory.py                 (102行)
├── test_multi_emotion.py          (603行)
├── test_persistence.py            (370行)
├── test_proto_goal_vector.py      (876行)
├── test_psyche_flow.py            (310行)
├── test_repeated_tendency.py      (732行)
├── test_responsibility.py         (534行)
├── test_responsibility_dispersion.py (809行)
├── test_scoped_goal.py            (670行)
├── test_self_reference.py         (858行)
├── test_short_term_loop.py        (397行)
├── test_silence_hesitation.py     (615行)
├── test_stability_valve.py        (674行)
├── test_state_update.py           (129行)
├── test_stm_emotion_coupling.py   (678行)
├── test_tendency_awareness.py     (644行)
├── test_self_model.py             (1164行)
├── test_temporal_self_difference.py (909行)
├── test_continuity_strain.py     (908行)
├── test_self_image_integration.py (907行)
├── test_identity_coherence.py    (994行)
├── test_self_narrative.py        (1133行)
├── test_episodic_memory.py      (1249行)
├── test_introspection_consumption.py (1023行)
├── test_expectation_formation.py (1075行)
├── test_other_agent_model.py    (1205行)
├── test_tone.py                   (592行)
├── test_transient_goal.py         (664行)
├── test_value_orientation.py      (599行)
└── test_context_sensitivity.py    (704行)
```

---

## 8. 今後の実装候補

現在の構造から自然に要請される機能のアイデア（未設計・未実装）。

| # | 候補名 | 概要 | 要請元 | 状態 |
|---|--------|------|--------|------|
| 1 | 自己物語 ↔ 自己観測チェーン統合 | self_narrativeの入力に自己観測チェーンの出力を接続する | self_narrative, self_model, temporal_self_difference, tendency_awareness | 完了 |
| 2 | エピソード記憶（自伝的記憶） | 「あのとき何が起き、どう感じたか」を個別の出来事として保持する構造。short_term_memoryは残留のみ、long_term_dynamicsは統計のみで、個別体験の蓄積がない | short_term_memory, self_narrative | 完了 |
| 3 | 内省の消費層 | introspection_trace, self_narrative, identity_coherenceの観測結果を「読んで自分について語る」層。内省は生成されるが消費先がない | introspection_trace, self_narrative, identity_coherence | 完了 |
| 4 | 予期・期待の形成 | 過去の傾向や経験から「次に何が起きそうか」を予測する構造。時間的連続性は過去方向のみで、未来方向の投射が弱い | repeated_tendency, temporal_self_difference, self_narrative | 完了 |
| 5 | 他者モデル | 「相手（視聴者）がどう感じているか」の推測構造。自己と他者の境界が構造として存在しない | context_sensitivity, self_model | 未着手 |
| 6 | 感情記憶の紐づけ | 特定の記憶に感情が染み付く仕組み。stm_emotion_couplingは短期の連動のみ | stm_emotion_coupling, short_term_memory | 未着手 |
| 7 | 自発的内的動機 | 感情や傾向から欲求が湧き上がる構造。goal系は候補生成と選択の仕組みだが「なぜそれをしたいか」の動機源がない | proto_goal_vector, repeated_tendency, multi_emotion | 未着手 |

---

*このドキュメントはCyrene AI システムの完全な技術仕様書です。*
*総コード行数: ~59,200行 / テスト数: 1,678*
