# Cyrene AI  - 完全システムアーキテクチャ仕様書

作成日: 2026-02-09
更新日: 2026-02-26
総コード行数: ~202,000行
総テスト数: 8,506テスト

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
│   │  (393行)  │    │  (941行)  │    │(53,158行) │    │  (437行)  │         │
│   └───────────┘    └───────────┘    └───────────┘    └───────────┘         │
│         │                │                │                │                │
│    dxcam/YOLO       Gemini API      心理処理        Style-Bert-VITS2       │
│    /EasyOCR                                                                 │
│                                                                             │
│                    ┌───────────┐                                            │
│                    │   main    │ ← メインループ制御                         │
│                    │  (299行)  │                                            │
│                    └───────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 技術スタック

| レイヤー | 技術 | 用途 |
|---------|------|------|
| 画面キャプチャ | dxcam | GPU直接アクセス高速キャプチャ |
| 物体検出 | YOLOv8n | リアルタイムオブジェクト認識 |
| 文字認識 | EasyOCR | 日本語+英語テキスト抽出 |
| 思考生成 | Gemini 3 Flash Preview | マルチモーダルAI推論 |
| 心理処理 | Python (自作) | 感情・判断・記憶システム |
| 音声合成 | Style-Bert-VITS2 | 高品質日本語TTS |
| 外部出力 | Warudo | 3Dレンダリング（オプション） |

---

## 2. コード統計

### 2.1 ディレクトリ別コード行数

| ディレクトリ | ファイル数 | 総行数 | 説明 |
|-------------|-----------|--------|------|
| psyche/ | 69 | 55,541 | 心理システム本体（orchestrator.py含む） |
| tests/ | 68 | 54,190 | 自動テストコード |
| src/ | 14 | 2,655 | 補助モジュール |
| tools/ | 2 | 418 | 長期シミュレーション等 |
| ルート | 6 | 2,580 | コアシステム |
| **合計** | **153** | **110,078** | |

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
| 10a | other_model_input_supply.py | 308 | 30 | 内省 | 他者モデル入力供給（external_context / reaction_log 生成） |
| 11 | emotional_memory_binding.py | 1,708 | 114 | 記憶 | 感情記憶の紐づけ（中長期感情痕跡） |
| 12 | intrinsic_motivation.py | 1,752 | 113 | 動機 | 自発的内的動機（感情・傾向由来の内的推進力） |
| 13 | responsibility_dispersion.py | 1,039 | 48 | 責任 | 責任の発散・昇華・時間分配 |
| 13a | policy_candidate_expansion.py | 1,388 | 86 | 判断 | ポリシー候補拡張（8断面×10軸、内面反映経路の増設） |
| 13b | memory_system_integration.py | 1,132 | 93 | 記憶 | 記憶系統統合（episodic↔long_term↔binding正規化、重複並立・競合併存・出所多様性） |
| 13c | other_model_real_feed.py | 1,481 | 102 | 内省 | 他者モデルリアルフィード統合（8観測断片抽出・正規化・競合併存・鮮度管理・安全弁） |
| 13d | text_dialogue_input.py | 1,559 | 102 | 入力 | テキスト対話入力経路（6段パイプライン・経路多様性・重複抑制・安全弁） |
| 13e | spontaneous_activation.py | 1,549 | 84 | 起動 | 自発起動経路（8断面交差・5段パイプライン・競合並立・安全弁） |
| 13f | value_orientation_validation.py | 1,211 | 88 | 検証 | 価値方向性実運用検証（8断面・6段パイプライン・差分並立・安全弁） |
| 13g | memory_forgetting_fixation.py | 1,052 | 85 | 記憶 | 記憶の忘却と固定化（8断面・6段パイプライン・段階忘却・復帰経路・安全弁） |
| 13h | action_result_observation.py | 1,638 | 128 | 観測 | 行動-結果の観測と蓄積（8断面・6段パイプライン・非正誤判定・時系列隣接記録・入力経路ラベル併記・安全弁4種） |
| 13i | other_model_dialogue_learning.py | 1,625 | 135 | 内省 | 他者観測の長期蓄積と仮説補助（8断面・8段パイプライン・相手別分離・仮説再生成方式・安全弁4種） |
| 13j | meta_emotion_cognition.py | 1,628 | 155 | 感情 | メタ感情認知と変動候補生成（8断面・7段パイプライン・常時等価列挙・Phase 1-2不変性保証・境界値到達記述追加・安全弁4種） |
| 13k | self_action_perception.py | 395 | 114 | 知覚 | 自己行動知覚（3段パイプライン・全記録等価・テキスト非解釈・判断系非接続・brain.py通知経路） |
| 13l | intent_action_gap.py | 397 | 129 | 知覚 | 意図-行動間の乖離認知（3段パイプライン・対構成→多断面記述→蓄積参照・全記録等価・パターン抽出禁止・3経路遮断・安全弁5種） |
| 13m | temporal_cognition.py | 809 | 212 | 知覚 | 時間認知構造（3段パイプライン・経過蓄積→8断面特徴量記述→参照提供・スライディングウィンドウ・段階値列挙型・帯域キャッシュ鮮度+入力経路間隔断面追加・パターン抽出禁止・4経路遮断・安全弁5種） |
| 13n | multi_path_recall.py | 807 | 105 | 記憶 | 記憶の多経路想起（3経路想起・感情連想/文脈連想/時間近接・経路等価性・顕著性バイアス抑制・ルーミネーション防止・忘却分離・安全弁5種） |
| 13o | introspection_cross_section.py | 731 | 130 | 内省 | 内省断面間の横断的記述（3段パイプライン・6断面並置・ウィンドウ25件（enrichment10件）・パターン抽出禁止・統合禁止・全断面等価・5経路遮断・安全弁5種） |
| 13p | perceptual_context.py | 646 | 116 | 知覚 | 知覚入力の内部文脈化（3段パイプライン・4断面段階値列挙型・感情変化頻度/意図変化頻度/話題重複度/感情価推移方向・テキスト比較禁止・4経路遮断・安全弁7種） |
| 13q | scoring_fluctuation.py | 647 | テスト内 | 判断 | スコアリングの構造的揺らぎ（5段パイプライン・内部状態由来の非決定性・感情/STM/drives/経過時間から変動度導出・振幅上限<ValueOrientation・状態蓄積なし・安全弁5種） |
| 13r | selection_attribution.py | 413 | 87 | 知覚 | 選択帰属（選択事実のREAD-ONLY記録・候補群構成+選択ラベル+バイアス源構成蓄積・全記録等価・パターン抽出禁止・5経路遮断・enrichment等価列挙（バイアス情報遮断）・安全弁5種） |
| 13s | reference_frequency_description.py | 822 | 106 | 内省 | 参照頻度の構造的記述（15箇所横断読み取り・断面構成・集中度/偏在度記述・変動記述・FIFO断面履歴・enrichment直接露出遮断・忘却経路遮断・想起経路遮断・multi_path_recall/spontaneous_recall追加・安全弁5種） |
| 13t | persistent_commitment.py | 1,037 | テスト内 | 目標 | 持続的取り組み保持（transient_goal昇格が唯一生成経路・複数並行保持・強度依存非線形減衰・慣性時間減衰・4解除条件・認知記録FIFO・資源競合・バイアス上限<VO・安全弁6種・自己強化ループ4重遮断） |
| 13u | behavioral_diversity_description.py | 664 | テスト内 | 内省 | 行動多様性の構造的記述（3断面横断読み取り・結果断面キー種類数/選択ラベル種類数/候補群サイズ分散度・段階値列挙型・FIFO蓄積・enrichment直接露出遮断・頻度情報構造的排除・安全弁8種） |
| 13v | spontaneous_recall.py | 1,025 | テスト内 | 記憶 | 記憶の自発的想起（4段パイプライン・3経路想起・感情変動連想/動機連想/揺らぎ連想・外部入力非依存・ルーミネーション防止・経路等価性・multi_path_recall経路分離・安全弁7種） |
| 13w | internal_contradiction_description.py | 787 | テスト内 | 内省 | 内部状態の矛盾並置記述（5段パイプライン・6断面対定義・数値的乖離検出・矛盾解消禁止・全記録等価・収束監視・evaluative語彙除去・安全弁7種） |
| 13x | interaction_accumulation.py | 678 | テスト内 | 相互作用 | 相互作用の蓄積記述（4段パイプライン・時間的隣接対構成・因果帰属禁止・全記録等価・FIFO自然消失・ルーミネーション防止・パターン抽出排除・安全弁5種） |
| 13y | emotional_backdrop_cognition.py | 989 | テスト内 | 感情 | 感情基調の持続認知（4段パイプライン・スライディングウィンドウ・8断面入力・段階的鮮度減衰・等価列挙のみ・移動平均禁止・パターン判定禁止・安全弁5種・経路遮断5種） |
| 13z | situational_self_presentation.py | 901 | テスト内 | 自己認知 | 状況依存的自己呈示の認知（3段パイプライン・相手別分離蓄積・種類数段階値・鮮度減衰・マッピング形成禁止・パターン抽出禁止・安全弁8種） |
| 13aa | introspection_longitudinal_view.py | 506 | テスト内 | 内省 | 内省の時間的縦断参照（3段パイプライン・横断→縦断変換・独自状態なし・全断面等価・全時点等価・パターン抽出禁止・安全弁5種） |
| 13ab | drive_variation_description.py | 1,079 | テスト内 | 駆動 | 駆動の変動記述（4段パイプライン・スライディングウィンドウ・8断面入力・段階的鮮度減衰・等価列挙のみ・移動平均禁止・安全弁5種・経路遮断6種） |
| 13ac | expectation_lifecycle_description.py | 904 | テスト内 | 予期 | 予期の成立・消失の事後記述（スナップショット比較・5状態遷移検出・FIFO蓄積・均一減衰・因果帰属禁止・統計量算出禁止・安全弁5種） |
| 13ad | input_pathway_balance.py | 813 | テスト内 | 入力 | 入力経路間の均衡記述（3経路横断読み取り・窓内カウント・段階値列挙型・FIFO蓄積・規範なし事実記述・パターン抽出禁止・安全弁5種） |
| 13ae | responsibility_temporal_trace.py | 653 | テスト内 | 責任 | 責任の時間的推移記述（スナップショット蓄積・段階値記述・FIFO・責任分散操作非介入・パターン抽出禁止・安全弁5種） |
| 13af | emotion_cooccurrence_description.py | 745 | テスト内 | 感情 | 感情間の共起記述（同時存在事実記録・種類のみ・頻度記録禁止・評価的判定禁止・FIFO・安全弁5種） |
| 13ag | other_boundary_accumulation.py | 1,043 | テスト内 | 他者認知 | 他者境界の多相蓄積（相手別分離・自他境界変動記述・FIFO・鮮度減衰・境界制御禁止・安全弁5種） |
| 13ah | forgetting_recall_balance.py | 729 | テスト内 | 記憶 | 忘却と想起の均衡記述（忘却速度/想起頻度の事実記述・窓内カウント・規範なし・パラメータ非介入・安全弁5種） |
| 13ai | attention_distribution_description.py | 920 | テスト内 | 自己認知 | 注意配分の構造的記述（処理帯域集中/分散記述・横断読み取り・段階値・FIFO・帯域制御禁止・安全弁5種） |
| 13aj | goal_hierarchy_propagation.py | 1,083 | テスト内 | 目標 | 目的階層間の隣接状態変化記述（3層限定・6段パイプライン・スナップショット比較・因果帰属禁止・enrichment直接露出遮断・安全弁7種） |
| 13ak | hypothesis_observation_pairing.py | 1,010 | テスト内 | 他者認知 | 仮説-観測の隣接対構成（6段パイプライン・時間的隣接のみ・正誤判定禁止・確認バイアス排除・相手別分離・ルーミネーション防止・安全弁7種） |
| 13al | other_hypothesis_emotion_return.py | 751 | 65 | 感情 | 他者仮説由来の感情帯域追加（4段パイプライン・キーワード辞書照合・帯域±0.02以下・memory_emotion_returnとの合算上限・ルーミネーション減衰・enrichment非露出・安全弁7種） |
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
| 14 | __init__.py | 1,858 | - | 基盤 | エクスポート定義 |
| 15 | tone.py | 698 | 36 | 出力 | トーン・ユーモア制御 |
| 16 | tendency_awareness.py | 651 | 44 | 内省 | 傾向の自己認知 |
| 16 | scoped_goal.py | 660 | 40 | 目的 | スコープ目的（1ターン） |
| 17 | stm_emotion_coupling.py | 604 | 40 | 感情 | 短期記憶-感情連携（orchestrator統合済み: 再活性化・蓄積） |
| 18 | multi_emotion.py | 495 | 36 | 感情 | 複数感情独立管理（orchestrator統合済み: 独立減衰） |
| 19 | responsibility.py | 480 | 32 | 責任 | 責任記録・評価 |
| 20 | dynamics.py | 474 | 24 | 感情 | 感情ダイナミクス相 |
| 21 | decision_bias.py | 465 | 30 | 判断 | 判断バイアス計算 |
| 22 | short_term_loop.py | 432 | 24 | 記憶 | 短期感情ループ |
| 23 | short_term_memory.py | 399 | 24 | 記憶 | 短期記憶管理 |
| 24 | persistence.py | 395 | 22 | 基盤 | 永続化システム |
| 25 | emotion_amplitude.py | 362 | 24 | 感情 | 感情振幅調整（orchestrator統合済み: dynamics相連動） |
| 26 | reaction_with_stm.py | 294 | - | 感情 | STM統合反応 |
| 27 | thought.py | 473 | 36 | 出力 | 思考候補生成・選択（15ポリシー動的選択、6断面スコアリング、safety/autonomy軸） |
| 28 | state.py | 258 | - | 基盤 | 心理状態データ構造 |
| 29 | snapshot.py | 239 | - | 基盤 | スナップショット管理 |
| 30 | responsibility_manager.py | 210 | - | 責任 | 責任マネージャー |
| 31 | reaction.py | 201 | - | 感情 | 反応処理 |
| 32 | expression.py | 178 | - | 出力 | 表現生成（brain.py接続済み, psyche enrichment付き） |
| 33 | perception.py | 341 | 294 | 入力 | 知覚処理（brain.py接続済み, LLM enrichment有効, 知覚バイアス: mood.valence→emotion_valence微弱加算） |
| 34 | memory_link.py | 101 | - | 記憶 | 記憶検索 |
| 35 | continuity_manager.py | 95 | - | 4柱 | 連続性管理 |
| 36 | attachment_manager.py | 95 | - | 4柱 | 愛着管理 |
| 37 | identity_manager.py | 90 | - | 4柱 | アイデンティティ管理 |
| 38 | projection_manager.py | 89 | - | 4柱 | 未来投射管理 |
| 39 | pillars.py | 76 | - | 4柱 | 4柱状態定義 |
| 40 | fear.py | 76 | - | 4柱 | 恐怖指数計算 |
| 41 | orchestrator.py | 4,949 | 63 | 統合 | 全モジュール統合管理（PsycheOrchestrator, 71システム, save/load v42(66項目永続化), enrichment(5セクション/48項目), select_policy_dict含む） |

### 2.3 コアシステムファイル

| ファイル | 行数 | 主要クラス/関数 | 説明 |
|---------|------|----------------|------|
| brain.py | 941 | CyreneBrain | 2-call思考生成（perception+expression, psyche enrichment, save/load, LLM parse_percept, think_text/think_spontaneous, PIL直接受渡し, fear_level公開, policy suggestions透明化） |
| voice.py | 437 | VoiceClient | Style-Bert-VITS2連携 |
| vision.py | 393 | GameCapture, HybridEye | 画面キャプチャ・分析 |
| main.py | 299 | main(), speak_sentences(), start_text_listener() | メインループ制御（3経路同列: テキスト→画面→自発, fear_level表示, PIL直接渡し） |

### 2.4 補助モジュール (src/)

| ファイル | 行数 | 説明 |
|---------|------|------|
| simulation.py | 354 | 長期挙動シミュレーション（PsycheOrchestrator経由） |
| api.py | 320 | FastAPI REST API（PsycheOrchestrator経由） |
| llm_wrapper.py | 278 | LLM抽象化レイヤー（画像対応コール含む） |
| memory_manager.py | 216 | 長期記憶+Embedding管理 |
| state_manager.py | 164 | 状態管理補助 |
| cli_tools.py | 161 | CLIツール |
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
│      - goals: [{id: "engage", description: "対話相手と関わる"}]    │
│   5. compute_fear_index() で恐怖指数計算                        │
│   6. PsycheState組み立て                                        │
└─────────────────────────────────────────────────────────────────┘

主要メソッド:
┌─────────────────────────────────────────────────────────────────┐
│ async think_streaming(image_path, vision_summary) -> AsyncGen   │
│                                                                 │
│   ★ 2-call構造 (perception + expression)                       │
│                                                                 │
│   Phase 1: Gemini知覚コール                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ llm_call_with_image(VISION_SYSTEM_PROMPT, prompt, image)│   │
│   │ → 画面の客観的記述テキスト（200文字以内）               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 2: parse_percept (知覚構造化)                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ perception.py: parse_percept(screen_description)        │   │
│   │ → Percept(emotion, intent, topics, valence)             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 3: psyche全フェーズ更新                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ orchestrator.post_response_update(percept, delta)       │   │
│   │ → 感情・ムード・ドライブ・恐怖・自己モデル等を更新     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 4: 記憶検索                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ recall_with_mood(percept, psyche, memory, top_k=3)      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 5: 方針選択                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ orchestrator.select_policy_dict(percept, memories)      │   │
│   │ → Phase 30-35（思考候補→バイアス→沈黙候補→安定化）     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 6: 沈黙判定                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ is_silence_policy(policy) → True なら return (無発話)    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 7: Gemini代弁コール                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ expression.py: render_expression(                        │   │
│   │   state, policy, memories, persona, llm_call,           │   │
│   │   screen_context, recent_history)                        │   │
│   │ → {"text": "セリフ", "meta": {emotion, intensity}}      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 8: 文分割 + yield                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 分割位置: 。！？!?\n♪♥♡★☆                               │   │
│   │ 各文をyield → main.pyで音声合成                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Phase 9: ログ + 定期記憶保存 (5ターンごと)                    │
│                                                                 │
│ async summarize_and_save():                                     │
│   1. 会話ログ最新10件取得                                       │
│   2. Gemini単発呼び出しで要約生成                               │
│      出力JSON: {summary, keywords, importance}                  │
│   3. memory.maybe_save(summary, keywords, importance)           │
│   4. 会話ログクリア                                             │
│                                                                 │
│ @property last_emotion -> str:                                  │
│   psycheから取得した最後の感情名 (main.py音声パラメータ用)      │
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
│   戻り値: Base64エンコードWAV（外部送信用）                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 main.py - メインループ

```
ヘルパー関数:
┌─────────────────────────────────────────────────────────────────┐
│ async start_text_listener(queue: asyncio.Queue):                │
│   stdinを別スレッドで読み、asyncio.Queueに投入                 │
│   (daemon thread, non-blocking)                                 │
│                                                                 │
│ async speak_sentences(brain, voice, sentence_gen) -> int:       │
│   共通発話パイプライン: streaming generator → voice.speak        │
│   感情はbrain.last_emotionから取得、style_weightを自動設定      │
│   戻り値: 発話した文数                                          │
└─────────────────────────────────────────────────────────────────┘

メインループ処理フロー (3経路同列):
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
│   │ 6. テキスト入力リスナー起動 (stdin → asyncio.Queue)     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   メインループ (3経路同列・恒常優先なし):                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ while True:                                             │   │
│   │                                                         │   │
│   │   [0] 終了キーチェック                                  │   │
│   │       if keyboard.is_pressed('l'): break                │   │
│   │                                                         │   │
│   │   [A] テキスト入力チェック (非ブロッキング)             │   │
│   │       user_text = text_queue.get_nowait()               │   │
│   │       → brain.think_streaming_text(user_text)           │   │
│   │       → speak_sentences()                               │   │
│   │                                                         │   │
│   │   [B] 画面キャプチャ (テキスト入力がなかった場合)       │   │
│   │       frame = capture.capture_frame()                   │   │
│   │       → hybrid_eye.analyze_frame(frame)                 │   │
│   │       → brain.think_streaming(image, vision_summary)    │   │
│   │       → speak_sentences()                               │   │
│   │                                                         │   │
│   │   [C] 自発起動チェック (外部入力がなかった場合)         │   │
│   │       idle_sec >= 30.0 の場合:                          │   │
│   │       → brain.think_streaming_spontaneous()             │   │
│   │       → speak_sentences()                               │   │
│   │                                                         │   │
│   │   [D] ループ遅延                                        │   │
│   │       await asyncio.sleep(0.1)                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   終了処理:                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. brain.summarize_and_save() - 長期記憶保存            │   │
│   │ 2. brain.save_state() - psyche状態保存                  │   │
│   │ 3. capture.release() - dxcamリソース解放                │   │
│   │ 4. voice.close() - HTTPクライアント終了                 │   │
│   │ 5. 一時ファイル削除                                     │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Psycheシステム詳細

### 4.0 PsycheOrchestrator (orchestrator.py) — 全モジュール統合管理

brain.py からは `PsycheOrchestrator` のみを参照し、全56モジュール（40システム）を一元管理する。

```
実行モデル:
┌────────────────────────────────────────────────────────────────┐
│ 毎ティック (Phase 1-7):                                         │
│   react_with_stm → dynamics → emotion_amplitude                │
│   → multi_emotion(独立減衰) → stm_emotion_coupling(再活性化)    │
│   → attachment → responsibility                                 │
│   → self_reference → repeated_tendency → fear_recompute        │
│                                                                 │
│ 3ティック毎 (Phase 8-14):                                       │
│   tendency_awareness → self_model → proto_goal_vector           │
│   → goal_candidates → transient_goal → scoped_goal             │
│   → intrinsic_motivation                                        │
│                                                                 │
│ 5ティック毎 (Phase 15-26):                                      │
│   temporal_diff → strain → self_image → coherence               │
│   → narrative → episodic → binding → memory_integration          │
│   → introspection → consumption → expectation                   │
│   → other_model → value_orientation                              │
│                                                                 │
│ 10ティック毎 (Phase 27-29):                                     │
│   stability_valve → long_term_dynamics → snapshot               │
│                                                                 │
│ プロンプト生成前 (Phase 30-35):                                  │
│   thought → decision_bias → tone → context_sensitivity          │
│   → silence_hesitation → stability_valve (bias)                 │
└────────────────────────────────────────────────────────────────┘

brain.py との接続 (2-call構造):
  __init__()               → PsycheOrchestrator(memory_count=...) + load()
  think_streaming() Phase2 → perception.parse_percept(screen_description, llm_call, state)
  think_streaming() Phase3 → orchestrator.post_response_update(percept, delta, user_id)
  think_streaming() Phase5 → orchestrator.select_policy_dict(percept, memories)
  think_streaming() Phase7 → expression.render_expression(..., psyche_enrichment=enrichment)
  save_state()             → orchestrator.save() (main.py shutdownから呼出)
  main.py shutdown         → brain.save_state() + orchestrator.save()
  summarize_and_save()     → orchestrator.on_memory_saved(summary, keywords, count)
  _build_prompt()          → 旧フロー用に保持（summarize_and_save等で使用）

get_prompt_enrichment(user_id) 出力セクション (5セクション):
  【心理状態（内面）】 → 感情, ムード, ドライブ, 恐怖, 支配的感情, 責任, 責任拡散, 安定弁, 感情連動
  【自己認識】         → 自己像, 一貫性, 傾向, 変化, 連続性緊張, 自己語り, 長期傾向
  【動機・目標】       → 動機, 目標候補, 期待, スコープ目標, 一時目標, 方向ベクトル, 候補拡張
  【記憶・内省】       → エピソード記憶, 感情結合, 内省消費, 他者モデル, 内省, 記憶統合
  【判断傾向】         → 判断バイアス, トーン推奨, 空気読み, 沈黙傾向

save()/load() 永続化対象 (v4, 20項目):
  core: psyche, loop_state, dynamics, tick_count
  v4追加: amplitude, value_orientation, self_ref_state, last_self_view,
          tendency_awareness, last_diff_summary, last_strain, last_self_image,
          last_coherence, last_narrative, last_episodes, last_bindings,
          last_trace, last_consumption, last_expectations, last_motives,
          last_other_model
  v3スナップショットとの後方互換性あり
```

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
他者モデル 完全実装仕様 (other_agent_model.py: 1,603行 / 112テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    自己側の観測・反応に偏っている現状に対し、                    │
│    「相手がどう感じているか」を推測する独立層を配置する。        │
│    自己と他者の境界を弱く構造化し、自我形成の前段条件を整える。  │
│    他者モデルは自己像を固定せず、外部に対する                    │
│    「推測の窓口」を用意するだけである。                          │
│                                                                 │
│  ID生成: uuid.uuid4().hex[:12] (12文字の一意識別子)             │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  Enum定義 (4種)                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ObservationSourceType (他者観測の入力源分類):                   │
│    EXTERNAL_CONTEXT = "external_context"  外部文脈由来          │
│    REACTION_LOG     = "reaction_log"      反応ログ由来          │
│    SELF_CONTRAST    = "self_contrast"     自己対比由来          │
│    MIXED            = "mixed"             複数ソース混合        │
│                                                                 │
│  InferenceBasis (推論根拠の種類):                                │
│    BEHAVIORAL = "behavioral"   行動的根拠                       │
│    CONTEXTUAL = "contextual"   文脈的根拠                       │
│    CONTRAST   = "contrast"     対比的根拠                       │
│    COMBINED   = "combined"     複合的な根拠                     │
│    UNDEFINED  = "undefined"    未確定                            │
│                                                                 │
│  HypothesisStrength (仮説の安定度 - 評価ではなく記述):          │
│    STRONG    = "strong"     strength >= 0.7                      │
│    MODERATE  = "moderate"   strength >= 0.4                      │
│    WEAK      = "weak"       strength >= 0.2                      │
│    FAINT     = "faint"      strength >= 0.05                     │
│    UNDEFINED = "undefined"  strength < 0.05                      │
│                                                                 │
│  HypothesisFreshness (仮説の新鮮度):                            │
│    FRESH  = "fresh"    freshness >= 0.8                          │
│    RECENT = "recent"   freshness >= 0.6                          │
│    AGING  = "aging"    freshness >= 0.4                          │
│    STALE  = "stale"    freshness >= 0.15                         │
│    FADED  = "faded"    freshness < 0.15                          │
│                                                                 │
│  レベル判定関数 (pure):                                         │
│    determine_freshness_level(freshness: float)                  │
│      → HypothesisFreshness                                      │
│    determine_strength_level(strength: float)                    │
│      → HypothesisStrength                                       │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  Core Dataclasses (全て frozen=True)                             │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ObservationLink (観測と仮説の弱い接続)                  │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    link_id: str                  一意識別子              │   │
│  │    hypothesis_id: str            紐付く仮説ID            │   │
│  │    source_type: ObservationSourceType  入力源            │   │
│  │    source_description: str       観測内容の記述          │   │
│  │    contribution: float           寄与度 (0.0〜1.0)       │   │
│  │                                                          │   │
│  │  生成時の寄与度計算:                                     │   │
│  │    contribution = max(0.1, 1.0 - idx * 0.15)             │   │
│  │    idx=0: 1.0, idx=1: 0.85, idx=2: 0.70, ...            │   │
│  │    最大リンク数: max_evidence_per_hypothesis (default=8) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ OtherStateHypothesis (他者の状態仮説 - コア構造)        │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    hypothesis_id: str             一意識別子             │   │
│  │    source_type: ObservationSourceType  入力源            │   │
│  │    basis: InferenceBasis          推論根拠               │   │
│  │    description: str               推測内容（断定しない） │   │
│  │    timestamp: str                 生成時刻               │   │
│  │    freshness: float               0.0〜1.0 (生成時1.0)   │   │
│  │    strength: float                0.0〜1.0 (根拠安定度)  │   │
│  │    reference_count: int           参照回数               │   │
│  │    evidence_ids: tuple[str, ...]  ObservationLink群ID    │   │
│  │    competing_ids: tuple[str, ...] 競合仮説IDリスト       │   │
│  │    revision_count: int            修正回数               │   │
│  │    undetermined_aspects:          未確定側面             │   │
│  │      tuple[str, ...]             生成時固定:             │   │
│  │      ("intent_uncertain", "state_approximate")           │   │
│  │                                                          │   │
│  │  メソッド:                                               │   │
│  │    get_freshness_level()                                 │   │
│  │      → determine_freshness_level(self.freshness)         │   │
│  │    get_strength_level()                                  │   │
│  │      → determine_strength_level(self.strength)           │   │
│  │                                                          │   │
│  │  変異メソッド (全て新インスタンスを返す, frozen):        │   │
│  │    with_freshness(new_freshness)                         │   │
│  │      → freshness = max(0.0, min(1.0, new_freshness))    │   │
│  │    with_strength(new_strength)                           │   │
│  │      → strength = max(0.0, min(1.0, new_strength))      │   │
│  │    with_reference()                                      │   │
│  │      → reference_count + 1                               │   │
│  │    revise(new_description)                               │   │
│  │      → description更新, revision_count + 1               │   │
│  │    with_competing(competing_id)                          │   │
│  │      → competing_ids に追加（重複チェックあり）          │   │
│  │      → 既に含む場合は self をそのまま返す                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SelfOtherBoundary (自己/他者の境界指標)                 │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    boundary_id: str                一意識別子            │   │
│  │    self_description: str           自己側の状態記述      │   │
│  │    other_description: str          他者仮説群の統合記述  │   │
│  │      → 最大200文字 ([:200]で切り詰め)                    │   │
│  │    divergence: float               0.0〜1.0 (乖離度)     │   │
│  │      → round(min(1.0, max(0.0, divergence)), 4)          │   │
│  │    boundary_aspects: tuple[str, ...] 差異の側面          │   │
│  │      → 最大5要素 ([:5]で切り詰め)                        │   │
│  │    timestamp: str                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ OtherModelStore (不変スナップショット)                   │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    hypotheses: tuple[OtherStateHypothesis, ...]          │   │
│  │    observation_links: tuple[ObservationLink, ...]        │   │
│  │    boundaries: tuple[SelfOtherBoundary, ...]             │   │
│  │    total_hypotheses_created: int                          │   │
│  │    total_revisions: int                                   │   │
│  │    total_expirations: int                                 │   │
│  │    average_freshness: float  (round 4桁)                 │   │
│  │    average_strength: float   (round 4桁)                 │   │
│  │    active_hypothesis_count: int                           │   │
│  │    competing_pair_count: int                              │   │
│  │    boundary_count: int                                    │   │
│  │    timestamp: str                                         │   │
│  │    description: str  (_generate_store_description)        │   │
│  │                                                          │   │
│  │  フィルタメソッド:                                       │   │
│  │    has_hypotheses() → bool                               │   │
│  │      len(hypotheses) > 0                                 │   │
│  │    get_active_hypotheses(stale_threshold=0.15) → tuple   │   │
│  │      freshness > stale_threshold のもののみ              │   │
│  │    get_strong_hypotheses() → tuple                       │   │
│  │      strength > 0.5 のもののみ                           │   │
│  │                                                          │   │
│  │  シリアライゼーション:                                   │   │
│  │    to_dict() → dict                                      │   │
│  │      hypotheses → _hypothesis_to_dict() で変換           │   │
│  │      observation_links → 各フィールドをdict化            │   │
│  │      boundaries → 各フィールドをdict化                   │   │
│  │        boundary_aspects: list化                          │   │
│  │    from_dict(data) → OtherModelStore                     │   │
│  │      デフォルト値: hypothesis_id必須,                     │   │
│  │        source_type="mixed", basis="undefined",           │   │
│  │        description="", freshness=0.0, strength=0.0,      │   │
│  │        reference_count=0, revision_count=0               │   │
│  │      ObservationLink: link_id必須,                       │   │
│  │        source_type="mixed", contribution=0.0             │   │
│  │      SelfOtherBoundary: boundary_id必須,                 │   │
│  │        divergence=0.0                                    │   │
│  │      スカラー統計: 全てデフォルト0 or 0.0 or ""          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ OtherAgentModelConfig (設定 - 判断に影響しない)          │   │
│  │                                                          │   │
│  │    max_hypotheses: int = 60                               │   │
│  │      (自己モデルより少なめ。他者推測は不安定)            │   │
│  │    base_decay_rate: float = 0.05                          │   │
│  │      (他者推測はやや速い減衰)                            │   │
│  │    strength_decay_rate: float = 0.03                      │   │
│  │    freshness_boost_on_reference: float = 0.10             │   │
│  │    stale_threshold: float = 0.15                          │   │
│  │    min_strength_for_retention: float = 0.05               │   │
│  │    max_evidence_per_hypothesis: int = 8                   │   │
│  │    max_boundaries: int = 10                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  抽出関数 (Pure, Duck Typing, None安全, dict/object両対応)       │
│  戻り値: list[(description, basis_hint, strength_hint,          │
│                evidence_source_descriptions)]                    │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ extract_from_external_context(context: Optional[Any])   │   │
│  │                                                          │   │
│  │  Duck typing attrs: pace, weight, density, continuity,  │   │
│  │                     responsiveness                       │   │
│  │  判定: hasattr(context, "responsiveness") and            │   │
│  │        hasattr(context, "weight")                        │   │
│  │  デフォルト値: 全て0.5                                   │   │
│  │                                                          │   │
│  │  抽出ルール (閾値 → 仮説 → basis → strength計算):       │   │
│  │                                                          │   │
│  │  ① responsiveness >= 0.7:                                │   │
│  │     "Other party appears engaged and responsive"         │   │
│  │     basis = "behavioral"                                 │   │
│  │     strength = min(1.0, responsiveness * 0.6)            │   │
│  │     evidence: ["Responsiveness: {:.2f}"]                 │   │
│  │                                                          │   │
│  │  ② responsiveness <= 0.3:                                │   │
│  │     "Other party appears disengaged or distant"          │   │
│  │     basis = "behavioral"                                 │   │
│  │     strength = min(1.0, (1.0 - responsiveness) * 0.5)    │   │
│  │     evidence: ["Responsiveness: {:.2f}"]                 │   │
│  │                                                          │   │
│  │  ③ weight >= 0.7:                                        │   │
│  │     "Interaction atmosphere feels heavy or tense"        │   │
│  │     basis = "contextual"                                 │   │
│  │     strength = min(1.0, weight * 0.5)                    │   │
│  │     evidence: ["Weight: {:.2f}"]                         │   │
│  │                                                          │   │
│  │  ④ pace >= 0.7:                                          │   │
│  │     "Interaction pace suggests energetic exchange"        │   │
│  │     basis = "contextual"                                 │   │
│  │     strength = min(1.0, pace * 0.4)                      │   │
│  │     evidence: ["Pace: {:.2f}"]                           │   │
│  │                                                          │   │
│  │  ⑤ 上記全て不成立 AND                                    │   │
│  │     0.3 < responsiveness < 0.7 AND                       │   │
│  │     0.3 < weight < 0.7:                                  │   │
│  │     "Other party state appears neutral or ambiguous"     │   │
│  │     basis = "contextual"                                 │   │
│  │     strength = 0.15 (固定)                               │   │
│  │     evidence: ["Responsiveness: {:.2f}, Weight: {:.2f}"] │   │
│  │                                                          │   │
│  │  dict入力: 同じ閾値ルール (①②③のみ, ④⑤なし)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ extract_from_reaction_log(log: Optional[Any])           │   │
│  │                                                          │   │
│  │  Duck typing attrs: entries[], source_text, intent,     │   │
│  │                     emotion_label, valence               │   │
│  │  判定: hasattr(log, "entries")                           │   │
│  │  処理上限: entries[:5] (直近5エントリのみ)               │   │
│  │                                                          │   │
│  │  抽出ルール:                                             │   │
│  │                                                          │   │
│  │  ① intent == "question":                                 │   │
│  │     "Other expressed questioning intent"                 │   │
│  │     basis = "behavioral"                                 │   │
│  │     strength = 0.4 (固定)                                │   │
│  │     evidence: ["Intent: question"]                       │   │
│  │     + source_text あれば ["Source: {[:60]}"] 追加        │   │
│  │                                                          │   │
│  │  ② valence > 0.3:                                        │   │
│  │     "Other party tone appears positive"                  │   │
│  │     basis = "behavioral"                                 │   │
│  │     strength = min(1.0, valence * 0.5)                   │   │
│  │     evidence: ["Valence: {:.2f}"]                        │   │
│  │     + emotion_label あれば ["Emotion: {label}"] 追加     │   │
│  │                                                          │   │
│  │  ③ valence < -0.3:                                       │   │
│  │     "Other party tone appears negative"                  │   │
│  │     basis = "behavioral"                                 │   │
│  │     strength = min(1.0, abs(valence) * 0.5)              │   │
│  │     evidence: ["Valence: {:.2f}"]                        │   │
│  │     + emotion_label あれば ["Emotion: {label}"] 追加     │   │
│  │                                                          │   │
│  │  dict入力: 同じルール (source_text省略)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ extract_from_self_contrast(                             │   │
│  │   self_state: Optional[Any],                            │   │
│  │   other_signals: Optional[Any])                         │   │
│  │                                                          │   │
│  │  self_state duck typing: intensity, description         │   │
│  │  other_signals duck typing: responsiveness, weight      │   │
│  │  両方None → 空リスト                                     │   │
│  │  デフォルト値: intensity=0.5, responsiveness=0.5,        │   │
│  │               weight=0.5                                 │   │
│  │                                                          │   │
│  │  抽出ルール:                                             │   │
│  │                                                          │   │
│  │  ① divergence = abs(self_intensity - responsiveness)     │   │
│  │     divergence >= 0.4:                                   │   │
│  │     "Contrast detected between self-state               │   │
│  │      (intensity={:.2f}) and other signals                │   │
│  │      (responsiveness={:.2f})"                            │   │
│  │     basis = "contrast"                                   │   │
│  │     strength = min(1.0, divergence * 0.7)                │   │
│  │     evidence: [                                          │   │
│  │       "Self intensity: {:.2f}",                          │   │
│  │       "Other responsiveness: {:.2f}",                    │   │
│  │       "Divergence: {:.2f}"                               │   │
│  │     ]                                                    │   │
│  │                                                          │   │
│  │  ② weight_div = abs(self_intensity - other_weight)       │   │
│  │     weight_div >= 0.5:                                   │   │
│  │     "Self-other weight divergence:                       │   │
│  │      self intensity={:.2f}, other weight={:.2f}"         │   │
│  │     basis = "contrast"                                   │   │
│  │     strength = min(1.0, weight_div * 0.6)                │   │
│  │     evidence: [                                          │   │
│  │       "Self intensity: {:.2f}",                          │   │
│  │       "Other weight: {:.2f}"                             │   │
│  │     ]                                                    │   │
│  │                                                          │   │
│  │  dict入力: 同じルール                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  計算関数 (Pure)                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ compute_observation_strength(links: list[ObservationLink])│  │
│  │ → float                                                  │   │
│  │                                                          │   │
│  │  加重集約 (単純平均ではない):                            │   │
│  │    weight = 1.0 / (1.0 + i * 0.2)                       │   │
│  │    i=0: w=1.0, i=1: w=0.833, i=2: w=0.714, ...         │   │
│  │    weighted_sum += link.contribution * weight             │   │
│  │    total_weight += weight                                 │   │
│  │    result = min(1.0, weighted_sum / total_weight)        │   │
│  │  空リスト → 0.0                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ detect_hypothesis_competitions(                         │   │
│  │   hypotheses: list[OtherStateHypothesis])               │   │
│  │ → list[tuple[str, str]]                                  │   │
│  │                                                          │   │
│  │  2段階の競合検出:                                        │   │
│  │                                                          │   │
│  │  Tier 1: 同source_type + 異basis                        │   │
│  │    description語彙のJaccard類似度 >= 0.2 → 競合          │   │
│  │    (同じ入力源から異なる解釈 = 弱い競合)                 │   │
│  │                                                          │   │
│  │  Tier 2: 異basis (source_type不問)                       │   │
│  │    description語彙のJaccard類似度 >= 0.4 → 競合          │   │
│  │    (異なる根拠で類似記述 = 強い競合)                     │   │
│  │                                                          │   │
│  │  Jaccard計算:                                            │   │
│  │    words_a = set(a.description.lower().split())          │   │
│  │    words_b = set(b.description.lower().split())          │   │
│  │    jaccard = |words_a ∩ words_b| / |words_a ∪ words_b|  │   │
│  │                                                          │   │
│  │  ペアは重複排除 (seen set)、i < j のみ走査              │   │
│  │  競合は排除しない（許容する設計原則）                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ determine_inference_basis(                              │   │
│  │   source_types: list[ObservationSourceType])            │   │
│  │ → InferenceBasis                                         │   │
│  │                                                          │   │
│  │  空リスト → UNDEFINED                                    │   │
│  │  unique > 1 → COMBINED                                   │   │
│  │  マッピング:                                             │   │
│  │    EXTERNAL_CONTEXT → CONTEXTUAL                         │   │
│  │    REACTION_LOG     → BEHAVIORAL                         │   │
│  │    SELF_CONTRAST    → CONTRAST                           │   │
│  │    MIXED            → COMBINED                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ generate_hypothesis_description(                        │   │
│  │   basis: InferenceBasis,                                │   │
│  │   source_descriptions: list[str])                       │   │
│  │ → str                                                    │   │
│  │                                                          │   │
│  │  プレフィックス:                                         │   │
│  │    BEHAVIORAL → "Based on observed behavior"             │   │
│  │    CONTEXTUAL → "Based on contextual signals"            │   │
│  │    CONTRAST   → "Based on self-other contrast"           │   │
│  │    COMBINED   → "Based on multiple sources"              │   │
│  │    UNDEFINED  → "Weak hypothesis"                        │   │
│  │  本文: "; ".join(d[:80] for d in descs[:3])             │   │
│  │  形式: "{prefix}: {combined_descriptions}"              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ compute_self_other_boundary(                            │   │
│  │   self_description: str,                                │   │
│  │   other_hypotheses: list[OtherStateHypothesis])         │   │
│  │ → SelfOtherBoundary                                      │   │
│  │                                                          │   │
│  │  仮説なし → divergence=0.0, aspects=(), 空境界          │   │
│  │    self_description: "No self-state description          │   │
│  │                       available" (空時)                   │   │
│  │    other_description: "No other-state hypotheses         │   │
│  │                        available"                        │   │
│  │                                                          │   │
│  │  仮説あり:                                               │   │
│  │    other統合: h.description[:80] for h in hyps[:5]       │   │
│  │    → "; ".join() → [:200]                                │   │
│  │                                                          │   │
│  │    乖離度計算 (語彙Jaccard反転):                         │   │
│  │      self_words = set(self_desc.lower().split())         │   │
│  │      other_words = 全仮説の語彙集合の和                  │   │
│  │      overlap = |self ∩ other| / |self ∪ other|           │   │
│  │      divergence = 1.0 - overlap                          │   │
│  │      union が空 → divergence = 0.5                       │   │
│  │                                                          │   │
│  │    境界側面 (boundary_aspects):                          │   │
│  │      各仮説の basis.value → "inference_{basis}" 形式     │   │
│  │      仮説が2件以上 → "multiple_hypotheses" 追加          │   │
│  │      最大5要素                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  OtherAgentModelSystem (メインシステムクラス)                    │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  内部状態:                                                      │
│    _config: OtherAgentModelConfig                               │
│    _hypotheses: list[OtherStateHypothesis]   仮説リスト         │
│    _observation_links: list[ObservationLink] 観測リンク          │
│    _boundaries: list[SelfOtherBoundary]     境界リスト          │
│    _total_created: int = 0                  生成総数             │
│    _total_revisions: int = 0                修正総数             │
│    _total_expirations: int = 0              失効総数             │
│    _last_store: Optional[OtherModelStore] = None                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ observe_other(                                          │   │
│  │   external_context=None, reaction_log=None,             │   │
│  │   self_state=None) → OtherModelStore                    │   │
│  │                                                          │   │
│  │  メイン処理フロー:                                       │   │
│  │                                                          │   │
│  │  Step 1: 3系統の抽出                                     │   │
│  │    extract_from_external_context(external_context)       │   │
│  │    extract_from_reaction_log(reaction_log)               │   │
│  │    extract_from_self_contrast(self_state, external_ctx)  │   │
│  │    ※ self_contrast の other_signals に external_context  │   │
│  │      を使用（他者信号の代替として）                      │   │
│  │                                                          │   │
│  │  Step 2: 仮説生成                                        │   │
│  │    各抽出結果 → OtherStateHypothesis 生成                │   │
│  │    basis_hint → InferenceBasis マッピング:               │   │
│  │      "behavioral" → BEHAVIORAL                           │   │
│  │      "contextual" → CONTEXTUAL                           │   │
│  │      "contrast"   → CONTRAST                             │   │
│  │      "combined"   → COMBINED                             │   │
│  │      その他       → UNDEFINED                            │   │
│  │    _generate_observation_links() で根拠リンク生成        │   │
│  │    freshness = 1.0 (初期値固定)                          │   │
│  │    strength = max(0.0, min(1.0, strength_hint))          │   │
│  │    undetermined_aspects = ("intent_uncertain",           │   │
│  │                            "state_approximate")          │   │
│  │                                                          │   │
│  │  Step 3: 競合検出                                        │   │
│  │    detect_hypothesis_competitions(全仮説)                │   │
│  │    → 競合ペア双方に with_competing() で相互リンク       │   │
│  │                                                          │   │
│  │  Step 4: 境界計算                                        │   │
│  │    self_state → description 取得 (duck/dict)             │   │
│  │    compute_self_other_boundary(self_desc, 全仮説)        │   │
│  │    → _boundaries に追加                                  │   │
│  │    容量超過時: pop(0) で FIFO 削除                       │   │
│  │      (max_boundaries=10)                                 │   │
│  │                                                          │   │
│  │  Step 5: 減衰適用                                        │   │
│  │    _apply_decay()                                        │   │
│  │                                                          │   │
│  │  Step 6: 容量制限                                        │   │
│  │    _enforce_capacity()                                   │   │
│  │                                                          │   │
│  │  Step 7: スナップショット生成                            │   │
│  │    _build_store(current_time) → OtherModelStore          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ decay_hypotheses() → OtherModelStore                    │   │
│  │   _apply_decay() を呼び出し → _build_store()             │   │
│  │                                                          │   │
│  │ reference_hypothesis(hypothesis_id: str) → None         │   │
│  │   仮説検索 → with_reference() + with_freshness(          │   │
│  │     freshness + freshness_boost_on_reference)            │   │
│  │   reference_count +1, freshness +0.10                    │   │
│  │                                                          │   │
│  │ revise_hypothesis(hypothesis_id: str,                   │   │
│  │                   new_description: str) → None          │   │
│  │   仮説検索 → revise(new_description)                    │   │
│  │   revision_count +1, _total_revisions +1                │   │
│  │                                                          │   │
│  │ get_active_hypotheses(max_count=10)                     │   │
│  │   → list[OtherStateHypothesis]                          │   │
│  │   freshness > stale_threshold のみ                       │   │
│  │   strength降順ソート、上位max_count件                    │   │
│  │                                                          │   │
│  │ get_store() → OtherModelStore                           │   │
│  │   _build_store(現在時刻)                                 │   │
│  │                                                          │   │
│  │ get_last_store() → Optional[OtherModelStore]            │   │
│  │   最後のスナップショット                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 内部メソッド                                             │   │
│  │                                                          │   │
│  │ _apply_decay():                                         │   │
│  │   各仮説に対して:                                        │   │
│  │     ref_modifier = max(0.5, 1.0 - ref_count * 0.1)      │   │
│  │       ref=0: 1.0, ref=1: 0.9, ref=2: 0.8, ...          │   │
│  │       ref=5+: 0.5 (下限)                                │   │
│  │     freshness_decay = base_decay_rate * ref_modifier     │   │
│  │       参照が多い仮説は減衰が遅い                        │   │
│  │     new_freshness = freshness - freshness_decay           │   │
│  │     new_strength = strength - strength_decay_rate         │   │
│  │                                                          │   │
│  │   除去条件:                                              │   │
│  │     new_freshness <= stale_threshold(0.15) AND           │   │
│  │     new_strength <= min_strength_for_retention(0.05)     │   │
│  │     → _total_expirations +1                              │   │
│  │     → 関連 observation_links も除去                      │   │
│  │                                                          │   │
│  │ _enforce_capacity():                                    │   │
│  │   while len > max_hypotheses:                            │   │
│  │     weakest = min by (strength, freshness) tuple         │   │
│  │     → 除去 + 関連links除去 + _total_expirations +1      │   │
│  │                                                          │   │
│  │ _generate_observation_links(hyp_id, source_type, descs):│   │
│  │   各descに対して:                                        │   │
│  │     contribution = max(0.1, 1.0 - idx * 0.15)           │   │
│  │   最大 max_evidence_per_hypothesis(8) 件                 │   │
│  │                                                          │   │
│  │ _build_store(current_time):                             │   │
│  │   active = [h for h if freshness > stale_threshold]      │   │
│  │   avg_freshness = sum/len (空→0.0) → round 4桁          │   │
│  │   avg_strength = sum/len (空→0.0) → round 4桁           │   │
│  │   competition_pairs = detect_hypothesis_competitions()   │   │
│  │     ※ _build_store のたびに再検出                       │   │
│  │   description = _generate_store_description()            │   │
│  │   → OtherModelStore 生成 → _last_store に保存            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ _generate_store_description():                          │   │
│  │                                                          │   │
│  │  total == 0: "No other-state hypotheses formed yet."     │   │
│  │                                                          │   │
│  │  生成フォーマット ("; " 区切り + "."):                    │   │
│  │    "{active} active hypotheses out of {total} total"     │   │
│  │                                                          │   │
│  │    強度ラベル:                                           │   │
│  │      avg_strength >= 0.5 → "generally strong hypotheses" │   │
│  │      avg_strength >= 0.2 → "moderate strength hypotheses"│   │
│  │      else               → "mostly weak hypotheses"       │   │
│  │                                                          │   │
│  │    competing > 0:  "{n} competing pairs"                 │   │
│  │    expirations > 0: "{n} expired"                        │   │
│  │    boundaries > 0:  "{n} boundaries"                     │   │
│  │    常に: "avg freshness: {:.2f}"                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  統合関数 (introspection integration)                            │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ observe_from_chain(system, external_context=None,       │   │
│  │   reaction_log=None, self_state=None)                   │   │
│  │ → OtherModelStore                                        │   │
│  │                                                          │   │
│  │ system.observe_other() への委譲ラッパー                  │   │
│  │ 各入力は読み取り専用で参照される                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ generate_other_model_tags(store=None, scale=1.0)        │   │
│  │ → list[dict]                                             │   │
│  │                                                          │   │
│  │ MUST NOT influence decisions (内省/認知のみ)             │   │
│  │                                                          │   │
│  │ 空/None → 1タグのみ:                                     │   │
│  │   category: "OTHER_MODEL_COUNT"                          │   │
│  │   label: "no_hypotheses"                                 │   │
│  │   weight: 0.03 * scale                                   │   │
│  │                                                          │   │
│  │ 仮説あり → 6カテゴリのタグ生成:                          │   │
│  │                                                          │   │
│  │ ① OTHER_MODEL_COUNT     weight: 0.06 * scale            │   │
│  │    label: "hypotheses_{active_count}"                    │   │
│  │    desc: "holds {n} active hypotheses"                   │   │
│  │                                                          │   │
│  │ ② OTHER_MODEL_STRENGTH  weight: 0.07 * scale            │   │
│  │    label: "strength_{level}"                             │   │
│  │    desc: "Average: {avg:.2f}, max: {max:.2f}"            │   │
│  │    max_strength = max(h.strength for h in hypotheses)    │   │
│  │                                                          │   │
│  │ ③ OTHER_MODEL_FRESHNESS weight: 0.05 * scale            │   │
│  │    label: "freshness_{level}"                            │   │
│  │    desc: "Average: {avg:.2f}"                            │   │
│  │                                                          │   │
│  │ ④ OTHER_MODEL_COMPETITION weight: 0.06 * scale          │   │
│  │    (competing_pair_count > 0 のときのみ生成)             │   │
│  │    label: "competing_{n}"                                │   │
│  │                                                          │   │
│  │ ⑤ OTHER_MODEL_BOUNDARY  weight: 0.05 * scale            │   │
│  │    (boundary_count > 0 のときのみ生成)                   │   │
│  │    label: "boundaries_{n}"                               │   │
│  │                                                          │   │
│  │ ⑥ OTHER_MODEL_INTEGRATED weight: 0.08 * scale           │   │
│  │    (description 非空のときのみ生成)                      │   │
│  │    label: "other_model_awareness"                        │   │
│  │    desc: store.description そのまま                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ get_other_model_summary(store=None) → str               │   │
│  │                                                          │   │
│  │ None/空 → "=== Other Agent Model State ===\n             │   │
│  │            No hypotheses formed yet."                    │   │
│  │                                                          │   │
│  │ 生成フォーマット:                                        │   │
│  │   "=== Other Agent Model State ==="                      │   │
│  │   "Total hypotheses: {len}"                              │   │
│  │   "Active hypotheses: {active_count}"                    │   │
│  │   "Total created: {total_created}"                       │   │
│  │   "Total revisions: {total_revisions}"                   │   │
│  │   "Total expirations: {total_expirations}"               │   │
│  │   "Average freshness: {:.2f}"                            │   │
│  │   "Average strength: {:.2f}"                             │   │
│  │   "Competing pairs: {n}"                                 │   │
│  │   "Boundaries: {n}"                                      │   │
│  │   ""                                                     │   │
│  │   "Top hypotheses:" (strength降順ソート、上位5件)        │   │
│  │   "  [{source_type}:{basis}] {desc[:80]}"                │   │
│  │   "    (strength: {level}, freshness: {level})"          │   │
│  │   ""                                                     │   │
│  │   "Integrated: {description}"                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ get_other_model_for_introspection(store=None) → dict    │   │
│  │                                                          │   │
│  │ MUST NOT be used as input to decision-making             │   │
│  │                                                          │   │
│  │ None → 全フィールド空/ゼロの dict                        │   │
│  │                                                          │   │
│  │ 返却 dict 構造:                                          │   │
│  │   {                                                      │   │
│  │     "has_hypotheses": bool,                              │   │
│  │     "total_hypotheses": int,                             │   │
│  │     "active_count": int,                                 │   │
│  │     "average_strength": float,                           │   │
│  │     "average_freshness": float,                          │   │
│  │     "source_distribution": {                             │   │
│  │       "external_context": int,                           │   │
│  │       "reaction_log": int,                               │   │
│  │       "self_contrast": int                               │   │
│  │     },                                                   │   │
│  │     "basis_distribution": {                              │   │
│  │       "behavioral": int,                                 │   │
│  │       "contextual": int,                                 │   │
│  │       "contrast": int                                    │   │
│  │     },                                                   │   │
│  │     "competing_pair_count": int,                         │   │
│  │     "boundary_count": int,                               │   │
│  │     "strongest_hypothesis_description":                  │   │
│  │       str (max(strength).description[:120]),             │   │
│  │     "timestamp": str                                     │   │
│  │   }                                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  検証関数 (テスト支援 - メタ検証)                                │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ verify_no_decision_impact(store) → bool                 │   │
│  │   store の public属性を走査し、意思決定に直接            │   │
│  │   影響する属性がないことを確認                            │   │
│  │   許容属性: timestamp, str型, Enum型, tuple型,           │   │
│  │   統計スカラー(total_*, average_*, active_*,             │   │
│  │              competing_*, boundary_count)                 │   │
│  │                                                          │   │
│  │ verify_no_goal_generation(system) → bool                │   │
│  │   forbidden patterns:                                    │   │
│  │     generate_goal, create_goal, set_goal,                │   │
│  │     force, fix, repair, restore, correct,                │   │
│  │     prescribe, enforce                                   │   │
│  │                                                          │   │
│  │ verify_read_only_principle(system) → bool               │   │
│  │   forbidden patterns:                                    │   │
│  │     update_emotion, update_memory, update_tendency,      │   │
│  │     update_value, update_decision, update_responsibility,│   │
│  │     set_emotion, set_memory, set_tendency,               │   │
│  │     modify_bias, apply_to_decision                       │   │
│  │                                                          │   │
│  │ verify_no_value_modification(system) → bool             │   │
│  │   forbidden patterns:                                    │   │
│  │     update_value, set_value, modify_value,               │   │
│  │     update_belief, set_belief, modify_belief,            │   │
│  │     define_identity, set_identity, fix_identity,         │   │
│  │     evaluate_morality, judge_action                      │   │
│  │                                                          │   │
│  │ verify_no_intent_assertion(system) → bool               │   │
│  │   ← 他者モデル固有の検証                                │   │
│  │   forbidden patterns:                                    │   │
│  │     assert_intent, determine_intent, confirm_intent,     │   │
│  │     judge_intent, classify_intent,                       │   │
│  │     assert_belief, determine_belief,                     │   │
│  │     assert_value, determine_value                        │   │
│  │                                                          │   │
│  │  検証方法: dir(system/store) の public メソッド名を       │   │
│  │  小文字化し、forbidden pattern を部分一致チェック        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  永続化・ユーティリティ                                          │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  create_config(**kwargs) → OtherAgentModelConfig                │
│  create_empty_store() → OtherModelStore (全フィールド空/ゼロ)   │
│  create_system(config=None) → OtherAgentModelSystem             │
│                                                                 │
│  save_other_model_state(store, filepath):                       │
│    store.to_dict() → json.dump(ensure_ascii=False, indent=2)    │
│  load_other_model_state(filepath) → OtherModelStore:            │
│    json.load() → OtherModelStore.from_dict()                    │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  設計制約 (design doc 準拠)                                      │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  - 他者の意図・価値・信念を断定しない                            │
│  - 正誤や善悪の評価を付与しない                                  │
│  - 目的や行動の最適化に結び付けない                              │
│  - 自己像や人格の方向性を固定しない                              │
│  - 候補は仮説として保持し固定しない                              │
│  - 競合する候補を許容する                                        │
│  - 後から訂正・撤回を許す (revise, 減衰による自然消滅)           │
│  - 判断選択層・目的生成・価値更新・責任評価に接続しない          │
│  - 外部出力の直接生成層に接続しない                              │
│  - 接続先: 内省記録層への参照素材、記憶参照の補助文脈            │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  __init__.py エクスポート (エイリアス一覧)                       │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  名前衝突回避のため以下をエイリアス化:                           │
│    determine_freshness_level                                    │
│      → determine_hypothesis_freshness_level                     │
│    determine_strength_level                                     │
│      → determine_hypothesis_strength_level                      │
│    observe_from_chain                                           │
│      → observe_other_from_chain                                 │
│    create_config                                                │
│      → create_other_model_config                                │
│    create_empty_store                                           │
│      → create_empty_other_model_store                           │
│    create_system                                                │
│      → create_other_model_system                                │
│    verify_no_decision_impact                                    │
│      → verify_other_model_no_decision_impact                    │
│    verify_no_goal_generation                                    │
│      → verify_other_model_no_goal_generation                    │
│    verify_read_only_principle                                   │
│      → verify_other_model_read_only_principle                   │
│    verify_no_value_modification                                 │
│      → verify_other_model_no_value_modification                 │
│    verify_no_intent_assertion                                   │
│      → (エイリアスなし: 他者モデル固有のため衝突しない)         │
│                                                                 │
│  エクスポート総数: 38シンボル                                    │
│    Enum: 4, Dataclass: 5, System: 1, Helper: 8,                 │
│    Integration: 4, Persistence: 2, Convenience: 3,              │
│    Verification: 5, LevelFunc: 2, Extraction: 3,                │
│    Computation: 1                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.4a 他者モデル入力供給

```
他者モデル入力供給 実装仕様 (other_model_input_supply.py: 308行 / 30テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    other_agent_modelのexternal_contextとreaction_logが常にNone  │
│    で渡されており仮説が一切生成されない問題を解消する。         │
│    入力供給は「観測情報の受け渡し」を成立させるためだけに設ける。│
│    他者状態の断定・出力方針の固定・評価軸の導入をしない。       │
│                                                                 │
│  データ構造:                                                    │
│    ContextSnapshot   - ExternalContext duck typing 互換          │
│      pace, weight, density, continuity, responsiveness          │
│      timestamp, missing_reason                                  │
│    ReactionBufferEntry - STM StimulusEntry 互換                 │
│      source_text, intent, emotion_label, valence                │
│      timestamp, supplied                                        │
│    InputSupplyState  - 全体状態                                 │
│      context_snapshot, reaction_buffer, supply_cursor            │
│      last_supply_time, decay_rate, max_buffer_size              │
│                                                                 │
│  関数:                                                          │
│    create_input_supply      - 初期状態生成                      │
│    update_from_percept      - 周期更新 (STM/dynamics/psyche)    │
│    decay_buffer             - 古い要素の自然減衰                │
│    supply_context           - ContextSnapshot 供給              │
│    supply_reaction_log      - STM互換反応ログ供給               │
│    get_input_supply_summary - サマリ文字列生成                   │
│                                                                 │
│  context計算式:                                                 │
│    pace = len(stm.entries) / stm.max_entries                    │
│    weight = (abs(mood.valence) + arousal) / 2                   │
│    density = len(percept.topics) / 5.0                          │
│    continuity = stm.context_continuity_score                    │
│    responsiveness = 直近エントリの経過時間から段階算出           │
│                                                                 │
│  設計制約:                                                      │
│    供給口は一箇所に統一                                         │
│    供給単位に時刻・由来・欠損タグ必須                           │
│    循環参照防止: supply_cursor による進行管理                    │
│    観測欠損時は中立値 + missing_reason="unobserved"             │
│    減衰と競合保持を常時有効                                     │
│                                                                 │
│  orchestrator配線:                                              │
│    _run_every_tick: self._last_percept = percept                │
│    _run_every_5_ticks Phase 25:                                 │
│      update_input_supply → decay_buffer → supply → observe      │
│    save/load: input_supply フィールド追加                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.4b 他者モデルリアルフィード統合

```
他者モデルリアルフィード統合 実装仕様 (other_model_real_feed.py: 1,063行 / 102テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    現在の他者推定は内部記録中心で、直近対話の反応差分が推定更新  │
│    に十分反映されない。実対話由来の反応断片を他者モデル入力へ    │
│    接続し、推定状態の停滞を防ぐ。                               │
│    他者の意図を断定せず、単回反応を恒常特性へ昇格させない。     │
│    競合する観測は排除せず並立保持して揺らぎ情報として渡す。     │
│                                                                 │
│  Enum定義 (4種):                                                │
│    ObservationFragmentType (8値): SPEECH_REACTION,              │
│      RESPONSE_INTERVAL, TOPIC_TRANSITION, EMOTIONAL_TONE,       │
│      CONTINUED_ENGAGEMENT, REJECTION_ACCEPTANCE,                │
│      CONTEXT_ALIGNMENT, RECENT_HISTORY                          │
│    FragmentFreshness: FRESH / RECENT / AGING / STALE / FADED   │
│    AlignmentStatus: ALIGNED / PARTIAL / UNALIGNED / UNKNOWN    │
│    ConflictStatus: NONE / PARALLEL / CONVERGENCE_RISK          │
│                                                                 │
│  データ構造:                                                    │
│    ObservationFragment  - 8種の抽出関数から生成される観測断片    │
│    ObservationUnit      - 正規化された観測単位（複数断片統合）   │
│    ConflictRecord       - 対立する観測単位のペア記録             │
│    FeedHistoryEntry     - 投入履歴の1エントリ                   │
│    HoldbackEntry        - 未投入保留の1エントリ                 │
│    RealFeedConfig       - 設定パラメータ（9項目）               │
│    RealFeedState        - 全体状態（11フィールド）              │
│    FeedResult           - process()の出力（6フィールド）         │
│                                                                 │
│  断片抽出関数 (8個、pure、duck-typed):                          │
│    extract_speech_reaction    - 発話の質・感情内容               │
│    extract_response_interval  - 入力間隔パターン                │
│    extract_topic_transition   - 話題変化の度合い                │
│    extract_emotional_tone     - 感情的色合い                    │
│    extract_continued_engagement - 継続的関与の程度              │
│    extract_rejection_acceptance - 承認・拒否信号                │
│    extract_context_alignment  - 文脈適合度                      │
│    extract_recent_history     - 直近やりとりの要約              │
│                                                                 │
│  処理パイプライン (10段):                                       │
│    1. 8断片抽出 → normalize_fragments → ObservationUnit化       │
│    2. align_units → 鮮度・値順ソート、整合状態判定              │
│    3. detect_feed_duplicates → 類似観測のグループ化             │
│    4. detect_feed_conflicts → 対立観測の並立保持                │
│    5. apply_freshness → 時間減衰、希薄化履歴更新               │
│    6. suppress_recent_series → 直近投入系列の抑制               │
│    7. ensure_type_diversity → 単一種別支配防止                  │
│    8. check_convergence → 単一解釈収束時に競合補充              │
│    9. check_stagnation → 停滞時に鮮度低下反映                  │
│   10. 出力制限 → max_output_units で切り捨て                    │
│                                                                 │
│  出力統合: enhance_context_with_feed()                          │
│    CONTINUED_ENGAGEMENT → responsiveness 上方修正               │
│    EMOTIONAL_TONE → weight 上方修正                             │
│    TOPIC_TRANSITION → density 調整                              │
│    RESPONSE_INTERVAL → pace 調整                                │
│    既存値を0.0-1.0にclamp、adjustment_weight=0.3               │
│                                                                 │
│  orchestrator配線:                                              │
│    Phase 25a: _real_feed_processor.process() 呼出               │
│    Phase 25: supply_context() 後に enhance_context_with_feed()  │
│    save/load v9: real_feed_state フィールド追加 (33項目)        │
│    enrichment #16: 【記憶・内省】に「観測フィード」行追加       │
│    systems: 40→41                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.4c テキスト対話入力経路

```
テキスト対話入力経路 実装仕様 (text_dialogue_input.py: 1,559行 / 102テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    現在の入力は画面知覚経路に依存しやすく、対話情報の流入機会    │
│    が限定される。テキスト対話を独立した入力経路として追加する。  │
│    既存入力経路を無効化しない。入力経路の追加を評価機構や行動    │
│    決定機構へ拡張しない。単一入力経路への恒常固定を生まない。   │
│                                                                 │
│  Enum定義 (6種):                                                │
│    InputRouteType: TEXT / SCREEN / API / UNKNOWN                │
│    InputFreshness: FRESH / RECENT / AGING / STALE / FADED       │
│    NormalizationStatus: RAW / NORMALIZED / FRAGMENT / EMPTY     │
│    ContextLinkStatus: LINKED / PARTIAL / UNLINKED / BROKEN      │
│    DuplicateStatus: UNIQUE / DUPLICATE / NEAR_DUPLICATE /       │
│      SUPPRESSED                                                  │
│    RouteConflictStatus: NONE / PARALLEL / SINGLE_LINE_RISK      │
│                                                                 │
│  データ構造 (11種):                                              │
│    InputUnit          - 正規化済み入力単位                       │
│    ContextLink        - 文脈連結情報（継続入力の接続）           │
│    DuplicateRecord    - 重複判定情報（可逆的抑制）               │
│    RouteConflict      - 同時入力競合（排除せず保持）             │
│    ReceiveHistoryEntry - 受信履歴（生成・変化・減衰）            │
│    SuppressionHistoryEntry - 再投入抑制履歴（可逆）              │
│    DecayHistoryEntry  - 希薄化履歴                               │
│    TextDialogueConfig - 設定パラメータ（14項目）                 │
│    TextDialogueState  - 全体状態（17フィールド）                 │
│    HandoffResult      - process()の出力（8フィールド）           │
│                                                                 │
│  処理パイプライン (6段 + 安全弁):                                │
│    1. receive_input → InputUnit生成                              │
│    2. normalize_unit → 表記ゆれ・空入力・断片→共通単位           │
│    3. attach_context → 単発/継続区別、直前対話接続               │
│    4. align_to_percept_format → 経路差吸収（意味解釈なし）       │
│    5. detect_duplicates → 同一内容抑制、異内容並立保持           │
│    6. prepare_handoff → 受け渡し結果構築                         │
│    + apply_freshness_decay → 時間減衰                            │
│    + suppress_recent_adoption → 自己強化ループ防止               │
│    + ensure_format_diversity → 短文/長文片側支配防止             │
│    + restore_multi_route → 単一経路支配→複線復元                 │
│    + check_empty_streak → 空入力連続→保留（経路停止なし）        │
│    + filter_circular_reference → 同サイクル再受信防止            │
│                                                                 │
│  統合: merge_with_percept()                                      │
│    テキスト入力をPercept形式と同列統合（優先固定なし）           │
│    経路識別情報を_route_infoとして下流参照可能に保持             │
│    既存のemotion/intent判断を維持（本機能は判断しない）          │
│                                                                 │
│  orchestrator配線:                                              │
│    Phase 25b: text_dialogue_processor（外部からprocess()呼出）   │
│    process_text_input(): brain.pyからの呼出口                    │
│    save/load v10: text_dialogue_state フィールド追加 (34項目)    │
│    enrichment #17: 【記憶・内省】に「入力経路」行追加            │
│    systems: 41→42                                               │
│                                                                 │
│  brain.py配線:                                                   │
│    think_text(): テキスト入力のみの非ストリーミング版            │
│    think_streaming_text(): テキスト入力のみのストリーミング版    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.4d 自発起動経路

```
自発起動経路 実装仕様 (spontaneous_activation.py: 1,549行 / 84テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    外部入力がない局面でも内部状態の変化を処理可能にする。        │
│    外部入力経路を置き換えず、内部動機を単一路線へ固定せず、      │
│    起動成立だけで行動内容を確定せず、継続駆動を無制限化しない。  │
│    出力は起動候補情報としてのみ流し、判断・評価・行動決定を      │
│    直接起動しない。                                              │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  8断面入力 (duck-typed extraction):                              │
│    INTRINSIC_MOTIVATION  内的動機断面                            │
│    DIRECTION_VECTOR      方向断面                                │
│    UNFINISHED_INTENT     未完了意図断面                          │
│    MEMORY_ECHO           記憶残響断面                            │
│    EMOTIONAL_TRANSITION  感情推移断面                            │
│    RESPONSIBILITY        責任断面                                │
│    RECENT_ACTION         直近行動履歴断面                        │
│    EXTERNAL_INPUT_ABSENCE 外部入力有無断面                       │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  5段パイプライン:                                                │
│    1. 起動候補抽出（複数断面交差で成立）                         │
│    2. 起動条件整列（連続差分参照、単回変動の恒常化防止）         │
│    3. 競合整理（並立保持、未採択候補は消去せず次回候補化へ戻す） │
│    4. 起動可否判定用情報化                                       │
│    5. 受け渡し準備                                               │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  安全弁:                                                        │
│    - 連続採択系列の抑制（自己強化ループ防止）                    │
│    - 過密化時クールダウン有効化（起動間隔偏り緩和）              │
│    - 単線候補時の代替系列補充（複線候補へ復帰）                  │
│    - 鮮度減衰（単回判定が恒久状態にならない）                    │
│    - 未採択候補の再浮上経路維持                                  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  brain.py統合:                                                   │
│    think_spontaneous(): 非ストリーミング自発思考                 │
│    think_streaming_spontaneous(): ストリーミング自発思考         │
│    外部入力存在時は notify_external_input() で競合回避           │
│                                                                 │
│  orchestrator統合:                                               │
│    check_spontaneous_activation(): 起動候補チェック              │
│    post_response_update時: notify_external_input()               │
│    save/load v11 (35フィールド)、enrichment #18、systems 43      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.4b 価値方向性実運用検証

```
価値方向性実運用検証 実装仕様 (value_orientation_validation.py: 1,211行 / 88テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    価値方向性の変化が実運用でどのように現れるかを継続観測する。  │
│    価値方向性そのものを変更しない。検証結果を判断確定へ接続しない│
│    評価軸を単一化して出力傾向を矯正しない。観測結果による直接    │
│    介入を行わない。                                              │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  8断面入力:                                                      │
│    VALUE_ORIENTATION    価値方向性断面                            │
│    ACTION_CANDIDATES    行動候補断面                              │
│    SELECTION_HISTORY    選択履歴断面                              │
│    CONTEXT              文脈断面                                  │
│    EMOTION_TRANSITION   感情推移断面                              │
│    MEMORY_REFERENCE     記憶参照断面                              │
│    RESPONSIBILITY       責任断面                                  │
│    TIME_ELAPSED         時間経過断面                              │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  6段パイプライン:                                                │
│    1. 観測対象抽出（8断面から観測記録を生成）                     │
│    2. 観測単位正規化（共通検証記述へ統一、断面差保持）           │
│    3. 時系列整列（単回/継続分離、鮮度更新）                     │
│    4. 差分記述化（不一致・収束・再分岐を並立記録）              │
│    5. 検証出力化（報告情報形式のみ、判断起動しない）            │
│    6. 受け渡し準備（安全弁チェック+クリーンアップ）             │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  内部保持:                                                       │
│    観測記録集合、検証記述単位、時系列索引、差分履歴、             │
│    再分岐履歴、観測鮮度状態、保留観測履歴、希薄化履歴           │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  安全弁:                                                        │
│    - 収束偏向時の代替系列補充（複線記述へ復帰）                 │
│    - 観測欠落時の保留再評価（検証経路停止回避）                 │
│    - 断面横断の混在参照維持（特定断面支配防止）                 │
│    - 鮮度減衰と希薄化（単回検証結果の恒久化防止）              │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  orchestrator統合:                                               │
│    Phase 26b: _build_vo_validation_inputs → process              │
│    save/load v12 (36フィールド)、enrichment #19、systems 44      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.7.5 自発的内的動機

```
自発的内的動機 完全実装仕様 (intrinsic_motivation.py: 1,752行 / 113テスト):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  思想:                                                          │
│    目的系は候補生成と選択の仕組みはあるが、「なぜそれをしたいか」│
│    の動機源がない。自発的内的動機は、感情や傾向から湧き上がる    │
│    内的な推進力を弱く形成する。動機は価値や信念を固定しない。    │
│    行動の決定ではなく、内側で生じる「向き」の痕跡として          │
│    自己形成の前段条件を整える。                                  │
│                                                                 │
│  ID生成: uuid.uuid4().hex[:12] (12文字の一意識別子)             │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  Enum定義 (4種)                                                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  MotiveSourceType (動機の入力源分類):                            │
│    EMOTION        = "emotion"         感情状態由来              │
│    TENDENCY       = "tendency"        反復傾向由来              │
│    GOAL_VECTOR    = "goal_vector"     方向ベクトル由来          │
│    GOAL_CANDIDATE = "goal_candidate"  目的候補由来              │
│    MIXED          = "mixed"           複数ソース混合            │
│                                                                 │
│  MotiveAffinity (動機衝動の根拠・性質):                         │
│    EMOTIONAL_SURGE = "emotional_surge"  感情的高揚              │
│    HABITUAL        = "habitual"         習慣的                  │
│    DIRECTIONAL     = "directional"      方向的                  │
│    ASPIRATIONAL    = "aspirational"     志向的                  │
│    COMPOSITE       = "composite"        複合                    │
│    UNDEFINED       = "undefined"        未確定                  │
│                                                                 │
│  MotiveStrength (動機の強度):                                   │
│    STRONG    = "strong"     strength >= 0.7                      │
│    MODERATE  = "moderate"   strength >= 0.4                      │
│    WEAK      = "weak"       strength >= 0.2                      │
│    FAINT     = "faint"      strength >= 0.05                     │
│    UNDEFINED = "undefined"  strength < 0.05                      │
│                                                                 │
│  MotiveFreshness (動機の新鮮度):                                │
│    FRESH  = "fresh"    freshness >= 0.8                          │
│    RECENT = "recent"   freshness >= 0.6                          │
│    AGING  = "aging"    freshness >= 0.4                          │
│    STALE  = "stale"    freshness >= 0.15                         │
│    FADED  = "faded"    freshness < 0.15                          │
│                                                                 │
│  レベル判定関数 (pure):                                         │
│    determine_freshness_level(freshness: float)                  │
│      → MotiveFreshness                                          │
│    determine_strength_level(strength: float)                    │
│      → MotiveStrength                                           │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  Core Dataclasses (全て frozen=True)                             │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MotiveImpulse (動機衝動 — 動機に付帯する1単位の衝動)   │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    impulse_id: str               一意識別子              │   │
│  │    label: str                    emotion_joy, etc.       │   │
│  │    intensity: float              0.0〜1.0                │   │
│  │    valence: float                -1.0〜1.0               │   │
│  │    freshness: float              0.0〜1.0                │   │
│  │    reference_count: int          参照回数                │   │
│  │    affinity: MotiveAffinity      衝動の性質              │   │
│  │    timestamp: str                生成時刻                │   │
│  │    source_description: str       ソース記述              │   │
│  │                                                          │   │
│  │  メソッド:                                               │   │
│  │    get_freshness_level() → MotiveFreshness               │   │
│  │    with_freshness(f) → MotiveImpulse                     │   │
│  │    with_intensity(i) → MotiveImpulse                     │   │
│  │    with_reference() → MotiveImpulse                      │   │
│  │    reattach(affinity) → MotiveImpulse                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MotiveLink (根拠リンク)                                  │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    link_id: str                  一意識別子              │   │
│  │    motive_id: str                紐付く動機ID            │   │
│  │    source_type: MotiveSourceType 入力源                  │   │
│  │    source_description: str       ソース記述              │   │
│  │    contribution: float           寄与度 (0.0〜1.0)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MotiveEntry (動機エントリ — コア構造)                    │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    motive_id: str                一意識別子              │   │
│  │    motive_key: str               hashされた動機識別子    │   │
│  │    motive_summary: str           動機内容の要約          │   │
│  │    impulses: tuple[MotiveImpulse, ...]  複数衝動を並立   │   │
│  │    motive_links: tuple[str, ...] MotiveLink ID群         │   │
│  │    freshness: float              0.0〜1.0                │   │
│  │    reference_count: int          参照回数                │   │
│  │    creation_timestamp: str                               │   │
│  │    last_reference_timestamp: str                         │   │
│  │    revision_count: int           修正回数                │   │
│  │    undetermined_aspects: tuple[str, ...]                 │   │
│  │      生成時固定: ("motive_approximate",                  │   │
│  │                   "impulse_provisional")                  │   │
│  │                                                          │   │
│  │  変異メソッド:                                           │   │
│  │    with_freshness, with_reference, with_impulses,        │   │
│  │    revise_summary, with_added_impulse                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ MotiveStore (不変スナップショット)                       │   │
│  │                                                          │   │
│  │  Fields:                                                 │   │
│  │    entries: tuple[MotiveEntry, ...]                       │   │
│  │    motive_links: tuple[MotiveLink, ...]                  │   │
│  │    total_entries_created: int                             │   │
│  │    total_impulses_created: int                            │   │
│  │    total_revisions: int                                   │   │
│  │    total_expirations: int                                 │   │
│  │    average_freshness: float                               │   │
│  │    average_impulse_count: float                           │   │
│  │    active_entry_count: int                                │   │
│  │    timestamp: str                                         │   │
│  │    description: str                                       │   │
│  │                                                          │   │
│  │  フィルタメソッド:                                       │   │
│  │    has_entries() → bool                                   │   │
│  │    get_active_entries(stale_threshold=0.15)               │   │
│  │    get_entries_for_key(motive_key) → tuple                │   │
│  │  シリアライゼーション: to_dict() / from_dict()           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ IntrinsicMotivationConfig (設定 - 判断に影響しない)      │   │
│  │                                                          │   │
│  │    max_entries: int = 150                                 │   │
│  │    max_impulses_per_entry: int = 7                        │   │
│  │    base_decay_rate: float = 0.025                         │   │
│  │    impulse_decay_rate: float = 0.02                       │   │
│  │    freshness_boost_on_reference: float = 0.10             │   │
│  │    impulse_boost_on_reference: float = 0.06               │   │
│  │    stale_threshold: float = 0.15                          │   │
│  │    min_freshness_for_retention: float = 0.05              │   │
│  │    max_motive_links: int = 10                             │   │
│  │    min_intensity_for_motive: float = 0.1                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  抽出関数 (Pure, Duck Typing, None安全, dict/object両対応)       │
│  戻り値: list[(motive_key, label, intensity, valence,            │
│               source_description, source_type)]                  │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  extract_from_emotion_state(emotion, mood)                      │
│    EmotionVector: joy, anger, sorrow, fear, surprise, love, fun │
│    Mood: valence, arousal                                       │
│    各感情 >= 0.15 → label = f"emotion_{field_name}"             │
│    motive_key = "__emotion_motive__"                             │
│                                                                 │
│  extract_from_tendencies(tendencies_state)                      │
│    .tendencies[], .pattern.category.value, .strength            │
│    strength >= 0.02 → intensity = min(1.0, strength * 5.0)     │
│    motive_key = generate_motive_key(category_value)             │
│                                                                 │
│  extract_from_goal_vectors(vector_state)                        │
│    .vectors[], .vector_id, .direction(dict), .magnitude         │
│    magnitude >= 0.1 → dominant direction key                    │
│    motive_key = generate_motive_key(vector_id)                  │
│                                                                 │
│  extract_from_goal_candidates(candidate_state)                  │
│    .candidates[], .candidate_id, .category, .intensity          │
│    intensity >= 0.1                                              │
│    motive_key = generate_motive_key(candidate_id)               │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  計算関数 (Pure)                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  compute_motive_strength(impulses) → float                      │
│    加重集約: weight = 1.0 / (1.0 + i * 0.2)                    │
│                                                                 │
│  detect_motive_coexistence(entries) → list[(label_a, label_b)]  │
│    同一エントリ内の衝動ペア検出                                 │
│                                                                 │
│  compute_motive_overlay(entry) → dict[str, float]               │
│    目的候補参照時の動機同伴                                     │
│    effective_intensity = impulse.intensity * impulse.freshness   │
│    同ラベルはmax統合                                            │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  IntrinsicMotivationSystem                                       │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  主要メソッド:                                                   │
│    sense_motives(emotion, mood, tendencies, vectors, candidates) │
│      → MotiveStore                                               │
│    decay_motives() → MotiveStore                                 │
│    reference_motive(motive_key) → None                           │
│    get_motive_overlay(motive_key) → dict[str, float]             │
│    revise_motive(motive_key, new_summary) → None                 │
│    get_active_motives(max_count=20) → list                       │
│    get_store() / get_last_store()                                │
│                                                                 │
│  ═══════════════════════════════════════════════════════════════ │
│  統合・検証・永続化                                              │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                 │
│  統合関数:                                                       │
│    sense_from_chain(system, emotion, mood, tendencies,           │
│                     vectors, candidates) → MotiveStore           │
│    generate_motive_tags(store, scale) → list[dict]               │
│      INTRINSIC_MOTIVE_COUNT(0.06), _FRESHNESS(0.05),            │
│      _RICHNESS(0.07), _DOMINANT(0.08), _INTEGRATED(0.08)        │
│    get_motive_summary(store) → str                               │
│    get_motive_for_introspection(store) → dict                    │
│                                                                 │
│  検証関数 (5):                                                   │
│    verify_no_decision_impact                                     │
│    verify_no_goal_generation                                     │
│    verify_read_only_principle                                    │
│    verify_no_value_modification                                  │
│    verify_no_motivation_prescription (固有)                      │
│                                                                 │
│  永続化: save_motive_state / load_motive_state                   │
│  便利関数: create_config, create_empty_store, create_system      │
│                                                                 │
│  エクスポート総数: 36シンボル                                    │
│    Enum: 4, Dataclass: 5, System: 1, Helper: 7,                 │
│    Integration: 4, Persistence: 2, Convenience: 3,              │
│    Verification: 5, LevelFunc: 2, Extraction: 4,                │
│    Computation: 3                                                │
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
│  │ PHASE 2: 知覚 (Gemini 1st call — perception)                       │   │
│  │                                                                     │   │
│  │   [2.1] Gemini知覚コール                                            │   │
│  │         llm_wrapper.py: llm_call_with_image(                        │   │
│  │           VISION_SYSTEM_PROMPT, prompt, image)                      │   │
│  │         → 画面の客観的記述テキスト（200文字以内）                   │   │
│  │                                                                     │   │
│  │   [2.2] 知覚構造化                                                  │   │
│  │         perception.py: parse_percept(screen_description)            │   │
│  │         → Percept(emotion, intent, topics, valence)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 3: psyche処理 (脳内処理 — ローカル)                           │   │
│  │                                                                     │   │
│  │   [3.1] 全フェーズ更新                                              │   │
│  │         orchestrator.post_response_update(percept, delta)           │   │
│  │         → 感情・ムード・ドライブ・愛着・恐怖・自己モデル等         │   │
│  │                                                                     │   │
│  │   [3.2] 記憶検索                                                    │   │
│  │         recall_with_mood(percept, psyche, memory, top_k=3)          │   │
│  │                                                                     │   │
│  │   [3.3] 方針選択                                                    │   │
│  │         orchestrator.select_policy_dict(percept, memories)          │   │
│  │         → Phase 30-35 (思考候補→バイアス→沈黙候補→安定化)         │   │
│  │                                                                     │   │
│  │   [3.4] 沈黙判定                                                    │   │
│  │         is_silence_policy(policy) → 沈黙ならPhase4スキップ          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 3.5: 代弁 (Gemini 2nd call — expression)                     │   │
│  │                                                                     │   │
│  │   [3.5.1] render_expression()                                       │   │
│  │           expression.py: render_expression(                         │   │
│  │             state, policy, memories, persona, llm_call,             │   │
│  │             screen_context, recent_history)                         │   │
│  │           → {"text": "ふふっ♪", "meta": {emotion, intensity}}       │   │
│  │                                                                     │   │
│  │   [3.5.2] 応答分割                                                  │   │
│  │           分割位置: 。！？!?\n♪♥♡★☆                                 │   │
│  │           → ["ふふっ♪", "このゲーム面白そう"]                       │   │
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
│  │   [4.5] 外部送信 (オプション)                                       │   │
│  │         Base64 WAV → WebSocket → 外部クライアント                   │   │
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
emotional_memory_binding.py-     -      -      -       -       -        -        -        -      -     -      -
intrinsic_motivation.py    -     -      -      -       -       -        -        -        -      ○     -      -
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

OtherModelInputSupply (他者モデル入力供給):
  other_model_input_supply.py が STM・dynamics・psyche状態から計算し
  ContextSnapshot (ExternalContext互換) と ReactionLogProxy (STM互換) を生成
  供給単位に時刻・由来・欠損タグを必須化 / 循環参照防止 / 減衰と競合保持を常時有効

OtherAgentModel (他者モデル):
  外部文脈（ContextSnapshot） ─────┐
  反応ログ（ReactionLogProxy） ────┼→ other_agent_model.py → 内省記録層
  自己状態（対比参照のみ） ────────┘   (generates OtherModelStore)    → 記憶参照補助
  「相手がどう感じているか」の推測を仮説として弱く保持
  入力3系統 (全て Optional[Any] + duck typing, dict/object両対応):
    [Source 1: ExternalContext]
      duck typing: pace, weight, density, continuity, responsiveness
      → responsiveness >= 0.7: "engaged" (behavioral, strength=resp*0.6)
      → responsiveness <= 0.3: "disengaged" (behavioral, strength=(1-resp)*0.5)
      → weight >= 0.7: "heavy atmosphere" (contextual, strength=weight*0.5)
      → pace >= 0.7: "energetic exchange" (contextual, strength=pace*0.4)
      → 0.3 < resp < 0.7 AND 0.3 < weight < 0.7: "neutral" (strength=0.15)
    [Source 2: ReactionLog]
      duck typing: entries[], source_text, intent, emotion_label, valence
      処理上限: entries[:5]
      → intent=="question": "questioning intent" (behavioral, strength=0.4)
      → valence > 0.3: "positive tone" (behavioral, strength=valence*0.5)
      → valence < -0.3: "negative tone" (behavioral, strength=|valence|*0.5)
    [Source 3: SelfState + ExternalContext対比]
      self duck typing: intensity, description
      other duck typing: responsiveness, weight
      → |intensity - responsiveness| >= 0.4: contrast (strength=divergence*0.7)
      → |intensity - weight| >= 0.5: weight divergence (strength=weight_div*0.6)
  処理フロー:
    Extract(3関数) → Hypothesis生成(freshness=1.0, undetermined固定)
      → ObservationLink生成(contribution=max(0.1, 1.0-idx*0.15), max8件)
      → 競合検出(Jaccard: 同source異basis>=0.2, 異basis>=0.4)
      → Boundary計算(語彙Jaccard反転→divergence, aspects=inference_{basis})
      → Decay適用(ref_modifier=max(0.5, 1.0-ref*0.1))
      → 容量制限(weakest by (strength,freshness) tuple)
      → Snapshot(_build_store: 毎回競合再検出)
  内部構造:
    OtherStateHypothesis: frozen, 変異メソッド5種(with_freshness/strength/
      reference/competing, revise), basis=BEHAVIORAL/CONTEXTUAL/CONTRAST
    ObservationLink: 観測と仮説の弱い接続(contribution 0.0〜1.0)
    SelfOtherBoundary: 自己/他者の乖離度(divergence 0.0〜1.0)
      max_boundaries=10, 超過時FIFO(pop(0))
    OtherModelStore: frozen snapshot, avg round4桁
      フィルタ: get_active(>stale), get_strong(>0.5)
      シリアライゼーション: to_dict/from_dict (JSON roundtrip)
  ライフサイクル:
    生成時 freshness=1.0 → base_decay_rate=0.05 * ref_modifier/ターン で減衰
    strength: -0.03/ターン
    参照時 freshness+0.10ブースト, reference_count+1
    修正可能（revise → revision_count+1）, 競合許容（competing_ids相互リンク）
    stale_threshold(0.15) AND min_strength(0.05) 以下で自然消滅
  容量: max_hypotheses=60, max_boundaries=10, max_evidence=8
  タグ出力 (weight * scale):
    OTHER_MODEL_COUNT(0.06), _STRENGTH(0.07), _FRESHNESS(0.05),
    _COMPETITION(0.06), _BOUNDARY(0.05), _INTEGRATED(0.08)
    空/None時: COUNT(0.03)のみ
  Summary: header + 10フィールド + Top5(strength降順) + Integrated
  Introspection dict: source_distribution, basis_distribution,
    strongest_hypothesis_description[:120], 統計値
  固有検証: verify_no_intent_assertion（意図断定メソッド禁止）
    + verify_no_decision_impact, _no_goal_generation,
      _read_only_principle, _no_value_modification
  __init__.py エイリアス: 10シンボル名変更(衝突回避)
  非接続: 判断選択層・目的生成・価値更新・責任評価・外部出力直接生成

EmotionalMemoryBinding (感情記憶の紐づけ):
  短期記憶（STM） ─────────────┐
  感情状態（EmotionVector） ────┤
  長期記憶参照結果 ─────────────┼→ emotional_memory_binding.py → 記憶参照層
  エピソード記憶 ───────────────┘   (generates BindingStore)        → 内省記録層
  特定の記憶に感情が「染み付く」中長期の結びつきを管理
  入力4系統 (全て Optional[Any] + duck typing, dict/object両対応):
    [Source 1: STM]
      duck typing: entries[], source_text, emotion_label, raw_intensity, valence
      処理上限: entries[:10], raw_intensity < 0.1 or neutral → skip
      memory_key = generate_memory_key(source_text)  # MD5[:12]
    [Source 2: EmotionState + Mood]
      duck typing: joy, anger, sorrow, fear, surprise, love, fun (>= 0.15)
      memory_key = "__current_emotion_state__" (特殊センチネル)
      Mood duck typing: valence, arousal
    [Source 3: RecalledMemories]
      list[dict] with "summary", "keywords"
      デフォルト intensity=0.3 (弱い紐づけ)
    [Source 4: Episodes]
      duck typing: episodes[], episode_id, summary, emotional_companion, vividness
      vividness < 0.2 → skip
      coexisting_emotions は intensity * 0.5 で追加
  処理フロー:
    Extract(4関数) → memory_keyでグループ化
      → 既存binding → _merge_traces（同ラベルはmax intensity, 異ラベルは追加）
      → 新規binding → traces生成(max 7, freshness=1.0, affinity=CONCURRENT)
      → BindingLink生成(contribution=max(0.1, 1.0-idx*0.15), max10件)
      → Decay適用(binding: ref_modifier=max(0.5, 1.0-ref*0.05))
                  (trace: trace_ref_mod=max(0.5, 1.0-ref*0.08))
      → 全trace消滅 AND freshness < min(0.05) → 除去
      → 容量制限(weakest by (freshness, trace_count) tuple, max=200)
      → Snapshot(_build_store)
  核心機能:
    get_emotional_accompaniment(memory_key): 記憶再参照時に感情痕跡が「同伴」
      effective_intensity = trace.intensity * trace.freshness
      同ラベルは max で統合
      reference_binding も呼び出し（再参照で強化）
  内部構造:
    EmotionalTrace: frozen, 変異メソッド4種(with_freshness/intensity/reference, reattach)
      emotion_label: joy/anger/sorrow/fear/surprise/love/fun
      affinity: CONCURRENT/REACTIVATED/ACCUMULATED/COMPOSITE/UNDEFINED
    BindingLink: 紐づけの根拠リンク(contribution 0.0〜1.0)
    MemoryBinding: frozen, 変異メソッド5種(with_freshness/reference/traces/
      revise_summary/with_added_trace)
      memory_key: MD5ハッシュ記憶識別子
    BindingStore: frozen snapshot, avg round4桁
      フィルタ: get_active_bindings(>stale), get_bindings_for_memory(key)
      シリアライゼーション: to_dict/from_dict (JSON roundtrip)
  ライフサイクル:
    生成時 freshness=1.0 → base_decay_rate=0.02 * ref_modifier/ターン で減衰
    trace: trace_decay_rate=0.015 * trace_ref_mod/ターン
    参照時 binding freshness+0.08, trace freshness+0.05, reference_count+1
    修正可能（revise_summary → revision_count+1）
    stale_threshold(0.15), min_freshness(0.05): 全trace消滅+freshness<min で自然消滅
  容量: max_bindings=200, max_traces_per_binding=7, max_binding_links=10
  タグ出力 (weight * scale):
    EMOTIONAL_BINDING_COUNT(0.06), _FRESHNESS(0.05), _RICHNESS(0.07),
    _DOMINANT(0.08), _INTEGRATED(0.08)
    空/None時: COUNT(0.03)のみ
  Summary: header + 9フィールド + Top5(freshness降順) + Integrated
  Introspection dict: emotion_distribution, dominant_emotion,
    strongest_binding_summary[:120], 統計値
  固有検証: verify_no_emotion_evaluation（感情評価メソッド禁止）
    + verify_no_decision_impact, _no_goal_generation,
      _read_only_principle, _no_value_modification
  __init__.py エイリアス: 12シンボル名変更(衝突回避)
  非接続: 判断選択層・目的生成・価値更新・責任評価・外部出力直接生成

IntrinsicMotivation (自発的内的動機):
  感情状態（EmotionVector） ────┐
  気分（Mood） ─────────────────┤
  反復傾向（Tendencies） ───────┼→ intrinsic_motivation.py → 内省記録層
  方向ベクトル（Vectors） ──────┤   (generates MotiveStore)          → 目的候補層(弱い付随)
  目的候補（Candidates） ───────┘   (observation only, NO decision impact)
  感情や傾向から湧き上がる内的な推進力を弱く形成
  入力4系統 (全て Optional[Any] + duck typing, dict/object両対応):
    [Source 1: EmotionState + Mood]
      duck typing: joy, anger, sorrow, fear, surprise, love, fun (>= 0.15)
      motive_key = "__emotion_motive__" (特殊センチネル)
      label = f"emotion_{field_name}"
    [Source 2: Tendencies]
      duck typing: .tendencies[], .pattern.category.value, .strength
      strength >= 0.02 → intensity = min(1.0, strength * 5.0)
    [Source 3: GoalVectors]
      duck typing: .vectors[], .vector_id, .direction(dict), .magnitude
      magnitude >= 0.1 → dominant direction key
    [Source 4: GoalCandidates]
      duck typing: .candidates[], .candidate_id, .category, .intensity
      intensity >= 0.1
  処理フロー:
    Extract(4関数) → motive_keyでグループ化
      → 既存entry → _merge_impulses（同ラベルはmax intensity, 異ラベルは追加）
      → 新規entry → impulses生成(max 7, freshness=1.0, affinity=source由来)
      → MotiveLink生成(contribution=max(0.1, 1.0-idx*0.15), max10件)
      → Decay適用(entry: ref_modifier=max(0.5, 1.0-ref*0.05))
                  (impulse: impulse_ref_mod=max(0.5, 1.0-ref*0.08))
      → 全impulse消滅 AND freshness < min(0.05) → 除去
      → 容量制限(weakest by (freshness, impulse_count) tuple, max=150)
      → Snapshot(_build_store)
  核心機能:
    get_motive_overlay(motive_key): 目的候補参照時に動機が「同伴」
      effective_intensity = impulse.intensity * impulse.freshness
      同ラベルは max で統合
      reference_motive も呼び出し（再参照で強化）
  内部構造:
    MotiveImpulse: frozen, 変異メソッド4種(with_freshness/intensity/reference, reattach)
      label: emotion_joy, tendency_approach, vector_explore, etc.
      affinity: EMOTIONAL_SURGE/HABITUAL/DIRECTIONAL/ASPIRATIONAL/COMPOSITE/UNDEFINED
    MotiveLink: 根拠リンク(contribution 0.0〜1.0)
    MotiveEntry: frozen, 変異メソッド5種(with_freshness/reference/impulses/
      revise_summary/with_added_impulse)
      motive_key: MD5ハッシュ動機識別子
    MotiveStore: frozen snapshot, avg round4桁
      フィルタ: get_active_entries(>stale), get_entries_for_key(key)
      シリアライゼーション: to_dict/from_dict (JSON roundtrip)
  ライフサイクル:
    生成時 freshness=1.0 → base_decay_rate=0.025 * ref_modifier/ターン で減衰
    impulse: impulse_decay_rate=0.02 * impulse_ref_mod/ターン
    参照時 entry freshness+0.10, impulse freshness+0.06, reference_count+1
    修正可能（revise_summary → revision_count+1）
    stale_threshold(0.15), min_freshness(0.05): 全impulse消滅+freshness<min で自然消滅
  容量: max_entries=150, max_impulses_per_entry=7, max_motive_links=10
  タグ出力 (weight * scale):
    INTRINSIC_MOTIVE_COUNT(0.06), _FRESHNESS(0.05), _RICHNESS(0.07),
    _DOMINANT(0.08), _INTEGRATED(0.08)
    空/None時: COUNT(0.03)のみ
  Summary: header + 9フィールド + Top5(freshness降順) + Integrated
  Introspection dict: impulse_distribution, dominant_impulse,
    strongest_motive_summary[:120], 統計値
  固有検証: verify_no_motivation_prescription（動機処方メソッド禁止）
    + verify_no_decision_impact, _no_goal_generation,
      _read_only_principle, _no_value_modification
  __init__.py エイリアス: 10シンボル名変更(衝突回避)
  非接続: 判断選択層・価値更新層・責任評価層・外部出力直接生成層

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
├── __init__.py                    (1183行) - エクスポート定義
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
├── other_model_input_supply.py  (308行) - 他者モデル入力供給（external_context / reaction_log 生成）
├── other_model_real_feed.py  (1,481行) - 他者モデルリアルフィード統合（8観測断片・10段パイプライン・安全弁）
├── text_dialogue_input.py   (1,559行) - テキスト対話入力経路（6段パイプライン・経路多様性・重複抑制・安全弁）
├── spontaneous_activation.py (1,549行) - 自発起動経路（8断面交差・5段パイプライン・競合並立・安全弁）
├── value_orientation_validation.py (1,211行) - 価値方向性実運用検証（8断面・6段パイプライン・差分並立・安全弁）
├── emotional_memory_binding.py (1708行) - 感情記憶の紐づけ（中長期感情痕跡）
├── intrinsic_motivation.py    (1752行) - 自発的内的動機（感情・傾向由来の内的推進力）
├── responsibility.py              (480行)  - 責任記録・評価
├── responsibility_manager.py      (210行)  - 責任マネージャー
├── responsibility_dispersion.py   (1039行) - 責任の発散・昇華
├── silence_hesitation.py          (724行)  - 沈黙・躊躇い表現
├── tone.py                        (698行)  - トーン・ユーモア制御
├── thought.py                     (473行)  - 思考候補生成・選択（15ポリシー動的選択）
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
├── test_other_model_input_supply.py (330行)
├── test_other_model_real_feed.py (1,006行)
├── test_text_dialogue_input.py  (1,025行)
├── test_spontaneous_activation.py (812行)
├── test_value_orientation_validation.py (1,039行)
├── test_emotional_memory_binding.py (1142行)
├── test_intrinsic_motivation.py (1157行)
├── test_tone.py                   (592行)
├── test_transient_goal.py         (664行)
├── test_value_orientation.py      (599行)
├── test_context_sensitivity.py    (704行)
├── test_perception.py             (661行)
├── test_expression.py             (695行)
├── test_reaction.py               (813行)
├── test_reaction_with_stm.py      (993行)
├── test_fear.py                   (528行)
├── test_memory_link.py            (598行)
├── test_short_term_memory.py      (1061行)
├── test_responsibility_manager.py (728行)
├── test_pillar_managers.py        (959行)
├── test_orchestrator.py           (1,229行)
└── test_phase_chain_integration.py (825行)
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
| 5 | 他者モデル | 「相手がどう感じているか」の推測構造。自己と他者の境界が構造として存在しない | context_sensitivity, self_model | 完了 |
| 6 | 感情記憶の紐づけ | 特定の記憶に感情が染み付く仕組み。stm_emotion_couplingは短期の連動のみ | stm_emotion_coupling, short_term_memory | 完了 |
| 7 | 自発的内的動機 | 感情や傾向から欲求が湧き上がる構造。goal系は候補生成と選択の仕組みだが「なぜそれをしたいか」の動機源がない | proto_goal_vector, repeated_tendency, multi_emotion | 完了 |
| 8 | 他者モデル入力供給 | other_agent_modelのexternal_context/reaction_logが常にNoneだった問題を解消。STM・dynamics・psyche状態から入力を生成しorchestrator経由で供給 | other_agent_model, short_term_memory, orchestrator | 完了 |
| 9 | orchestrator未接続入力の配線 | Phase 19/20/33でexternal_context=None固定、Phase 21でmemories=None固定。input_supplyのContextSnapshotおよびbrain.pyのrecalled_memoriesを渡す配線が必要 | orchestrator, self_narrative, episodic_memory, emotional_memory_binding, context_sensitivity | 完了 |
| 10 | save/load未対応モジュールの永続化 | repeated_tendency, proto_goal_vector, goal_candidates, transient_goal, stability_valveをorchestratorのsave/loadに追加。scoped_goalは設計上エフェメラルのため対象外。snapshot v4→v5 (27フィールド) | orchestrator, stability_valve | 完了 |
| 11 | save/load v5→v6: 残り3モジュール永続化 | responsibility_dispersion, context_sensitivity, stm_emotion_couplingの3モジュールをorchestratorのsave/loadに追加。CouplingInfluenceにto_dict/from_dict新規追加。snapshot v5→v6 (30フィールド) | orchestrator, stm_emotion_coupling | 完了 |

### 8.1 未接続入力の配線 (#9) — 完了

orchestrator.py 内で切断されていた4箇所を接続済み:

| Phase | モジュール | 引数 | 接続先 |
|-------|-----------|------|--------|
| 19 | self_narrative | external_context | supply_context(self._input_supply) — 文脈由来のナラティブ断片が生成される |
| 20 | episodic_memory | external_context | supply_context(self._input_supply) — エピソード記録に外部文脈が反映される |
| 21 | emotional_memory_binding | memories | self._last_recalled_memories — brain.pyのrecall_with_mood結果を受け取る |
| 33 | context_sensitivity | context | supply_context(self._input_supply) → ExternalContext変換 — 実際の空気読みが機能する |

brain.py側: recall_with_mood後に orchestrator.set_recalled_memories(memories) を呼び出す。

### 8.2 save/load永続化対応 (#10) — 完了

snapshot v5 (27フィールド) で以下5モジュールの永続化を追加:

| モジュール | snapshotキー | 状態クラス | 対応内容 |
|-----------|-------------|-----------|----------|
| repeated_tendency | tendency_state | RepeatedTendencyState | to_dict/from_dict 既存 → save/load追加 |
| proto_goal_vector | vector_state | VectorState | to_dict/from_dict 既存 → save/load追加 |
| goal_candidates | candidate_state | CandidateState | to_dict/from_dict 既存 → save/load追加 |
| transient_goal | transient_goal_state | TransientGoalState | to_dict/from_dict 既存 → save/load追加 |
| stability_valve | stability_valve | StabilityValve | to_dict/from_dict 新規追加 → save/load追加 |

**対象外**: scoped_goal — 設計上エフェメラル（メモリ内のみ、永続化しない）

### 8.3a save/load永続化対応 (#11) — 完了

snapshot v6 (30フィールド) で以下3モジュールの永続化を追加:

| モジュール | snapshotキー | 状態クラス | 対応内容 |
|-----------|-------------|-----------|----------|
| responsibility_dispersion | dispersion_state | DispersionState | to_dict/from_dict 既存（Pydantic） → save/load追加 |
| context_sensitivity | context_sensitivity_state | ContextState | to_dict/from_dict 既存 → save/load追加 |
| stm_emotion_coupling | last_coupling | CouplingInfluence | to_dict/from_dict 新規追加 → save/load追加 |

### 8.3 全設計書 × 全実装 照合レポート — 2026-02-12

設計書39本 × 実装53ファイル × orchestrator配線（Phase 1-35）を網羅照合した結果。

#### 8.3.1 tick配線（入出力）: 全モジュール正常接続 ✅

Phase 1-7（毎tick）、Phase 8-14（3tick毎）、Phase 15-26（5tick毎）、Phase 27-29（10tick毎）、Phase 30-30b-31-35（policy生成時: 30b=候補拡張）
— 全35フェーズの入力・出力は設計書通りに配線済み。ミスマッチなし。

#### 8.3.2 設計書あり → 実装なし（0件） ✅

全設計書の実装が完了。

| 設計書 | 想定モジュール | 備考 |
|--------|---------------|------|
| design_long_term_simulation.md | tools/long_term_sim.py | 実装完了 ✅ |

※ design_logging_production.md → 実装完了 ✅（INFO→DEBUG降格 + configure_logging統合）

#### 8.3.3 個別設計書なし → 統合設計書内に記述済み（11件）

初回コミット(2026-01-31)で統合設計書と同時に実装されたモジュール群。
個別の `design_モジュール名.md` は存在しないが、設計自体は以下の統合設計書に含まれている。

| Python モジュール | 設計根拠 | 備考 |
|-------------------|----------|------|
| attachment_manager.py | design_loss.md | 4柱の1つ |
| identity_manager.py | design_loss.md, identity.md | 4柱の1つ |
| continuity_manager.py | design_loss.md | 4柱の1つ |
| projection_manager.py | design_loss.md | 4柱の1つ |
| reaction_with_stm.py | design_short_term_loop.md | Phase 1の中核反応処理 |
| perception.py | final_architecture_spec.md | brain.py 2-call構造の知覚側 |
| expression.py | final_architecture_spec.md | brain.py 2-call構造の代弁側 |
| thought.py | design_psyche.md, design_spec.md | Phase 30の候補生成 |
| memory_link.py | final_architecture_spec.md | brain.pyの記憶想起連携 |
| reaction.py | design_psyche.md | 基本反応処理 |
| snapshot.py | design_persistence.md | スナップショットデータクラス |

#### 8.3.4 save/load 永続化の欠落 — 解消済み ✅

v6 (28フィールド) で以下3件を追加し、全モジュールの永続化が完了:

| モジュール | 状態クラス | 対応 |
|-----------|-----------|------|
| responsibility_dispersion | DispersionState | v6で追加 ✅ |
| context_sensitivity | ContextState | v6で追加 ✅ |
| stm_emotion_coupling | CouplingInfluence | v6で追加（to_dict/from_dict新規実装） ✅ |

※ responsibility_manager / long_term_dynamics は別途JSONファイルに自力永続化済み
※ scoped_goal は設計上エフェメラルのため対象外

#### 8.3.5 get_prompt_enrichment() — 全項目配線完了 ✅

Geminiへの心理コンテキスト注入: 5セクション構成、全項目配線済み。
Phase 30-35 の判断バイアス群は `_generate_final_candidates()` で計算後に `self._last_*` にキャッシュし、enrichment で参照する。

#### 8.3.6 命名の不整合 — 解消済み ✅

| ファイル | 対応 |
|---------|------|
| SELF_NARRATIVE_DESIGN.md → design_self_narrative.md | リネーム完了 ✅ |

#### 8.3.7 照合サマリ

| カテゴリ | 状態 |
|---------|------|
| tick配線（入出力） | 35/35 ✅ |
| 設計→実装 | 39/39 ✅ |
| 実装→設計 | 53/53 ✅ （11件は統合設計書内に記述） |
| save/load | 30/30 ✅ （v6で完了、v12で36フィールド） |
| prompt enrichment | 全項目配線完了 ✅ |
| 命名不整合 | ✅ 解消済み |

---

## 9. 今後の実装計画（優先順）

psyche内部の設計・実装・配線・永続化・enrichmentは全完了。
以下は「工学的自我」の実現に向けて不足している領域を、依存関係とリスクの低い順に整理したもの。

### 9.1 実装順序

| 順序 | 項目 | リスク | 依存 | 概要 |
|------|------|--------|------|------|
| ① | テスト追加（12モジュール） ✅完了 | ゼロ | なし | 9ファイル675テスト追加済（2,131→2,806） |
| ② | ポリシー候補拡張 ✅完了 | 低 | ① | policy_candidate_expansion.py (1,388行/86テスト) 8断面×10軸。orchestrator Phase 30b、save/load v7 (31フィールド) |
| ③ | 記憶系統統合 ✅完了 | 中 | ① | memory_system_integration.py (1,132行/93テスト) 3系統正規化・重複並立・競合併存。orchestrator Phase 21b、save/load v8 (32フィールド) |
| ④ | 他者モデルへのリアルフィード ✅完了 | 中 | ③ | other_model_real_feed.py (1,481行/102テスト) 8観測断片・10段パイプライン。orchestrator Phase 25a、save/load v9 (33フィールド) |
| ⑤ | 入力経路拡充（テキスト対話） ✅完了 | 中〜高 | ①〜④ | text_dialogue_input.py (1,559行/102テスト) 6段パイプライン・経路多様性。orchestrator Phase 25b、save/load v10 (34フィールド)。brain.py think_text/think_streaming_text追加 |
| ⑥ | 自発性の追加 ✅完了 | 高 | ①〜⑤ | spontaneous_activation.py (1,549行/84テスト) 8断面交差・5段パイプライン。orchestrator check_spontaneous_activation()、save/load v11 (35フィールド)。brain.py think_spontaneous/think_streaming_spontaneous追加 |
| ⑦ | value_orientation 実運用検証 ✅完了 | 低 | ⑥ | value_orientation_validation.py (1,211行/88テスト) 8断面・6段パイプライン。orchestrator Phase 26b、save/load v12 (36フィールド)。Phase 26のバグ修正（update_orientation引数不正） |
| ⑧ | value_orientation未接続関数の接続 ✅完了 | 低 | ⑦ | 実装済み4関数をorchestratorに接続: Phase 26に責任シグナル追加、Phase 35bに価値軸バイアス適用、select_policy_dict後にupdate_from_decision |
| ⑨ | 記憶の忘却と固定化 ✅完了 | 低 | なし | memory_forgetting_fixation.py (1,052行/85テスト) 8断面・6段パイプライン・段階忘却・復帰経路。orchestrator Phase 21c、save/load v13 (37フィールド) |
| ⑩ | 行動-結果の観測と蓄積 ✅完了 | 中 | ⑧ | action_result_observation.py (1,626行/109テスト) 8断面・6段パイプライン・非正誤判定・時系列隣接記録。orchestrator Phase 7a/26c、save/load v14 (38フィールド) |
| ⑪ | 他者観測の長期蓄積と仮説補助 ✅完了 | 中 | ⑩ | other_model_dialogue_learning.py (1,625行/135テスト) 8断面・8段パイプライン・相手別分離・反復非反復等重量・仮説再生成方式。orchestrator Phase 25c、save/load v15 (39フィールド) |
| ⑫ | メタ感情認知と変動候補生成 ✅完了 | 中〜高 | ⑨⑩ | meta_emotion_cognition.py (1,608行/141テスト) 8断面・7段パイプライン・常時等価候補列挙・Phase 1-2不変性保証。orchestrator Phase 14b、save/load v16 (40フィールド) |
| ⑬ | 自己行動知覚 ✅完了 | 低 | ⑩ | self_action_perception.py (395行/114テスト) 3段パイプライン・全記録等価・テキスト非解釈。orchestrator notify_self_output()/enrichment #24、brain.py 6メソッド通知追加、action_result output_text補完、save/load v17 (41フィールド) |
| ⑭ | 予期差分の参照経路拡張 ✅完了 | 低 | ⑩⑬ | 新モジュールなし（orchestrator.py拡張のみ）。Phase 26d差分記録の多断面化（予期/行動/結果/文脈の4断面）、get_expectation_diff_summary()アクセサ、enrichment #25、save/load v18 (42フィールド) |
| ⑮ | 内部状態→行動経路の接続強化 ✅完了 | 低 | ⑭ | 新モジュールなし。expression.py _build_render_prompt()を3層構造に改善（行動制約/状況/内面的文脈）、EXPRESSION_SYSTEM_PROMPT に入力読み方セクション追加、ポリシー遵守率向上のためのプロンプト構造改善 |
| ⑯ | 意図-行動間の乖離認知 ✅完了 | 低 | ⑬⑮ | intent_action_gap.py (397行/129テスト) 3段パイプライン（対構成→多断面記述→蓄積参照）・全記録等価・パターン抽出禁止・3経路遮断。orchestrator Phase 26e、enrichment #26、save/load v19 (43フィールド) |
| ⑰ | 時間認知構造 ✅完了 | 低 | なし | temporal_cognition.py (617行/149テスト) 3段パイプライン（経過蓄積→6断面特徴量記述→参照提供）・スライディングウィンドウ・段階値列挙型・4経路遮断。orchestrator Phase 7b/14c、enrichment #27、save/load v20 (44フィールド) |
| ⑱ | 記憶の多経路想起 ✅完了 | 低 | ③⑨⑰ | multi_path_recall.py (807行/105テスト) 3経路想起（感情連想/文脈連想/時間近接）・経路等価性・顕著性バイアス抑制・ルーミネーション防止・忘却分離。orchestrator Phase 21d、enrichment #28、save/load v21 (45フィールド) |
| ⑲ | 内省断面間の横断的記述 ✅完了 | 低 | ⑫⑯⑰ | introspection_cross_section.py (729行/130テスト) 3段パイプライン（断面値収集→スナップショット蓄積→参照受渡）・6断面並置（self_model/temporal_self_difference/identity_coherence/self_narrative/introspection_consumption/meta_emotion_cognition）・全断面等価・パターン抽出禁止・統合禁止・5経路遮断。orchestrator Phase 14d、enrichment #29、save/load v22 (47フィールド) |
| ⑳ | 知覚入力の内部文脈化 ✅完了 | 低 | ⑰⑱ | perceptual_context.py (646行/116テスト) 3段パイプライン（知覚サマリ蓄積→4断面段階値記述→参照受渡）・感情ラベル変化頻度/意図ラベル変化頻度/話題重複度/感情価推移方向・テキスト比較禁止・topics意味判定禁止・4経路遮断。orchestrator Phase 7c/14e、enrichment #30、save/load v22 (47フィールド) |
| ㉑ | 内省ウィンドウ拡大 ✅完了 | 最小 | ⑲ | introspection_cross_section.pyのウィンドウサイズ10→25に拡大（enrichment出力は直近10件のみ維持）。反固定化第1段階（パラメータ変更のみ）。討論結果: 条件付き推奨 |
| ㉒ | スコアリングの構造的揺らぎ ✅完了 | 低 | ⑳ | scoring_fluctuation.py (647行/テスト内) 5段パイプライン（変動量抽出→合成→制限→ポリシー別生成→加算）・内部状態由来（感情/STM/drives/経過時間）・振幅上限<ValueOrientation(+-5%)・状態蓄積なし・安全弁5種。orchestrator Phase 35c（最後の加算層）。反固定化第2段階。討論結果: 条件付き推奨。解析結果: 低固定化リスク |
| ㉓ | 選択帰属 ✅完了 | 低 | ⑧ | selection_attribution.py (413行/87テスト) 選択事実のREAD-ONLY記録（候補群構成+選択ラベル+バイアス源構成+ティック+タイムスタンプ）・全記録等価・パターン抽出禁止・5経路遮断・enrichment等価列挙（バイアス情報遮断）。Cycle 3候補3拡張: バイアス源名一覧をスコアなしで併記。orchestrator select_policy後record_selection()、enrichment #31、save/load v23 (48フィールド)。Agency第1段階 |
| ㉔ | 参照頻度の構造的記述 ✅完了 | 低 | - | reference_frequency_description.py (782行/93テスト) 12箇所横断読み取り専用集約層・断面構成（集中度/偏在度）・FIFO断面履歴（30件上限）・変動記述（再導出型）・enrichment直接露出遮断・忘却経路遮断・想起経路遮断・安全弁5種。orchestrator Phase 24b、save/load v24 (49フィールド)。反固定化第3段階。討論結果: 条件付き推奨。設計解析: 低固定化リスク。実装解析: 低固定化リスク |
| ㉕ | 持続的コミットメント ✅完了 | 中→低（修正済） | ⑧ | persistent_commitment.py (1,037行/73テスト) transient_goal昇格が唯一生成経路・複数並行保持（上限付き）・強度依存非線形減衰（飽和構造）・慣性時間減衰・4解除条件（時間/内部状態/競合/達成認知）・認知記録FIFO・資源競合（揺らぎ付き帯域分配）・バイアス上限<VO(+-5%)・安全弁6種・自己強化ループ4重遮断。orchestrator Phase 12b/35b2、enrichment #32、save/load v25 (50フィールド)。Agency第2段階。討論結果: 条件付き推奨（7条件）。設計解析: 低固定化リスク。実装解析: 中→修正後低 |
| ㉗ | Agency Stage 3: 不整合度サマリー ✅完了 | - | ㉕ | 討論結果: 要再検討（新モジュール不要）。定量評価: 既存構造(persistent_commitment+stability_valve+context_sensitivity)で-20~-30%の間接的抵抗カバー済み。残ギャップ「不整合の明示的記述」をenrichment追加で解消。orchestrator get_prompt_enrichment()に内部-外部間張力サマリー追加（32行）。保持方向バイアス/外部文脈慎重度/価値軸傾斜の3断面をREAD-ONLY参照。張力情報なしの場合は出力しない。テスト+2 (4,985) |
| ㉘ | 安定化の構造的記述 ✅完了 | 低 | ㉔ | stabilization_description.py (512行/71テスト) 2断面限定（信号源多寡+時間的変動幅）・横断読み取り層（6信号源二値読み取り+temporal_self_difference差分参照）・3段パイプライン（読み取り→断面構成→FIFO蓄積）・enrichment直接露出遮断・忘却経路遮断・パターン抽出禁止・全記録等価・安全弁5種。orchestrator Phase 15b、save/load v26 (51フィールド)。反固定化第4段階。討論結果: 条件付き推奨（7条件）。設計解析: 低固定化リスク。実装解析: 低固定化リスク |
| ㉖ | 結合テスト強化 ✅完了 | - | - | orchestratorスモークテスト14件（全パス通過・select_policy_dict・Phase発火・enrichment検証・連続稼働安定性）+ Phase間連鎖動作テスト48件（every-tick/3-tick/5-tick/10-tick/policy-selection/cross-phase data flow）。test_orchestrator.py (1,229行) + test_phase_chain_integration.py (825行)。総テスト数: 4,983 |

### 9.2 各項目の詳細

#### ① テスト追加（12モジュール） ✅完了

9テストファイル・675テスト追加（2,131→2,806テスト）:

| テストファイル | テスト数 | 対象モジュール |
|---------------|---------|---------------|
| test_perception.py | 90 | perception.py |
| test_expression.py | 60 | expression.py |
| test_reaction.py | 73 | reaction.py |
| test_reaction_with_stm.py | 77 | reaction_with_stm.py |
| test_fear.py | 53 | fear.py |
| test_memory_link.py | 43 | memory_link.py |
| test_short_term_memory.py | 104 | short_term_memory.py |
| test_responsibility_manager.py | 44 | responsibility_manager.py |
| test_pillar_managers.py | 131 | attachment/continuity/identity/projection_manager.py |

#### ② ポリシー候補拡張 ✅完了
policy_candidate_expansion.py (1,388行/86テスト)
- 8入力断面（感情/記憶/傾向/責任/対話/自己観測/他者推定/目的）を特徴断片に変換
- 10候補軸（接近/保留/探索/転換/維持/修復/境界調整/確認/委譲/内省反映）の活性度を都度再決定
- 複数断面の交差で候補生成（単一断面支配を抑制）
- 候補履歴（残存+希薄化）、抑制履歴（可逆）、競合履歴（未採択保持+再注入）
- 単線化警告+代替補充、抑制恒常化検知+緩和
- orchestrator Phase 30b統合、save/load v7（31フィールド）、enrichment #14

#### ③ 記憶系統統合 ✅完了
memory_system_integration.py (1,132行/93テスト)
- 3系統（episodic / long_term / binding）を共通記述単位(UnifiedMemoryUnit)に正規化
- 重複調整は統合消去ではなく、同一事象の複数視点を並立保持
- 競合保持は矛盾を解消せず併存させ、揺らぎ情報として保持
- 出所横断の混在提示を維持し、単一視点への収束を防止
- 参照履歴と再利用履歴は累積と希薄化を併置（可逆）
- 競合不可視化の自動復元（安全弁）、直近再採用抑制
- orchestrator Phase 21b統合、save/load v8（32フィールド）、enrichment #15

#### ④ 他者モデルへのリアルフィード ✅完了
other_model_real_feed.py (1,481行/102テスト)
- 8種の観測断片（発話反応・応答間隔・話題遷移・感情トーン・継続関与・拒否受容・文脈整合・直近履歴）を抽出
- 10段処理パイプライン: 正規化→整列→重複統合→競合併存→鮮度減衰→系列抑制→多様性確保→収束安全弁→停滞安全弁→出力制限
- enhance_context_with_feed() で既存 ContextSnapshot を差分調整（上書きしない）
- 競合観測は排除せず並立保持、単一解釈収束時は holdback から補充
- orchestrator Phase 25a統合、save/load v9（33フィールド）、enrichment #16

#### ⑤ 入力経路拡充 ✅完了
- text_dialogue_input.py (1,559行/102テスト)
- 6段パイプライン: 受信→正規化→文脈付与→既存形式整合→重複調整→受け渡し
- 安全弁: 空入力連続→保留、単一経路支配→複線復元、形式多様性維持、循環参照防止、自己強化ループ防止
- orchestrator Phase 25b、save/load v10 (34フィールド)、enrichment #17、systems 41→42
- brain.py: think_text() / think_streaming_text() 追加（テキスト入力のみの思考経路）

#### ⑥ 自発性の追加 ✅完了
spontaneous_activation.py (1,549行/84テスト)
- 8断面入力: 内的動機・方向・未完了意図・記憶残響・感情推移・責任・直近行動・外部入力有無
- 5段パイプライン: 候補抽出(断面交差)→条件整列(連続差分)→競合整理(並立保持)→可否判定→受渡
- 安全弁: 連続採択抑制、過密化クールダウン、単線候補時代替補充、鮮度減衰、未採択再浮上
- orchestrator check_spontaneous_activation()、save/load v11 (35フィールド)、enrichment #18、systems 42→43
- brain.py: think_spontaneous() / think_streaming_spontaneous() 追加（外部入力なし時の自発思考経路）

#### ⑦ value_orientation 実運用検証 ✅完了
value_orientation_validation.py (1,211行/88テスト)
- 8断面入力（価値方向性/行動候補/選択履歴/文脈/感情推移/記憶参照/責任/時間経過）
- 6段パイプライン: 観測対象抽出→観測単位正規化→時系列整列→差分記述化→検証出力化→受け渡し準備
- 差分は不一致・収束・再分岐を並立記録（単線的結論化を回避）
- 安全弁: 収束偏向→代替系列補充、観測欠落→保留再評価、断面支配→混在参照復元
- 検証出力は報告情報形式のみ（判断・評価・行動決定を直接起動しない）
- orchestrator Phase 26b統合、save/load v12（36フィールド）、enrichment #19、systems 44
- Phase 26バグ修正: update_orientation()の引数不正（signal_type/signal_value → emotion_signal）

#### ⑧ value_orientation未接続関数の接続 ✅完了
- value_orientation.pyに実装済みだがorchestratorから呼ばれていない4関数を接続
- update_from_decision(): select_policy_dict後にポリシー選択結果を価値軸へフィードバック（超高慣性の微小更新）
- generate_responsibility_signal(): Phase 26で感情に加え責任シグナルも価値軸更新に反映
- apply_orientation_to_candidates(): Phase 35b として候補スコアに価値軸バイアスを適用
- 設計書不要（既存実装済みコードの接続のみ）

#### ⑨ 記憶の忘却と固定化 ✅完了
- memory_forgetting_fixation.py: 1,052行 / 85テスト
- 8断面入力: 記憶参照頻度、再利用間隔、時系列、競合系列、感情連結、文脈連結、保護状態、固定化兆候
- 6段パイプライン: 忘却候補抽出→固定化兆候抽出→候補整列→競合保持→段階忘却情報化→受け渡し準備
- 段階忘却: ACTIVE→WEAKENING→FADING→NEAR_INVISIBLE→INVISIBLE（可逆）
- 固定化レベル: NONE→MILD→MODERATE→STRONG（交差断面閾値）
- 安全弁: 収束偏向警告→代替系列補充、過密化→進行緩和、自己強化ループ防止
- orchestrator Phase 21c、save/load v13 (37フィールド)、enrichment #20

#### ⑩ 行動-結果の観測と蓄積 ✅完了
- action_result_observation.py: 1,638行 / 128テスト
- Cycle 3候補6拡張: input_pathway_label（入力経路ラベル）を行動記録に併記。enrichment遮断
- 8断面入力: 直近行動、外部反応、内部状態変化、感情推移、文脈、時間経過、他者観測、記憶参照
- 6段パイプライン: 対構成→多断面評価記述→文脈帰属付与→整列蓄積→減衰忘却→受渡準備
- 非正誤判定: 結果を成功/失敗で二値評価しない。複数断面で並立記述
- 時系列隣接記録: 因果断定ではなく「行動Xの後に結果Yが観測された」のみ
- 段階忘却: ACTIVE→WEAKENING→FADING→NEAR_INVISIBLE→INVISIBLE（memory_forgetting_fixationパターン準拠）
- 安全弁4種: パターン収束警告、断面偏り警告、シグナル供給強度自動減衰、バッファ過密制御
- 即時構成禁止: 構成バッファで行動→結果に数ティックのバッファ必須
- orchestrator Phase 7a(毎ティック行動記録)/Phase 26c(5ティック毎結果処理)、save/load v14 (38フィールド)、enrichment #21

#### ⑪ 他者観測の長期蓄積と仮説補助 ✅完了
- other_model_dialogue_learning.py: 1,625行 / 135テスト
- 8断面入力: 短期観測断片、行動-結果他者断面、対話文脈、相手識別、感情トーン、反応間隔、話題遷移、蓄積鮮度
- 8段パイプライン: 蓄積候補抽出→正規化・文脈付与→相手別整列→反復・非反復識別→仮説再生成材料構成→競合並立整理→減衰・忘却→受渡準備
- 相手別分離管理: user_idごとに独立蓄積（本モジュール内限定、既存other_agent_model変更なし）
- 反復+非反復の等重量蓄積: 確認バイアスの構造化防止
- 仮説再生成方式: strength直接加算禁止、再生成+競合並立
- 因果帰属禁止: ⑩から継承、時系列的隣接記録のみ
- 永続化データの有限寿命: セッション境界で鮮度一律減衰
- 安全弁4種: 他者像単一化防止、確認バイアス防止、自己成就的予言防止、相手別偏り検出
- 位置づけ: 入力供給層→短期観測層→**長期蓄積層（本機能）**→推測層
- orchestrator Phase 25c（Phase 25aと25の間）、save/load v15 (39フィールド)、enrichment #22

#### ⑫ メタ感情認知と変動候補生成 ✅完了
- meta_emotion_cognition.py: 1,628行 / 155テスト
- Cycle 3候補8拡張: boundary_dimensions/boundary_count（境界値到達の事実記述）をTransitionFeatureに追加。enrichment遮断
- 元の計画名「感情調整戦略」から討論を経て名称変更（「調整」は規範的暗示を含むため）
- Phase 1-2のパラメータを一切変更しない（不変性保証）
- 入力8断面: 感情状態、ダイナミクス相、STM-感情連動結果、自己モデル感情記述、振幅状態、対話文脈、記憶参照、蓄積鮮度（全てREAD-ONLY）
- 7段パイプライン: 状態取得→推移パターン特徴抽出→持続パターン検出→変動候補列挙→候補整列・競合保持→蓄積→受渡準備
- 推移パターンはカテゴリラベルではなく数値特徴量として保持
- 変動候補はトリガーベースではなく常時列挙・等価候補（優劣なし）
- self_model.EmotionalStateViewとの関係: 現時点特徴 vs 時間軸推移特徴（補完関係）
- 討論で禁止された項目: 感情減衰パラメータ変更(5)、ムード直接介入(8)、STM連動介入(9)
- 将来の介入追加はemotion_amplitude.py経由のみ（代替案Bパターン）
- 安全弁4種: 変動候補収束防止、特徴量偏り防止、供給集中防止、蓄積偏り防止
- orchestrator Phase 14b（3ティック毎帯、Phase 14の後）、save/load v16 (40フィールド)、enrichment #23

#### ⑬ 自己行動知覚 ✅完了
- self_action_perception.py: 395行 / 114テスト
- 構造的欠落の補完: Geminiの応答テキストがpsycheにフィードバックされない欠落を解消
- 3段パイプライン: 受領保持→行動結果補完→参照情報受渡
- 受領保持: 応答テキスト+ポリシーラベル+ティック+タイムスタンプの対を時系列蓄積（上限押し出し）
- 行動結果補完: action_result_observationの入力に「実際の出力テキスト」(output_text)を追加
- 参照情報受渡: enrichmentセクション追加 + 内省系モジュールへのREAD-ONLY参照
- 全記録等価: 重み・スコア・優先度なし。テキスト非解釈（生テキスト保持）
- 判断系への非接続: ポリシー選択・バイアス計算・安定化弁への直接入力経路を禁止
- Phase 1-2不変性: 感情パイプラインのパラメータを変更しない
- brain.py: 6つのthinkメソッド全てにnotify_self_output()呼び出し追加（代弁コール完了後）
- orchestrator: notify_self_output()（set_recalled_memoriesパターン）、enrichment #24、save/load v17 (41フィールド)
- 討論結果: A-2推奨（工学的自我の根本要件、自己参照の閉合）

#### ⑮ 内部状態→行動経路の接続強化 ✅完了
- 新モジュールなし — expression.py と llm_wrapper.py のプロンプトテンプレート構造改善
- 討論結果: A-1条件付き推奨（既存Phase 30-35が既に構造的影響経路、新モジュールではなく接続強化として扱うべき）
- expression.py `_build_render_prompt()` を3層構造に改善:
  - 第1層「行動制約（確定済み — 変更禁止）」: ポリシーラベル/根拠/トーン/感情/気分/恐怖を構造的に配置
  - 第2層「内面的文脈（参照情報）」: enrichment 5セクションを明示的な囲みの中に配置
  - 第3層「状況」: 画面/会話/記憶を場面把握用に配置
- llm_wrapper.py `EXPRESSION_SYSTEM_PROMPT` に「入力の読み方」セクション追加:
  - 行動制約セクションの方針は確定済み・変更禁止の明示
  - 内面的文脈は機械的読み上げ禁止・全項目言及不要の明示
- セクション区切り線（═══）による視覚的分離
- 既存のフォールバック処理・パース処理・禁止パターンフィルタは不変

#### ⑱ 記憶の多経路想起 ✅完了
- multi_path_recall.py: 807行 / 105テスト
- 討論結果: C-3条件付き推奨（想起経路の多様化、6条件付き）
- 3経路の想起候補生成:
  1. 感情連想経路: 現在の感情状態と記憶の感情痕跡の近接度による候補列挙
  2. 文脈連想経路: 現在の知覚内容/トピックと記憶のトピック属性の重複による候補列挙
  3. 時間近接経路: 現在時刻と記憶のタイムスタンプの時間的距離による候補列挙
- 5種の安全弁:
  1. 経路間等価性（経路ごとの候補上限を同一に設定）
  2. 顕著性バイアス抑制（感情痕跡が弱い記憶の一定割合混入）
  3. ルーミネーション防止（直近想起履歴によるスライディングウィンドウ抑制）
  4. 忘却処理との分離（参照頻度非通知、INVISIBLE記憶除外）
  5. 外部API想起との整合（enrichment参照のみ、上書きなし）
- 3経路の構造的遮断: →忘却参照頻度、→感情パイプライン、→想起自己フィードバック
- orchestrator Phase 21d（5ティック毎、記憶系統統合後・enrichment前）
- enrichment #28: 記憶・内省セクションに経路ラベル付き候補を等価列挙
- save/load v21 (45フィールド): multi_path_recall_stateフィールド追加

#### ⑰ 時間認知構造 ✅完了
- temporal_cognition.py: 809行 / 212テスト
- 討論結果: C-1条件付き推奨（既存tick数ベース処理を一切変更しない純粋な参照情報供給として設計）
- Cycle 3候補1拡張: 帯域キャッシュ鮮度断面追加（4帯域の最終更新間隔を段階値で記述）
- Cycle 3候補5拡張: 入力経路間隔断面追加（3経路（text/screen/spontaneous）の最終使用からの経過を段階値で記述）
- 3段パイプライン:
  1. 経過記録蓄積: 毎ティック呼び出し時にtick番号・経過秒・タイムスタンプをスライディングウィンドウに蓄積（上限付きFIFO）
  2. 8断面特徴量記述（3ティック毎）:
     - 活動密度断面: ティック発生間隔の分布特性
     - 記憶蓄積間隔断面: エピソード記憶の蓄積の疎密
     - 感情変動頻度断面: 感情変動の頻度
     - 物語断片間隔断面: 自己物語断片の時間的間隔
     - 外部入力間隔断面: 外部入力の到着間隔
     - 総合経過断面: 累積経過秒と累積ティック数の比
     - 帯域キャッシュ鮮度断面: 4帯域（毎ティック/3ティック/5ティック/10ティック）の最終更新からの経過を段階値で記述
     - 入力経路間隔断面: 3入力経路（text/screen/spontaneous）の最終使用からの経過を段階値で記述
  3. 参照情報受渡準備: enrichmentテキスト（全断面等価列挙）+ READ-ONLYアクセサ
- 段階値は列挙型（DENSE / SOMEWHAT_DENSE / NORMAL / SOMEWHAT_SPARSE / SPARSE）+ 鮮度用（RECENT / SOMEWHAT_RECENT / MODERATE / SOMEWHAT_STALE / STALE）
- 5種の安全弁: 断面等価性、スライディングウィンドウ自然更新、パターン抽出禁止、単一数値統合禁止、enrichment強調禁止
- 4経路の構造的遮断: →ティックベース処理パラメータ、→感情パイプライン、→記憶忘却/固定化パラメータ、→予期形成
- temporal_self_differenceとの責務分離: 「自己像の差分」vs「時間経過の特徴」で対象が異なる
- orchestrator Phase 7b（毎ティック蓄積）/ Phase 14c（3ティック毎記述）/ notify_external_input（外部入力通知）
- enrichment #27: 自己認識セクションに時間的特徴量を等価列挙
- save/load v20 (44フィールド): temporal_cognition_stateフィールド追加

#### ⑯ 意図-行動間の乖離認知 ✅完了
- intent_action_gap.py: 397行 / 129テスト
- 討論結果: C-2推奨（⑬⑮完了により構造的に実装可能、自己参照の閉合精密化の次の一歩）
- 3段パイプライン:
  1. 対構成: 自己行動知覚記録とポリシー選択情報を対にする（対構成に失敗した場合はスキップ計数のみ）
  2. 多断面記述: ラベル断面（ポリシーラベル/出力テキスト冒頭）、テキスト断面（出力テキスト/ポリシー根拠の冒頭）、時間断面（tick番号）、文脈断面（文脈情報）の4断面で乖離を記述
  3. 蓄積・参照提供: リスト上限付きFIFO蓄積、enrichment #26で直近記録を等価列挙
- 7項目の禁止事項: 乖離度数値算出、パターン抽出、傾向化、統計処理、重み付与、自己矯正ループ、enrichment強調
- 5種の安全弁: 記録等価性、上限消滅、パターン抽出禁止、自己矯正ループ遮断、enrichment強調禁止
- 3経路の構造的遮断: →ポリシー選択、→行動-結果観測、→予期形成
- orchestrator Phase 26e: self_action_perceptionの最新記録とポリシー情報を入力として乖離記録を生成
- save/load v19 (43フィールド): intent_action_gap_stateフィールド追加

#### ⑭ 予期差分の参照経路拡張 ✅完了
- 新モジュールなし — orchestrator.py の既存 Phase 26d を拡張
- 既存の `_expectation_action_diff_log` が「蓄積されるのみでどこにも届かない」状態を解消
- Phase 26d差分記録の多断面化: 予期断面（内容記述/生成源/基盤/強度）、行動断面（ポリシーラベル/パターンキー）、結果断面（利用可能な結果断面キー一覧）、文脈断面（tick）
- get_expectation_diff_summary() アクセサ: 総件数・直近記録・断面キー一覧を読み取り専用で返す（変換・評価・選別なし）
- enrichment #25: 記憶・内省セクションに予期差分記録の等価列挙を追加（強調禁止）
- save/load v18 (42フィールド): diff_logリスト全体を1フィールドで永続化
- フィードバック経路の3重遮断: 差分記録→予期形成、差分記録→行動-結果観測、差分記録→ポリシー選択の接続を禁止
- 新奇性バイアス防止: 記録選別禁止、enrichment強調禁止、アクセサフィルタリング非搭載
- 討論結果: B-4条件付き推奨（既存diff_logの参照経路追加として）

#### ⑲ 内省断面間の横断的記述 ✅完了

- 設計書: design_introspection_cross_section.md
- 討論結果: 条件付き推奨（5条件: パターン抽出禁止/全断面等価/統合禁止/対象6モジュール絞り込み/ウィンドウ制限）
- 解析結果: 低固定化リスク（候補5件、いずれも量的制約パラメータ）
- 実装: introspection_cross_section.py (729行/130テスト)
- 3段パイプライン: 断面値収集→スナップショット構成・蓄積→参照情報受渡
- 初期対象6モジュール: self_model, temporal_self_difference, identity_coherence, self_narrative, introspection_consumption, meta_emotion_cognition
- 拡張候補（初期実装に含まず）: intent_action_gap, temporal_cognition, expectation_formation, intrinsic_motivation, continuity_strain, self_image_integration
- スライディングウィンドウによるFIFO蓄積、全スナップショット等価
- 安全弁5種: パターン抽出禁止/全断面等価/統合禁止/ウィンドウサイズ制限/判断系書き込み遮断
- 経路遮断5種: →各内省系モジュール入力/→ポリシー選択/→感情パイプライン/→記憶忘却固定化/→予期形成
- orchestrator Phase 14d（3ティック毎、6モジュールのキャッシュ出力を束ねて渡す）
- enrichment #29: 記憶・内省セクションに内省横断スナップショットの等価列挙（強調禁止）
- save/load v22 (47フィールド): スナップショットウィンドウ+直前スナップショット
- self_image_integrationとの責務分離: 「統合」vs「並置」の構造的区別を明記

#### ⑳ 知覚入力の内部文脈化 ✅完了

- 設計書: design_perceptual_context.md
- 討論結果: 条件付き推奨（6条件: テキスト比較禁止/topics意味判定禁止/段階値列挙型/⑱入力変更禁止/context_sensitivity自動接続禁止/型定義禁止）
- 解析結果: 低固定化リスク（候補4件、いずれも軽微）
- 実装: perceptual_context.py (646行/116テスト)
- 3段パイプライン: 知覚サマリ蓄積（毎ティック）→4断面特徴量記述（3ティック毎）→参照情報受渡
- Percept 4要素のみ参照: emotion（ラベル）, intent（ラベル）, topics（文字列リスト）, emotion_valence（数値）。text/meaningは非参照
- 4断面: 感情ラベル変化頻度/意図ラベル変化頻度/話題重複度（文字列完全一致のみ）/感情価推移方向
- temporal_cognitionと同じ段階値列挙型パターン。全断面等価
- 安全弁7種: テキスト比較禁止/topics意味判定禁止/断面等価/断面間統合禁止/パターン抽出禁止/型定義禁止/enrichment強調禁止
- 経路遮断4種: →感情パイプライン/→知覚解析/→multi_path_recall内部処理/→context_sensitivity連続性パラメータ
- orchestrator Phase 7c（毎ティック蓄積）/Phase 14e（3ティック毎記述）
- enrichment #30: 自己認識セクションに知覚推移の4断面等価列挙（強調禁止）
- save/load v22 (47フィールド): 知覚サマリウィンドウ+特徴量スナップショット+直前スナップショット

#### ㉑ 内省ウィンドウ拡大 ✅完了
- introspection_cross_section.pyのスライディングウィンドウサイズを10→25に拡大
- enrichment出力は直近10件のみに制限（enrichment肥大化防止）
- 残りの15件は内部参照用にのみ保持
- 反固定化第1段階（パラメータ変更のみ、新モジュールなし）
- 討論結果: discussion_anti_fixation_vs_self_formation_20260220.md（条件付き推奨）

#### ㉒ スコアリングの構造的揺らぎ ✅完了
- scoring_fluctuation.py: 647行
- 設計書: design_scoring_fluctuation.md
- 討論結果: 条件付き推奨（反固定化第2段階）
- 解析結果: 低固定化リスク（analysis_scoring_fluctuation_fixation_20260220.md）
- thought.pyの決定論的スコアリング（同一入力→同一出力の固定化）を緩和
- 5段パイプライン:
  1. 変動量抽出: 4入力源（感情多次元偏り/STM蓄積形状/駆動不均衡度/処理間隔）からスカラー変動度
  2. 変動度合成: 最大値と平均値の中間的統合（単一支配防止）
  3. 振幅制限: 上限 < ValueOrientation最大バイアス(+-5%)、下限 > 0（消失防止）
  4. ポリシー別揺らぎ値生成: ポリシー特性×変動成分の相互作用
  5. スコアへの加算: 全バイアス適用完了後の最後の加算層
- 永続化対象の内部状態なし（呼び出しごとに完結する純粋変換）
- 安全弁5種: 振幅絶対上限/状態蓄積禁止/入力源逆流遮断/ValueOrientation更新非介入/下限消失防止
- orchestrator Phase 35c（全バイアス適用後の最終加算層）

#### ㉓ 選択帰属 ✅完了
- selection_attribution.py: 413行 / 87テスト
- 設計書: design_selection_attribution.md
- 討論結果: 条件付き推奨（Agency第1段階）
- 解析結果: 低固定化リスク（analysis_selection_attribution_fixation_20260220.md）
- Cycle 3候補3拡張: バイアス源構成（名前一覧）を選択記録に併記。スコア・重み・方向性は記録しない（名前のみ）。enrichmentにバイアス情報は露出しない
- 方針選択の「そのとき」を記録する構造（既存の自己行動知覚は「後」、意図行動乖離は「間」を記録）
- 記録内容: 選択ポリシーラベル + 候補群ラベル一覧 + 候補数 + バイアス源名一覧 + ティック番号 + タイムスタンプ
- 候補のスコアは受領しない（全記録等価の原則維持）
- 蓄積リスト: 上限付きFIFO、最古押し出しが唯一の消失経路
- 出力2経路のみ: enrichment（直近記録の等価列挙、バイアス情報遮断）+ 内省系参照（READ-ONLY）
- 5経路遮断: →候補生成/→バイアス計算/→安定化弁/→感情処理/→責任計算
- 安全弁5種: 全記録等価/パターン抽出禁止/経路遮断不変性/enrichment等価列挙/上限入れ替わり保証
- orchestrator: select_policy_dict後にrecord_selection()、enrichment #31、save/load v23 (48フィールド)

#### ㉔ 参照頻度の構造的記述 ✅完了

- 設計書: design_reference_frequency_description.md
- 討論結果: 条件付き推奨（反固定化第3段階、discussion_anti_fixation_vs_self_formation_20260220.md）
- 設計解析結果: 低固定化リスク（analysis_reference_frequency_fixation_20260220.md）
- 実装解析結果: 低固定化リスク（analysis_reference_frequency_impl_fixation_20260220.md）
- Cycle 3候補9拡張: multi_path_recall/spontaneous_recallの経路別カウントを読み取り対象に追加（12→15箇所）
- 15箇所の既存モジュールのreference_countを横断的に読み取り専用で収集する集約層
- 収集元: episodic_memory/emotional_memory_binding(結合+痕跡)/introspection_consumption/expectation_formation/intrinsic_motivation(動機+衝動)/self_narrative/other_agent_model/self_reference/action_result_observation/other_model_dialogue_learning/memory_forgetting_fixation/multi_path_recall/spontaneous_recall
- 断面構成: 集中度（ジニ係数）+ 構造別偏在度（構造間ジニ係数）
- 断面履歴: FIFO（30件上限）、変動記述は毎回再導出（累積蓄積しない）
- enrichmentへの直接露出を遮断（安全弁5）
- 忘却パイプラインとの経路遮断
- 想起経路選択への影響遮断
- 安全弁5種: 全記録等価維持/評価的変換禁止/累積的傾向抑制/断面履歴有限性/出力経路不拡張
- orchestrator: Phase 24b（Phase 24 expectation_formation の後、Phase 25a の前）、save/load v24 (49フィールド)

#### ㉕ 持続的コミットメント ✅完了

- 設計書: design_persistent_commitment.md
- 討論結果: 条件付き推奨（Agency第2段階、discussion_persistent_commitment_20260221.md、7条件付き）
- 設計解析結果: 低固定化リスク（analysis_persistent_commitment_design_fixation_20260221.md）
- 実装解析結果: 中→修正後低（analysis_persistent_commitment_impl_fixation_20260221.md、3件修正済み）
- transient_goalの構造的限界（単一アクティブ/無抵抗切替/一様減衰/認知不在/軽量責任）を補完
- 唯一の生成経路: transient_goalからの昇格のみ（4経路遮断）
- 複数保持項目の並行保持（上限付き、資源競合で帯域分配）
- 強度依存の非線形減衰（飽和構造: 高強度域で加速減衰）
- 慣性値: 強度独立で時間減衰（自己強化ループ遮断）
- 4つの解除条件: 時間減衰/内部状態変動/競合保持項目出現/達成認知記録
- 認知記録: FIFO、READ-ONLY、評価判定なし、パターン抽出禁止
- バイアス出力: 三重上限制約（単一/方向別/総量）、max_total_bias=0.12 < VO 0.15
- 安全弁6種: 集中度監視/慣性累積上限/同一方向抑制(時間減衰付き)/最大保持期間/バイアス総量上限/緊急一括減衰
- 自己強化ループ4重遮断: 再昇格バイアス不加算/慣性独立減衰/同一方向バイアスキャップ/認知記録判断経路切断
- 修正済み問題: same_direction_counts時間減衰追加、enrichment累積カウンタ→スライディングウィンドウ化、ハードコードcategory_affinities除去
- orchestrator: Phase 12b（transient_goal後・scoped_goal前）、Phase 35b2（value_orientation後・scoring_fluctuation前）、enrichment #32、save/load v25 (50フィールド)

#### ㉖ 結合テスト強化 ✅完了

- orchestratorスモークテスト（test_orchestrator.py TestSmokeFullPipeline: 14テスト）
  - 全パス通過: 30tick→enrichment→suggestions→select_policy_dict→save→load→resume→save
  - select_policy_dict動作検証（返り値構造、少ティック、多様入力、save/load後）
  - Phase発火確認（3/5/10ティック後の各Phase属性設定）
  - enrichment5セクション存在・サイズ合理性検証
  - 連続稼働安定性（50ティック、途中save/load/resume）
- Phase間連鎖動作テスト（test_phase_chain_integration.py: 48テスト）
  - TestEveryTickPhaseChain: Phase 1/2/5/6/7/7a/7b/7c の毎ティック発火検証
  - TestEvery3TickPhaseChain: Phase 8-14e の3ティック発火・負のテスト
  - TestEvery5TickPhaseChain: Phase 15-26e の5ティック発火・連鎖順序検証
  - TestEvery10TickPhaseChain: Phase 27-28 の10ティック発火・負のテスト
  - TestPolicySelectionPhaseChain: Phase 30-35c の全Phase実行検証
  - TestCrossPhaseDataFlow: Phase間データフロー・save/load後再開

#### ㉘ 安定化の構造的記述 ✅完了

- 設計書: design_stabilization_description.md
- 討論結果: 条件付き推奨（反固定化第4段階、discussion_stabilization_description_20260222.md、7条件付き）
- 設計解析結果: 低固定化リスク（analysis_stabilization_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_stabilization_impl_fixation_20260222.md）
- 4断面提案→2断面に限定（信号源多寡 + 時間的変動幅）。「競合の有無」はstability_valveのdecision_fixationと重複、「方向の一致度」は断面間相関計算禁止原則に抵触のため除外
- 横断的読み取り層（reference_frequency_descriptionの設計思想に倣う）
- 断面1（信号源多寡）: 感情/STM/一時的目的/持続的取り組み/自発起動/外部入力の6信号源から非ゼロ/ゼロの二値のみ読み取りカウント。記憶系はreference_frequency_descriptionと重複するため除外
- 断面2（時間的変動幅）: temporal_self_differenceの差分サマリーをREAD-ONLY参照。新たな変動幅計算は行わない（パターン抽出禁止原則の維持）
- 3段パイプライン: 読み取り→断面構成→FIFO蓄積（上限30件、最古押し出し）
- enrichment直接露出遮断（get_enrichment_data()メソッドを持たない）
- 安全弁5種: 全記録等価/パターン抽出禁止/enrichment直接露出遮断/忘却経路遮断/出力経路不拡張
- orchestrator: Phase 15b（temporal_self_difference更新後・continuity_strain前）、save/load v26 (51フィールド)

#### ㉙ 行動多様性の構造的記述 ✅完了

- 設計書: design_behavioral_diversity_description.md
- 討論結果: 短期適応→短期認知への問題再定義後、候補7（多様度記述）の方向性を条件付き推奨（discussion_short_term_adaptation_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_behavioral_diversity_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_behavioral_diversity_impl_fixation_20260222.md）
- 行動結果観測構造と選択帰属構造の蓄積群から「種類数」のみを横断的にREAD-ONLY読み取りし、構造的多様性を記述する層
- 入力源2つ: action_result_observation（結果断面キー種類数）、selection_attribution（ポリシーラベル種類数、候補群サイズ分散度）
- 3段パイプライン: 読み取り→断面構成（3断面・段階値列挙型・全断面等価）→FIFO蓄積（上限30件）
- 3断面: 結果断面キー種類数（TypeCountLevel）、選択ラベル種類数（TypeCountLevel）、候補群サイズ分散度（DispersionLevel）
- 頻度情報の構造的排除: 「AがN回」ではなく「何種類あるか」の種類数のみ。出現回数は中間結果としても蓄積・出力しない
- enrichment直接露出遮断（get_enrichment_data()メソッドを持たない）
- 安全弁8種: 全記録等価/パターン抽出禁止/enrichment直接露出遮断/忘却経路遮断/想起経路遮断/頻度情報構造的不在/出力経路不拡張/既存モジュール安全弁維持保証
- orchestrator: Phase 26c2（action_result_observation後・予期差分記録前）、save/load v27 (52フィールド)

#### ㉚ 記憶の自発的想起（非参照型想起） ✅完了

- 設計書: design_spontaneous_recall.md
- 討論結果: 条件付き推奨（工学的自我との関連度最高、discussion_next_gaps_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_spontaneous_recall_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_spontaneous_recall_impl_fixation_20260222.md）
- 外部入力なしで内部状態変動のみを契機とする記憶浮上。multi_path_recall（外部トリガー型3経路）との経路分離を構造的に保証
- 入力8種（全READ-ONLY）: 感情状態、内的動機、方向ベクトル、連続性の揺らぎ、時間認知、記憶庫（統合単位）、感情痕跡、忘却段階情報
- 4段パイプライン: 内部状態断面抽出→3経路候補列挙→安全弁適用→出力整形
- 3経路（全等価）: 感情変動連想（前サイクルとの感情差分を起点）、動機連想（動機断片と記憶の重複度）、揺らぎ連想（連続性の揺らぎ閾値超えを契機）
- 安全弁7種: 経路間等価性/ルーミネーション防止（スライディングウィンドウ）/顕著性バイアス抑制/忘却処理分離/感情逆流遮断/判断系非接続/multi_path_recall経路分離
- 循環遮断4種: 感情→想起→感情、想起→動機→想起、想起→起動→想起、想起頻度→忘却速度→想起対象
- orchestrator: Phase 21e（multi_path_recall後）、enrichment #33（記憶・内省セクション）、save/load v28 (53フィールド)

#### ㉛ 内部状態の矛盾並置記述 ✅完了

- 設計書: design_internal_contradiction_description.md
- 討論結果: 条件付き推奨（内省断面間の不一致を解消せず記述、discussion_next_gaps_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_contradiction_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_contradiction_impl_fixation_20260222.md）
- 内省系モジュール群の出力間に生じる矛盾を検出・記述するが、解消・調停は行わない
- 5段パイプライン: 対構成→乖離検出→矛盾対記述→FIFO蓄積→参照受渡
- 6定義済み対（固定、動的選択なし）: 感情vs感情安定度、自己像安定度vs時間差分、同一性一貫度vs安定化信号数、自己像連続性vs揺らぎ度合い、感情vs感情的トーン、内省横断断面内部対
- 収束監視: 同一矛盾の連続出現を抑制（suppression_window）、ただし記録等価性は維持
- 評価的語彙サニタイズ: 「異常」「失敗」等の評価語を記述から排除
- 安全弁7種: 全記録等価/解消経路不在/意味判断禁止/パターン抽出禁止/enrichment直接露出遮断/忘却経路遮断/出力経路不拡張
- orchestrator: Phase 14f（3ティック周期、introspection_cross_section後）、enrichment #34（感情・トーンセクション）、save/load v29 (54フィールド)

#### ㉜ 相互作用の蓄積記述 ✅完了

- 設計書: design_interaction_accumulation.md
- 討論結果: 条件付き推奨（自他境界の構造化に貢献、discussion_next_gaps_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_interaction_accumulation_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_interaction_accumulation_impl_fixation_20260222.md）
- 自己行動知覚（表出テキスト）と他者モデルリアルフィード（他者反応）の時系列隣接対を構成・蓄積する層
- 入力2系統（全READ-ONLY）: self_action_perception記録、other_model_real_feed観測断片
- 4段パイプライン: 隣接対構成（時間的近接のみ・内容判定禁止）→対の記述（事実列挙）→FIFO蓄積（上限付き・選択的保持禁止）→参照受渡（enrichment等価列挙+READ-ONLYアクセサ）
- action_result_observationとの責務分離: action_result_observationは汎用「ポリシー→多断面結果」で判断系へ微弱シグナル供給、本機能は特定「表出テキスト→他者反応」で判断系への経路を一切持たない
- 安全弁5種: 全記録等価性/FIFO自然消失/ルーミネーション防止/パターン抽出構造的排除/判断系経路遮断
- orchestrator: Phase 25d（other_model_dialogue_learning後）、enrichment #35（相互作用セクション）、save/load v30 (55フィールド)

#### ㉝ 感情基調の持続認知 ✅完了

- 設計書: design_emotional_backdrop_cognition.md
- 討論結果: 条件付き推奨（dynamics-value_orientation間の時間スケール空白、discussion_next_gaps_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_emotional_backdrop_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_emotional_backdrop_impl_fixation_20260222.md）
- dynamics（数ターン単位）とvalue_orientation（数百ターン単位）の間の数十ターン単位の感情的構成を記述する層
- 入力8断面（全READ-ONLY）: 感情状態、ムード、ダイナミクス相、振幅、メタ感情認知推移特徴量、蓄積鮮度、対話経過、時間認知
- 4段パイプライン: 感情状態収集（スライディングウィンドウ）→窓内構成記述（等価列挙のみ・移動平均禁止・統合指標禁止）→蓄積処理（段階的鮮度減衰: active→weakening→fading→near_invisible→invisible）→受渡準備
- meta_emotion_cognitionとの責務分離: メタ感情認知は「推移特徴の認知と変動可能性の列挙」、本機能は「広い時間窓の感情状態の等価列挙」
- 安全弁5種: 低変動性監視/蓄積偏り検出/enrichment出力量制限/収束監視/自己像固定化遮断
- 経路遮断5種: 感情パイプラインパラメータ/ポリシー候補拡張直接供給/ムード直接値/記憶忘却パラメータ/ダイナミクス設定値
- orchestrator: Phase 14g（3ティック周期、internal_contradiction_description後）、enrichment #36（感情・トーンセクション）、save/load v31 (56フィールド)

#### ㉞ 状況依存的自己呈示の認知 ✅完了

- 設計書: design_situational_self_presentation.md
- 討論結果: 条件付き推奨（相手別の自己認知の構造的空白、discussion_next_gaps_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_situational_self_presentation_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_situational_self_presentation_impl_fixation_20260222.md）
- 自己行動知覚の記録を相手識別情報で分離蓄積し、相手別の出力構成を種類数段階値で記述する層
- 入力3系統（全READ-ONLY）: self_action_perception記録、相手識別情報（user_id）、selection_attribution記録
- 3段パイプライン: 相手別記録受領蓄積（相手別FIFO・鮮度減衰）→構成記述生成（種類数段階値・毎サイクル独立再計算・非累積）→参照受渡準備（enrichment等価列挙+READ-ONLYアクセサ）
- self_action_perceptionとの責務分離: 自己行動知覚は全出力を等価記録、本機能は相手別の事後的分離蓄積
- 安全弁8種: 記録等価性/パターン抽出禁止/FIFO自然消失/構成記述非累積性/マッピング形成禁止/enrichment露出制限/ポリシー選択経路遮断/収束監視
- orchestrator: Phase 7d（毎ティック、self_action_perception後）、enrichment #37（自己認知セクション）、save/load v32 (57フィールド)

#### ㉟ 内省の時間的縦断参照 ✅完了

- 設計書: design_introspection_longitudinal_view.md
- 討論結果: 条件付き推奨（横断的記述の補完として機能的に有用、discussion_next_gaps_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_introspection_longitudinal_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_introspection_longitudinal_impl_fixation_20260222.md）
- introspection_cross_sectionのスナップショットウィンドウ（25件）を唯一の入力源とし、「1時点・6断面」の横断的並置を「1断面・複数時点」の縦断的並置に視点変換する薄い変換層
- **独自の永続的内部状態を保持しない**（save/load不要 — 横断的記述のsave/loadに完全依存）
- 3段パイプライン: スナップショットウィンドウ取得→断面別時系列並置変換→参照受渡準備
- introspection_cross_sectionとの責務分離: 横断的記述はデータ蓄積・管理・スナップショット構成、本機能は視点変換のみ
- 安全弁5種: パターン抽出禁止/全断面等価/全時点等価/独自状態蓄積禁止/書き込み経路遮断
- orchestrator: Phase 14h（3ティック周期、introspection_cross_section後）、enrichment #38（記憶・内省セクション）

#### ㊱ 駆動の変動記述 ✅完了

- 設計書: design_drive_variation_description.md
- 討論結果: 推奨（感情3層に対しdrivesゼロの構造的非対称性の解消、discussion_next_gaps_cycle2_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_drive_variation_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_drive_variation_impl_fixation_20260222.md）
- DriveVector（curiosity/affiliation/autonomy/competence）の時間的推移をスライディングウィンドウで等価列挙する層
- 入力8断面（全READ-ONLY）: 駆動状態、感情基調認知、メタ感情認知、蓄積鮮度、対話経過、時間認知、ムード、反応更新
- 4段パイプライン: 駆動状態収集（スライディングウィンドウ）→窓内構成記述（等価列挙のみ・移動平均禁止・統合指標禁止）→蓄積処理（段階的鮮度減衰）→受渡準備
- emotional_backdrop_cognitionとの構造的対称性: 感情基調は感情パイプライン観測、本機能は駆動ベクトル観測
- 安全弁5種: 低変動性監視/蓄積偏り検出/enrichment出力量制限/収束監視/恒常性強調遮断
- 経路遮断6種: 駆動値/反応パラメータ/動機生成入力/ポリシー拡張供給/感情パイプライン/記憶忘却パラメータ
- orchestrator: Phase 14i（3ティック周期、emotional_backdrop_cognition後）、enrichment #39（動機・目標セクション）、save/load v33 (58フィールド)

#### ㊲ 予期の成立・消失の事後記述 ✅完了

- 設計書: design_expectation_lifecycle_description.md
- 討論結果: 推奨（予期のライフサイクル記述は構造的に明確な欠落、discussion_next_gaps_cycle2_20260222.md）
- 設計解析結果: 低固定化リスク（analysis_expectation_lifecycle_design_fixation_20260222.md）
- 実装解析結果: 低固定化リスク（analysis_expectation_lifecycle_impl_fixation_20260222.md）
- expectation_formationの予期のライフサイクル全体（生成→減衰→消失 or 的中 or 修正）の軌跡を事後的に記録する層
- スナップショット比較による5種類の状態遷移検出: 生成(generation)、消失(disappearance)、修正(revision)、強度変化(strength_change)、鮮度変化(freshness_change)
- FIFO蓄積（上限200件）、均一鮮度減衰（遷移種別による差別なし）
- 因果帰属禁止（予期が的中した「理由」を推測しない）、統計量算出禁止（的中率・精度等）
- 安全弁5種: 蓄積上限/均一減衰/収束監視/enrichment出力量制限/内容記述長制限
- orchestrator: Phase 26f（5ティック周期、intent_action_gap後）、enrichment #40（動機・目標セクション）、save/load v34 (59フィールド)

#### ㊳ 入力経路間の均衡記述 ✅完了

- 設計書: design_input_pathway_balance.md
- 討論結果: 条件付き推奨（3入力経路の使用実績の事後認知、discussion_next_gaps_cycle2_20260222.md）
- テキスト経路/画面知覚経路/自発起動経路の3入力経路がどの程度使用されているかの構造的記述
- 窓内カウント方式（単方向累積禁止）、段階値列挙型
- 「均衡すべき」という規範なし — 事実記述のみ
- 安全弁5種
- orchestrator: Phase 7e（毎ティック）、enrichment #41、save/load v35 (60フィールド)

#### ㊴ 責任の時間的推移記述 ✅完了

- 設計書: design_responsibility_temporal_trace.md
- 討論結果: 条件付き推奨（責任に特化した推移の記述構造が不在、discussion_next_gaps_cycle2_20260222.md）
- 責任重量・分布の時間的変遷をスナップショット蓄積し横断参照に供する層
- responsibility/responsibility_dispersionからREAD-ONLY参照
- 段階値記述、FIFO蓄積、責任分散操作への書き込み経路遮断
- 安全弁5種
- orchestrator: Phase 26g（5ティック周期、expectation_lifecycle後）、enrichment #42（責任セクション）、save/load v36 (61フィールド)

#### ㊵ 感情間の共起記述 ✅完了

- 設計書: design_emotion_cooccurrence_description.md
- 討論結果: 条件付き推奨（複数感情の同時存在事実の記録構造が不在、discussion_next_gaps_cycle2_20260222.md）
- 複数感情の同時存在（共起）の事実を記録する層。「どの感情同士が同時に高い値を持ったか」の種類のみを記録
- 出現頻度の記録禁止、評価的判定禁止（「共起は異常」等の判定なし）
- パターン抽出禁止、等価列挙、FIFO蓄積
- 安全弁5種
- orchestrator: Phase 14j（3ティック周期、drive_variation後）、enrichment #43（感情セクション）、save/load v37 (62フィールド)

#### ㊶ 他者境界の多相蓄積 ✅完了

- 設計書: design_other_boundary_accumulation.md
- 討論結果: 条件付き推奨（相手別の自他境界推移蓄積が不在、discussion_next_gaps_cycle2_20260222.md）
- other_agent_modelのSelfOtherBoundary（自他境界の乖離度）を相手別・時間軸で蓄積し変動を記述する層
- other_model_dialogue_learningの相手別分離パターンに倣う
- 境界の制御禁止（記述のみ）、パターン抽出禁止、等価列挙、FIFO、鮮度減衰
- 安全弁5種
- orchestrator: Phase 25e（5ティック周期、interaction_accumulation後）、enrichment #44（他者認知セクション）、save/load v38 (63フィールド)

#### ㊷ 忘却と想起の均衡記述 ✅完了

- 設計書: design_forgetting_recall_balance.md
- 討論結果: 条件付き推奨（忘却と想起の均衡の事後認知が不在、discussion_next_gaps_cycle2_20260222.md）
- memory_forgetting_fixation（忘却速度）とmulti_path_recall/spontaneous_recall（想起頻度）の稼働実績を窓内カウントで事実記述する層
- 窓内カウント方式（単方向累積禁止）、段階値列挙型
- 「均衡すべき」という規範なし — 事実記述のみ。忘却/想起パラメータへの書き込み経路遮断
- 安全弁5種: 蓄積上限/窓サイズ固定/収束監視/enrichment出力量制限/段階値更新制限
- orchestrator: Phase 21f（5ティック周期、memory_forgetting_fixation後）、enrichment #45（記憶セクション）、save/load v39 (64フィールド)

#### ㊸ 注意配分の構造的記述 ✅完了

- 設計書: design_attention_distribution_description.md
- 討論結果: 条件付き推奨（処理帯域の集中/分散の認知構造が不在、discussion_next_gaps_cycle2_20260222.md）
- orchestratorの各Phase処理時間・入力経路使用パターンから処理帯域の集中度/分散度を横断読み取りで記述する層
- 段階値列挙型、FIFO蓄積、帯域制御禁止（記述のみ）
- 安全弁5種: 蓄積上限/窓サイズ固定/収束監視/enrichment出力量制限/段階値更新制限
- orchestrator: Phase 7f（毎ティック、input_pathway_balance後）、enrichment #46（自己認知セクション）、save/load v40 (65フィールド)

#### ㊹ 目的階層間の隣接状態変化記述 ✅完了

- 設計書: design_goal_hierarchy_propagation.md
- 討論結果: 条件付き推奨（7層目的階層の伝搬が未認知、discussion_next_gaps_cycle3_20260223.md）
- 設計解析結果: 低固定化リスク（analysis_goal_hierarchy_propagation_design_fixation_20260223.md）
- transient_goal→persistent_commitment→value_orientationの3層に限定して、隣接層の状態変化の時間的同時性を事後的に記録する層
- 6段パイプライン: スナップショット取得→変化検出→隣接同時性記録構成→FIFO蓄積+鮮度減衰→収束監視→参照提供
- 段階値のみ使用（生の数値を含めない）、因果帰属禁止、パターン抽出禁止
- enrichment直接露出遮断（reference_frequency_descriptionと同パターン）
- 安全弁7種: 全記録等価/因果帰属排除/enrichment直接露出遮断/FIFO有限性/3層逆流経路不在/段階値限定/収束監視内部限定
- orchestrator: Phase 26h（5ティック周期、responsibility_temporal_trace後）、enrichmentなし（直接露出遮断）、save/load v41 (66フィールド)

#### ㊺ 仮説-観測の隣接対構成 ✅完了

- 設計書: design_hypothesis_observation_pairing.md
- 討論結果: 条件付き推奨（仮説と後続観測の対応付けが不在、discussion_next_gaps_cycle3_20260223.md）
- 設計解析結果: 低固定化リスク（analysis_hypothesis_observation_pairing_design_fixation_20260223.md）
- other_agent_modelの仮説と後続のother_model_real_feedの観測断片を時間的隣接のみで対構成し、相手別に分離蓄積する層
- 6段パイプライン: 仮説スナップショット取得→観測記述取得→隣接対構成→相手別分離蓄積→鮮度管理→受渡準備
- 仮説の正誤判定禁止、確認バイアスの構造的排除（内容的整合性を対構成基準に用いない）
- interaction_accumulationパターン流用（因果帰属禁止・FIFO・ルーミネーション防止）
- other_model_dialogue_learningとの単方向参照保証
- 安全弁7種: 等価性/確認バイアス排除/FIFO消失/ルーミネーション防止/パターン抽出排除/単方向参照保証/判断系経路遮断
- orchestrator: Phase 25f（5ティック周期、other_boundary_accumulation後）、enrichment #48（他者認知セクション）、save/load v42 (66フィールド)

### 9.3 Cycle 4: 既存構造の改善・統合（記述層追加なし）

Cycle 3討論で指摘された限界費用問題（記述層追加の漸減的寄与、enrichment肥大化、Phase数飽和）を踏まえ、新規psycheモジュール追加ゼロ、全項目が既存構造の改善・統合・品質強化。

| # | 項目 | 種別 | 依存 | 概要 |
|---|------|------|------|------|
| C4-8 | 結合テスト拡充 ✅完了 | 品質強化 | - | test_integration_extended.py (751行/40テスト) save/load resume・cold-start defaults・enrichment integrity・50-tick stability |
| C4-1 | thought.pyポリシー動的化 ✅完了 | 動的改善 | C4-8 | POLICIES 6→15、safety/autonomy軸追加、6断面スコアリング条件、動的選択3-5、安全弁。thought.py 293→473行、orchestrator extended_inputs配線、value_orientation 9ラベル追加 |
| C4-2+6 | enrichment圧縮+プロンプト効率化 ✅完了 | 接続面改善 | C4-1 | enrichment_compression.py (500行/62テスト) 3段パイプライン（二値変動検出→粒度選択→フォーマット圧縮）、orchestrator get_prompt_enrichment()を_collect_enrichment_items()+build_compressed_enrichment()に分離、安定項目「(安定)」短縮、安全弁5種。save/load非対象 |
| C4-3 | save/load一貫性検証 ✅完了 | 品質強化 | - | persistence_integrity.py (606→912行/156テスト) 43蓄積上限パターン追加(16→59)、Pattern 6型構造確認(8パターン)追加。persistence.py load()自動検証+環境変数制御 |
| C4-9 | self_action_perceptionフィードバック拡充 ✅完了 | 既存統合 | C4-1 | orchestrator Phase 22にself_action_perception→introspection_trace間接経路追加（メタ情報のみ: has_output/text_length/policy_label/tick）。28テスト |
| C4-7 | perception.py知覚強化 ✅完了 | 接続面改善 | - | perception.py (311行/259テスト) 感情辞書16→~100、意図辞書10→~70、全走査最大valence選出、topics補完/上限10。PERCEPTION_SYSTEM_PROMPT感情20種/意図17種に拡張 |
| C4-4 | orchestrator Phase宣言的定義 ✅完了 | 既存統合 | C4-1 | phase_declaration.py (1,435行/53テスト) 80 Phase定義レコード(frozen)×6帯域、データ依存グラフ導出、永続化/enrichmentマッピング。段階1: 宣言的定義+検証のみ、実行エンジン非導入 |
| C4-5+10 | 初回起動+セッション再開品質 ✅完了 | 実運用改善 | C4-3 | enrichment_compression.py拡張(500行/74テスト) 空状態「(未蓄積)」統一置換、セッション境界鮮度注釈（経過時間段階値+ティック数、50ティック無条件消失）。save/loadフィールド非追加 |

### 9.4 Cycle 5: 既存構造の深化・品質・実運用準備（記述層追加なし）

Cycle 4に続き新規psycheモジュール追加ゼロ。テスト基盤・保守性・外部ツール・対話品質の改善に集中。

| # | 項目 | 種別 | 概要 |
|---|------|------|------|
| C5-9 | テスト基盤拡充 ✅完了 | 品質強化 | test_extended_stability.py (51テスト) 200ティック安定性・境界条件17件・経路切替14件 |
| C5-8 | orchestratorメソッド分割 ✅完了 | 保守性改善 | orchestrator.py 5,079→5,231行。enrichment収集5分割・save/load各5分割・5ティック帯域5分割・候補生成3分割。コード再配置のみ、ロジック不変 |
| C5-1+7 | long_term_sim包括的拡張 ✅完了 | 実運用改善 | long_term_sim.py (419→792行/58テスト) 5新シナリオ・拡張レコード(enrichment文字数)・統計サマリー・差分レポート・CLI --stats/--compare |
| C5-3 | brain.py対話コンテキスト管理 ✅完了 | 接続面改善 | brain.py ContextEntry+DialogueContextManager・FIFO(100)/window(20)・時間間隔注釈3段階・ポリシーラベル構造的排除。expression.py後方互換。67テスト |
| C5-2 | Phase実行エンジン段階2 ✅完了 | 保守性改善 | psyche/phase_execution_engine.py(206行/44テスト) 10ティック帯域(Phase 27/28/29)限定の宣言的実行。ハンドラ登録・標準実行パターン・フォールバック。save/load非影響・enrichment非接続 |
| C5-4 | メインループ3経路制御改善 ✅完了 | 実運用改善 | loop_interval_controller.py(421行/77テスト) 画面キャプチャ適応間隔・自発起動結果連動間隔・テキスト-画面連携制御。main.py統合。psyche非接続 |
| C5-5 | save/load構造的圧縮 ✅完了 | 保守性改善 | psyche/persistence_helpers.py(563行/72テスト) マイグレーションチェーン系統化・共通保存復元ヘルパー・セマンティックグルーピング宣言。後方互換性維持 |
| C5-10 | 実運用ログ・モニタリング基盤 ✅完了 | 実運用改善 | tools/execution_monitor.py(932行/68テスト) Phase実行時間計測・enrichment圧縮比記録・API呼出記録・状態スナップショット。READ-ONLY観測のみ |

### 9.5 Cycle 6: 保守性・品質・実運用耐性（新psycheモジュールゼロ継続）

Cycle 4-5に続き記述層追加ゼロ。orchestrator分離・エンジン拡大・エラー耐性・結合テスト拡充に集中。

| # | 項目 | 種別 | 概要 |
|---|------|------|------|
| C6-3 | Phase帯域別結合テスト拡充 ✅完了 | 品質強化 | test_phase_band_chain_integration.py(1,239行) 5ティック帯域Phase 15-26連鎖検証・enrichment-永続化一貫性・save/load再開。テストのみ、psyche変更なし |
| C6-5 | 5ティック帯域物理的ファイル分離 ✅完了 | 保守性改善 | orchestrator_5tick_phases.py(1,093行) orchestrator.py 5,394→4,434行(960行削減)。ロジック不変・コード物理移動のみ |
| C6-6 | Gemini APIエラー耐性改善 ✅完了 | 実運用改善 | src/api_error_resilience.py(484行) 指数バックオフ再試行・レート制限対応・タイムアウト保護。brain.py(1,332行)・llm_wrapper.py拡張。test_api_error_resilience.py(1,195行)。psyche非依存 |
| C6-1 | Phase実行エンジン段階3(3ティック帯域拡大) ✅完了 | 保守性改善 | phase_execution_engine.py(206→359行) 複数帯域対応・帯域別有効/無効。orchestrator.py 3ティック帯域Phase 8-14ハンドラ登録。等価性テスト129件。10ティック帯域既存動作維持 |
| C6-7 | save/load復帰時ウォームアップ ✅完了 | 品質改善 | save_load_warmup.py(新規~230行) 32ウォームアップエントリ・3再導出種別(R/A/S)。orchestrator.py load()にexecute_warmup()呼び出し追加。53テスト |
| C6-2 | enrichment出力分布記述 ✅完了 | 観測改善 | tools/execution_monitor.py(+298行) EnrichmentDistributionMonitor。項目別出力特性・セクション別集計・FIFO履歴・重複検出。psyche非変更・永続化非対象。76テスト |
| C6-8 | ポリシー選択ログ構造化 ✅完了 | 観測改善 | tools/policy_selection_log.py(新規~800行) PolicySelectionLogger。thought.py collect_breakdown追加。psyche外部ツール・enrichment非接続。テスト付き |
| C6-9 | __init__.pyエクスポート構造整理 ✅完了 | 保守性改善 | psyche/__init__.py 21カテゴリセクション分け・型種別コメント・__all__再構成。エクスポート名変更なし（完全後方互換）。テスト付き |
| C6-10 | e2eスモークテスト ✅完了 | 品質改善 | tests/test_e2e_smoke.py 5層検証（接続/知覚/代弁/統合/テキスト入力）。APIキー未設定時skip。psyche変更なし |
| C6-4 | perception自己像注入 ❌スキップ | — | 討論で「要再検討」判定。自己像の知覚層への注入はpsycheの出力固定化リスクが高い |

### 9.6 State Dynamics Enrichment: 状態動態の拡充（Cycle 6完了後）

Cycle 6までの開発で観測・記述層が30+モジュールに達した一方、「状態を動かす層」（reaction/thought/policy選択）の構造は初期からほぼ変わっていなかった。この構造的偏り（「30台のカメラで1つの部屋を撮影」問題）を解消するため、5段階で状態動態を拡充。

| # | 段階 | 種別 | 概要 |
|---|------|------|------|
| SD-1 | expected_drive_changeの実適用 ✅完了 | 帰還経路 | orchestrator.py拡張のみ。ポリシー選択後にexpected_drive_changeをドライブに実適用。安全弁5種（軸別上限/範囲クランプ/1回限定/不変性/正帰還禁止）。自己抑制構造（全宣言値が負）。29テスト |
| SD-3 | ドライブ動態の状態依存化 ✅完了 | 動態改善 | reaction.py(201→547行)。固定加減算を5断面（感情-ドライブ連動/ドライブ間相互作用/目的階層/時間経過/覚醒-ドライブ）からの状態依存導出に置換。安全弁6種。純粋関数・蓄積なし。71テスト |
| SD-2 | 記憶想起→感情帰還 ✅完了 | 帰還経路 | memory_emotion_return.py(新規~430行)。4段パイプライン。multi_path_recall/spontaneous_recall両系統等価。ルーミネーション二重遮断（ティック境界+減衰）。enrichment非露出。安全弁7種。Phase 21g、save/load v43 (67フィールド)。59テスト |
| SD-4 | ムードの自律化 ✅完了 | 動態改善 | reaction.py内拡張。alpha=0.1 EMAを3段パイプライン（多入力源目標導出/追従速度導出/更新）に置換。6入力源（感情/ドライブ/目的/恐怖/責任/時間認知）。valence/arousal独立追従。安全弁6種。純粋関数・蓄積なし。66テスト |
| SD-5 | 選択結果→感情帰還 ✅完了 | 帰還経路 | orchestrator.py拡張。3段パイプライン。帯域をドライブ帰還の半分以下(0.075)に制限。正帰還ループ3重遮断（帯域制限+方向非固定+距離比例抑制）。純粋関数・蓄積なし。安全弁6種。36テスト |

**構造的変化の要約**:
- **帰還経路**: ポリシー選択→ドライブ(SD-1)、ポリシー選択→感情(SD-5)、記憶想起→感情(SD-2) の3経路を新設
- **動態改善**: ドライブ更新の状態依存化(SD-3)、ムードの自律化(SD-4) で内部状態の変化パターンを多様化
- **新モジュール**: memory_emotion_return.py (1件)
- **既存拡張**: reaction.py(201→547行)、orchestrator.py拡張
- **テスト追加**: 261テスト（29+71+59+66+36）
- **永続化**: save/load v43 (67フィールド)

### 9.7 バグ修正（HIGH優先度3件）

コードレビューで検出されたHIGH優先度バグ3件を修正。

| # | バグ | 影響 | 修正内容 |
|---|------|------|----------|
| H-1 | FearIndex永続化データ消失 | state.py to_dict()がcomposite値のみ保存、4柱個別リスク(identity/attachment/continuity/projection)が消失 | to_dict()を4柱+composite辞書形式に変更、from_dict()に3分岐復元（辞書/数値/None）+後方互換。12テスト |
| H-3 | alignment言語不一致 | transient_goal/scoped_goal/repeated_tendencyのcategory_policy_affinityキーが英語、thought.py POLICIESラベルが日本語で常にデフォルト値 | 3ファイルのaffinityキーを日本語に統一。14テスト |
| H-2 | レガシーAPI迂回 | api.py/simulation.pyがorchestratorの70+システムを迂回してpsyche関数を直接呼出 | 両ファイルをPsycheOrchestrator経由に書き換え。thinker.py/renderer.py(非参照)を削除。6テスト |

**削除ファイル**: src/thinker.py (167行)、src/renderer.py (160行) — H-2修正後に参照ゼロを確認し削除

### 9.8 思想整合性監査と修正

全コード（60+ファイル）をphilosophy.mdの7基準で監査し、6件の要注意事項を検出。討論の結果4件を修正。

**監査基準（7項目）:**
1. 自我・人格・価値の直接定義をしていないか
2. READ-ONLYの原則を守っているか
3. 安全弁が停止・禁止・矯正をしていないか
4. 正解/不正解/善悪評価を内部に持ち込んでいないか
5. identity.mdが「器の形」の範囲に留まっているか
6. enrichmentが介入ではなく伝達になっているか
7. ハードコード定数が収束を強制していないか

**修正4件:**

| # | 件名 | 判定 | 修正内容 |
|---|------|------|----------|
| 件2 | thought.py「からかう」強制差し替え | 修正推奨 | select_policy()行268-276の強制差し替えロジック削除。(1)-(3)のスコアペナルティ（帯域制限）のみで制御。規範的コメント修正 |
| 件1 | identity.md Section 3行動規範固定 | 条件付き修正 | 行動規範記述4箇所削除（「絶対にしない」「狂わず」「自信を持つ」「全肯定やめて」）。Section 3を「表現スタイルガイドライン」に改題。判断プロセス4ステップ削除、表現手法2項目のみ維持 |
| 件3 | api.py attachment常時正更新 | 条件付き修正 | positive=True固定→positive=(percept.emotion_valence >= 0.0)に動的化。src層管理データでpsyche内部attachmentには影響しないが、思想的整合性を確保 |
| 件6 | enrichment_compression.pyフッター | 条件付き修正 | 「反応」→「発話」に語彙修正（Geminiへの伝達指示の厳密化） |

**現状維持2件:**
- 件4: multi_emotion.py感情減衰率差異（love=0.01 vs surprise=0.15）— 人間の感情構造モデリングとして合理的
- 件5: api.py importance推定 — src層メモリ管理パラメータ、psyche内部に影響なし

**監査レポート**: audit_core_layer.md / audit_emotion_memory.md / audit_self_goal_other.md / audit_observation_infra.md
**討論記録**: discussion_audit_findings_20260228.md

### 9.9 実行時検証と3要注意事項の修正

長期シミュレーション（全11シナリオ・510ターン）による実行時検証で検出された3つの要注意事項を修正。

**検証結果**: 全70+システムがエラーなしで動作。12種類のポリシーが選択。save/load/resume正常。永続化整合性77パターン全通過。

**検出された3要注意事項と修正:**

| # | 問題 | 根本原因 | 修正内容 | 結果 |
|---|------|---------|----------|------|
| 1 | ポリシー集中（repeated_failure=100%共感） | thought.py _score_candidate()の4断面が同方向累積（スコア差4.56、揺らぎ0.12では逆転不能） | 断面別帯域制限±1.5（全断面均一）、intent="expression"等価寄与追加、非線形圧縮安全弁 | 部分解消: escalation_collapse 6ポリシー（max66%）、repeated_failure 92%→2ポリシー |
| 2 | fear_level=0.32固定 | 4柱リスクがシミュレーション中に不変（attachment bonds/memory_count更新経路なし） | long_term_sim.pyにon_memory_saved()追加（\|valence\|>0.3で発火）、psyche変更なし | 部分解消: 3/4シナリオで変動（0.11-0.32）、neutral_baselineは設計通り固定 |
| 3 | curiosityドライブ全シナリオ単調減衰 | 全15ポリシーのexpected_drive_changeでcuriosity全て負値、回復経路ゼロ | reaction.py: joy→curiosity連動追加、question充足量緩和(-0.08→-0.04)、expression微小回復。thought.py: curiosity消耗値緩和（全て負維持） | 部分解消: 単調減少解消（回復スパイクあり）、ただし低水準に留まる |

**修正方針の原則:**
- 安全弁「正の帰還禁止」（orchestrator.py）は変更なし
- 全ポリシーのcuriosity消耗値は負値維持（0.0にもしない）
- fear_levelはシミュレーション側のみで対処（psycheモジュール非変更）
- 帯域制限は全断面均一値（恣意的な断面別差異を禁止）

**テスト追加**: 17テスト（sim 7 + reaction 3 + thought 7）

**討論記録**: discussion_fix_observations_20260228.md
**分析レポート**: fix_observations_analysis.md
**設計書**: design_curiosity_recovery_fix.md / design_policy_concentration_fix.md / design_fear_level_fix.md
**検証結果**: verification_results.md

### 9.10 構造的欠落3件の補完

コードレビューで特定された3つの構造的欠落（知覚-内部状態間の結合不在、他者モデル-感情間の帰還経路不在、経験強度と価値更新帯域の非連動）を補完。

| # | 欠落 | 討論判定 | 実装内容 |
|---|------|---------|----------|
| 1 | 知覚フィルタリング | 条件付き推奨 | perception.py拡張: mood.valence→emotion_valenceへの微弱バイアス加算。帯域±0.04、純粋関数、状態なし。安全弁5種。35テスト |
| 2 | 他者仮説由来の感情帯域追加 | 条件付き推奨 | other_hypothesis_emotion_return.py新規: 4段パイプライン、キーワード辞書照合、帯域±0.02以下、FIFO事実記録。安全弁7種。Phase 21系統、save/load v44 (68フィールド)。65テスト |
| 3 | 経験強度による価値更新帯域拡大 | 要再検討→代替案採用 | orchestrator_5tick_phases.py拡張: 既存policy_dimension_map経由、経験強度（3断面乗算）による帯域拡大係数、冷却期間、confidence damping維持。安全弁7種。Phase 26-EXP。38テスト |

**討論記録**: discussion_structural_gaps_20260228.md
**設計書**: design_perceptual_bias.md / design_other_hypothesis_emotion_return.md / design_experience_driven_value_update.md
**テスト追加**: 138テスト（35+65+38）

### 9.11 Cycle 7 Tier 1: 検証基盤・構造分離・モニタリング

Cycle 7は既存構造の検証・最適化・保守に特化（新psycheモジュールゼロ）。Tier 1として推奨3候補を実装。

| # | 候補 | 討論判定 | 実装内容 |
|---|------|---------|----------|
| 5 | save/load永続化の包括的回帰テスト | 推奨 | test_save_load_regression.py (109テスト): 全68フィールドround-trip検証、マイグレーションチェーン互換性検証（v1→v44）、フィールド欠損時の非破壊性検証。psyche変更なし |
| 6 | orchestrator毎ティック帯域分離 | 推奨 | orchestrator_1tick_phases.py (447行): Phase 1-7実行コードの物理的分離。orchestrator.py (5,043→4,767行)。orchestrator_5tick_phases.pyと同一の委譲パターン。ロジック変更なし |
| 1 | 帰還経路の動作検証と相互干渉検出 | 推奨 | return_pathway_monitor.py (329行) + test (786行/59テスト): 3帰還経路(記憶→感情/選択→感情/他者仮説→感情)の発火記録・同一ティック合算記述・セッション累積。execution_monitor基盤内配置。enrichment非接続・永続化非対象 |

**追加の更新:**
- persistence_helpers.py: CURRENT_VERSION 43→44、マイグレーションエントリv43-v44追加
- orchestrator_5tick_phases.py: 帰還経路発火通知点追加
- テスト更新: test_integration_extended.py, test_orchestrator.py (version 44対応), test_init_cleanup.py (エクスポート数1311→1323)

**候補リスト**: candidates_cycle7.md (10候補)
**討論記録**: discussion_cycle7_20260301.md (3推奨/7条件付き推奨)
**設計書**: design_save_load_regression_test.md / design_1tick_band_separation.md / design_return_pathway_verification.md
**テスト追加**: 168テスト (109 + 59)

---

*このドキュメントはCyrene AI システムの完全な技術仕様書です。*
*総コード行数: ~208,000行 / テスト数: 8,883*
