
# 概要
以下は「Gemini を声（発話レンダラ）としてのみ使う」ことを前提にした対話AIプロジェクトの実装指示です。あなた（Cloud Code）はこの仕様に従って、実行可能なリポジトリ（src/, psyche/, data/, tests/, Dockerfile, run.sh, README 等）を生成してください。外部 LLM 呼び出しは抽象化関数 llm_call(prompt, params) を使うこと。APIキーは環境変数で読み込む前提にする。

**最重要ルール**  
Geminiには「思考（判断・方針決定・感情更新・記憶管理）」をさせない。Geminiの役割は次の2つに限定する：
1. parse_percept（意味抽出・JSON形式での意図/sentiment/topics抽出）※補助的
2. render_expression（与えられた PsycheState / policy / memory_snippet / persona を忠実に自然文へ変換すること）

Geminiが出力に判断や解釈・状態変更を含めないよう、API呼び出し用のsystem promptテンプレートを必ず生成すること。

# 要件（出力物）
以下ファイル群を生成すること（各ファイルは docstring と型注釈、簡単な usage を含むこと）：
- src/
  - memory_manager.py
  - attachment_manager.py
  - identity_manager.py
  - projection_manager.py
  - emotion_model.py
  - state_manager.py
  - thinker.py
  - renderer.py
  - api.py (FastAPI; POST /respond 実装)
  - cli_tools.py (backup/compact/simulate_loss --test-mode 必須)
  - llm_wrapper.py (llm_call 抽象化; Gemini 用 system prompt テンプレート含む; streaming/timeout 設定)
- psyche/
  - __init__.py
  - state.py (PsycheState: Pydantic; decay, clamp_values, to_dict)
  - perception.py (parse_percept wrapper calling llm_wrapper)
  - reaction.py (react: update PsycheState)
  - memory_link.py (recall_by_mood wrapper)
  - thought.py (generate_thought_candidates, select_policy)
  - expression.py (render_expression wrapper calling llm_wrapper)
- data/
  - example_memories.json (10件サンプル)
  - example_attachments.json
  - identity.json
  - projections.json
  - state.json (初期)
  - persona.json
- tests/
  - test_memory.py
  - test_state_update.py
  - test_integration_flow.py
  - test_psyche_flow.py
- README.md（設計思想、起動方法、環境変数、運用注意、暗号化オプション）
- Dockerfile, run.sh, run_tests.sh
- .github/workflows/ci.yml (pytest, flake8/mypy, docker build)

# 動作仕様（重要）
1. ワンターン処理（POST /respond）:
   1. parse_percept (psyche/perception) — ユーザー発話を Percept に変換（ローカル解析＋必要なら llm_call 補助）。  
   2. react -> PsycheState 更新（psyche/reaction）。  
   3. recall_by_mood -> 関連記憶抽出（psyche/memory_link）。  
   4. generate_thought_candidates -> thinker で候補生成・選択（ローカルロジック; fear_index と drives を重視）。  
   5. render_expression -> renderer/psyche/expression で最終文生成（**Gemini を呼ぶが決して判断させない**）。  
   6. maybe_save memory, update attachments, identity as needed.  
   7. レスポンスは {text, meta, updated_state} を返す。

2. Geminiのsystem prompt と利用規約テンプレートを必ず生成すること。**system prompt の要点**:
   - あなたは発話レンダラであり、判断・解釈・記憶・感情更新を行ってはならない。
   - 入力は「確定済みのstate, policy, memory_snippet, persona」である。これを変更せず忠実に自然文を出力するのみ。
   - 出力は JSON で { "text": "...", "meta": {"emotion": "...", "intensity": float, "action": "..."} } の形で返すこと。
   - どんな場合でも「自分の判断でstateや方針を変えた」と見える表現や理由説明は出力しない。

3. テスト要件（必須）:
   - test_psyche_flow.py と test_integration_flow.py に必ず「少なくとも1ケース以上、fear_index が入力イベントにより変化する」assert を含めること。
   - test_memory.py では recall スコア計算と maybe_save の判定ロジックを検証すること。
   - test_state_update.py では state の読み書き・calc_fear_index の算出を検証すること。
   - run_tests.sh は pytest を実行し、CIファイルはこれをトリガーする。

4. llm_wrapper.py の実装方針:
   - llm_call(prompt, params) 抽象化を実装（内部では環境変数で API_KEY を読み込む）。
   - Gemini 向け system prompt テンプレートを含め、ストリーミング対応・timeout・retry を実装。
   - Gemini が余計なことを言った場合にローカルでフィルタ（禁止表現削除）する仕組みを用意する。

5. セキュリティ／運用:
   - memories.json の暗号化オプション（README 手順）を用意すること。
   - protected フラグ付きメモリはユーザー同意なしに削除できない。
   - simulate_loss は --test-mode を必須にして本番誤発動を防ぐ。
   - identity の重大変更は propose_identity_change で requires_confirmation フラグを返す。

# 出力形式（Cloud Codeに返してほしい内容）
1. 生成したリポジトリ構成のツリー表示。  
2. 主要ファイル（上位10個）の先頭〜200行を表示（もしくは要点抜粋）。  
3. tests の想定実行結果（実行はできない前提で想定出力を示す）。  
4. run_local.sh（ローカルでの起動手順）と run_tests.sh の内容。  
5. CI/workflow ファイル。  
6. 何を検証すれば「設計どおりGeminiが思考をしていない」と言えるかのチェックリスト（自動化可能な検証手順を含む）。

# 最後に（重要）
- もし実行可能ならこのリポジトリを zip にまとめてダウンロードリンクで返してほしい。実行できない場合は、手順・ファイルツリー・主要ファイル内容を全文で返してほしい。  
- 返答は「実装済み」のトーンで、作ったコードに対する短い説明と「どこがまだ要注意か」を必ず付けて返すこと。  
- 生成物にダミー実装（pass/TOOD/ランダム返し）を多用せず、主要ロジック（react, calc_fear_index, generate_thought_candidates, select_policy, render_expression）は実際に状態を変える実装を含めること。

以上。