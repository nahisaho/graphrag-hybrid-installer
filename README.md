# GraphRAG Hybrid Installer

[![GitHub](https://img.shields.io/badge/GitHub-nahisaho%2Fgraphrag--hybrid--insaller-blue?logo=github)](https://github.com/nahisaho/graphrag-hybrid-insaller)

**scispaCy + GiNZA + ドメイン辞書** によるハイブリッドGraphRAG環境を対話式で一括セットアップするインストーラーです。

## 特徴

- **ハイブリッドNLP抽出**: scispaCy（英語科学論文）+ GiNZA（日本語）+ ドメイン辞書（複合名詞補完）
- **NLPエッジ最適化パッチ**: GraphRAG v3.0.6 の `build_noun_graph.py` に対する Top-K + 共起フィルタパッチ（リレーション爆発問題の修正）
- **ストップワードレンマタイズ**: 形態変化に対応した停止語フィルタ（`investigation` → `investigated`, `investigating` 等も自動除外）
- **PERSON NER 強化**: `en_core_web_sm` による研究者名の補助的抽出。人名エンティティは Top-K フィルタをバイパス
- **PERSON NER 精度改善 (v0.3.0)**: ジャーナル略語・セクションヘッダ・結合エンティティの誤検出除去。低頻度人名フィルタ（`min_person_freq=3`）
- **対話式セットアップ**: LLM/Embedding/NLPモードを選択するだけで設定完了
- **複数プロバイダー対応**: OpenAI / Azure OpenAI / Ollama
- **LazyGraphRAG対応**: `fast` メソッドによる高速インデックス構築（LLM不要のNLPベース）
- **MCP Server対応**: Anthropic MCP プロトコルでClaude Desktop / VS Code Copilot から検索可能

## クイックスタート

```bash
# インストーラーを実行
bash install.sh

# または、インストール先を指定
bash install.sh --target-dir /path/to/my-project
```

対話式プロンプトで以下を設定します:

1. **プロジェクトディレクトリ** — ファイルの出力先
2. **入力ファイルの場所** — Markdown/テキストファイルのディレクトリ
3. **LLMプロバイダー** — OpenAI / Azure OpenAI / Ollama
4. **Embeddingプロバイダー** — Ollama bge-m3 / OpenAI / Azure
5. **NLPモード** — hybrid / scispacy / ginza

## インストール後の使い方

```bash
cd /path/to/my-project

# 1. 入力ファイルを配置
cp /path/to/your/files/*.md input/

# 2. ドメイン辞書を構築（オプション、推奨）
./build_dictionary.sh

# 3. インデックスを構築
./run_index.sh fast      # LazyGraphRAG（NLPベース、高速）
./run_index.sh standard  # 標準GraphRAG（LLMベース、高精度）

# 4. クエリを実行
./run_query.sh local "磁性材料の最新の研究動向は？"
./run_query.sh global "東北大学の主要な研究分野は？"

# 5. MCP Server を起動（Claude Desktop / VS Code 連携）
./run_mcp_server.sh stdio      # stdio モード（推奨）
./run_mcp_server.sh http 8765  # HTTP モード（リモート接続）
```

## MCP Server の使い方

### Claude Desktop で使う

1. `mcp_config.json` の内容を Claude Desktop の設定にコピー:

```bash
# macOS
cp mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
cp mcp_config.json ~/.config/claude/claude_desktop_config.json
```

2. Claude Desktop を再起動すると、GraphRAG のツールが利用可能になります。

### VS Code Copilot で使う

1. `.vscode/mcp.json` を作成:

```json
{
  "servers": {
    "graphrag": {
      "command": "python3",
      "args": ["/path/to/project/src/graphrag_mcp_server.py"],
      "env": {
        "GRAPHRAG_ROOT": "/path/to/project"
      }
    }
  }
}
```

2. Copilot Chat で `@graphrag` ツールが利用可能になります。

### 利用可能な MCP ツール

| ツール | 説明 | 用途 |
|--------|------|------|
| `graphrag_local_search` | エンティティ特化検索 | 特定の人物・材料・手法の詳細 |
| `graphrag_global_search` | テーマ横断検索 | 全体的な傾向・パターン分析 |
| `graphrag_drift_search` | ハイブリッド検索 | 詳細+概要の統合検索 |
| `graphrag_basic_search` | 基本テキスト検索 | クイック検索 |
| `graphrag_index_status` | インデックス状態確認 | エンティティ数等の確認 |

## NLPエッジ最適化パッチ

### 背景 — Lazy モードのリレーション爆発問題

GraphRAG v3.0.6 の Lazy（Fast）モードでは、`build_noun_graph.py` の `_extract_edges()` が `itertools.combinations()` で各チャンク内の **全エンティティペア** を生成します。科学論文など専門的なテキストでは1チャンクあたり約60の名詞句が抽出され、C(60,2) = **1,770ペア/チャンク** という O(N²) のリレーション爆発が発生します。

この結果、NLPベースの「高速」モードが LLMベースの Standard モードより **高コスト** になるという逆説的な状況が生じます。

| モード | エンティティ | リレーション | LLMコスト (gpt-4o-mini) |
|--------|-------------|-------------|------------------------|
| Standard (LLM) | 605 | 1,858 | $0.600 |
| Lazy (NLP) パッチ前 | 1,147 | **120,287** | **$0.929** |
| Lazy (NLP) パッチ後 | 1,147 | **2,660** | **$0.097** |

> **注**: この問題に対する修正は [microsoft/graphrag PR #2273](https://github.com/microsoft/graphrag/pull/2273) として提出済みです。

### パッチの3つの戦略

1. **Top-K エンティティ制限** (`max_entities_per_chunk=17`): チャンクごとに出現頻度上位K個のエンティティのみペアリング → C(17,2)=136 ペア/チャンク
2. **最小共起回数フィルタ** (`min_co_occurrence=2`): 1つのチャンクにしか共起しないエッジを除去（偶発的共起の排除）
3. **学術ストップワード除外**: `settings.yaml` の `exclude_nouns` に70+語の学術汎用語を追加（v0.3.0 で拡張）

### パッチの適用・復元

```bash
# v0.3.0 統合パッチ: Top-K + PERSON NER + 精度改善（インストール時に自動実行）
python3 src/patch_person_ner.py --max-k 17 --min-cooccurrence 2 --min-person-freq 3

# ストップワードレンマタイズパッチ（インストール時に自動実行）
python3 src/patch_stopword_lemma.py

# ドライラン（変更内容を確認のみ）
python3 src/patch_person_ner.py --dry-run
python3 src/patch_stopword_lemma.py --dry-run

# パッチを復元（オリジナルに戻す）
python3 src/patch_person_ner.py --restore
python3 src/patch_stopword_lemma.py --restore

# v0.1.0 互換: Top-K のみ（PERSON NER なし）
python3 src/patch_noun_graph.py --max-k 17 --min-cooccurrence 2
```

## ストップワードレンマタイズ（v0.2.0）

GraphRAG の元のストップワードフィルタは完全一致比較のため、`investigation` を除外しても
`investigated`, `investigating`, `investigations` は除外されません。

v0.2.0 のパッチは NLTK SnowballStemmer を使用してステムベースのマッチングを行います:

| exclude_nouns 登録語 | 自動除外される変化形 |
|---------------------|-------------------|
| `investigation` | investigated, investigating, investigations |
| `figure` | figures, figured, figuring |
| `result` | results, resulted, resulting |
| `study` | studies, studied, studying |

## PERSON NER 強化（v0.2.0 → v0.3.0）

Lazy モードでは scispaCy が科学テキスト向けに訓練されているため、人名の抽出精度が低い問題がありました。
v0.2.0 では `en_core_web_sm`（一般英語 NER モデル）による補助的な PERSON エンティティ抽出を追加しました。

v0.3.0 では誤検出フィルタを大幅に強化:

| 特徴 | 説明 |
|------|------|
| 抽出モデル | `en_core_web_sm` (spaCy 標準英語モデル) |
| 対象ラベル | PERSON のみ |
| Top-K バイパス | 人名エンティティは頻度に関係なくペアリング対象 |
| キャッシュ | `extract_person_entities` として個別にキャッシュ |
| フォールバック | `en_core_web_sm` 未インストール時は自動スキップ |
| 学術キーワードフィルタ | ジャーナル略語（J. APPL, J. MATER 等）を自動除外 |
| セクションヘッダ除外 | ローマ数字パターン（I. INTRODUCTION 等）を除外 |
| 結合エンティティ除外 | ET AL、>3単語、末尾句読点を除外 |
| 最小頻度閾値 | `min_person_freq=3`（デフォルト）で低頻度人名をスキップ |

### Top-K パラメータの選択ガイド

| K値 | リレーション数 | コスト | 品質 | 推奨用途 |
|-----|--------------|--------|------|---------|
| 10 | 1,060 | $0.072 | ★☆☆ | コスト最優先 |
| 15 | 2,660 | $0.097 | ★★★ | 品質・コスト良好 |
| **17** | **3,400** | **$0.120** | **★★★** | **推奨（研究者名も抽出可能）** |
| 20 | 5,108 | $0.149 | ★★★ | 品質重視 |
| 30 | 12,172 | $0.270 | ★★★ | 高密度グラフ |

> K=17 が品質・コストの最適解です。K=15 に比べ研究者名の特定能力が大幅に向上し、
> Standard モードと90%同等の検索品質を、1/5 のコストで実現します。

## NLPモードの比較

| モード | 使用モデル | 対象 | 特徴 |
|--------|-----------|------|------|
| `hybrid` | scispaCy + GiNZA + 辞書 | 多言語科学論文 | 最高精度（推奨） |
| `scispacy` | en_core_sci_lg | 英語科学論文 | 高カバレッジ |
| `ginza` | ja_ginza | 日本語テキスト | 日本語最適化 |

## LLM/Embeddingプロバイダーの比較

### LLM（Community Reports生成用）

| プロバイダー | モデル例 | コスト | 備考 |
|-------------|---------|--------|------|
| OpenAI | gpt-4o-mini | ~$0.15/1M tokens | 安定・高速 |
| Azure OpenAI | gpt-4o-mini | 同上 | エンタープライズ |
| Ollama | qwen2.5:7b | 無料 | ローカル実行 |

### Embedding

| プロバイダー | モデル | 次元数 | コスト | 日本語 |
|-------------|--------|--------|--------|--------|
| Ollama bge-m3 | bge-m3 | 1024 | 無料 | ◎ |
| OpenAI small | text-embedding-3-small | 1536 | $0.02/1M tokens | ○ |
| OpenAI large | text-embedding-3-large | 3072 | $0.13/1M tokens | ◎ |
| Azure OpenAI | text-embedding-3-large | 3072 | 同上 | ◎ |

## ファイル構成

```
project/
├── settings.yaml              # GraphRAG設定（自動生成）
├── .env                       # APIキー・エンドポイント
├── mcp_config.json            # MCP クライアント設定（自動生成）
├── domain_dictionary.json     # ドメイン辞書（build_dictionary.shで生成）
├── input/                     # 入力ファイル（Markdown/テキスト）
├── output/                    # インデックス出力（Parquet + LanceDB）
├── cache/                     # LLMキャッシュ
├── logs/                      # 実行ログ
├── prompts/                   # GraphRAGプロンプト
├── src/
│   ├── hybrid_extractor.py    # ハイブリッドNounPhraseExtractor
│   ├── run_graphrag_hybrid.py # CLI ラッパー（Monkey-Patch）
│   ├── graphrag_mcp_server.py # MCP Server（Anthropic MCP対応）
│   ├── build_domain_dictionary.py # ドメイン辞書構築
│   ├── patch_noun_graph.py    # NLPエッジ最適化パッチ v0.1.0（Top-K + 共起フィルタ）
│   ├── patch_person_ner.py    # PERSON NER + Top-K 統合パッチ v0.3.0
│   ├── patch_stopword_lemma.py # ストップワードレンマタイズパッチ v0.2.0+
│   └── generate_settings.py   # 設定ファイル生成（学術ストップワード含む）
├── build_dictionary.sh        # 辞書構築ショートカット
├── run_index.sh               # インデックス構築ショートカット
├── run_query.sh               # クエリ実行ショートカット
└── run_mcp_server.sh          # MCP Server起動ショートカット
```

## ドメイン辞書のカスタマイズ

### 基本的な構築

```bash
# デフォルト設定（en_core_sci_lg + ja_ginza）
./build_dictionary.sh

# カテゴリ分類CSVがある場合
python3 src/build_domain_dictionary.py \
  --input-dir input \
  --output domain_dictionary.json \
  --model en_core_sci_lg \
  --ja-model ja_ginza \
  --categories-csv papers_classification.csv
```

### 辞書フォーマット

```json
{
  "version": "2.0",
  "total_unique_terms": 1303,
  "categories": {
    "physics": {
      "term_count": 100,
      "terms": {
        "magnetic field": 47,
        "crystal structure": 35
      }
    }
  }
}
```

## 環境変数

| 変数 | 説明 | 必須 |
|------|------|------|
| `GRAPHRAG_API_KEY` | LLM用APIキー | Yes（Ollama以外） |
| `GRAPHRAG_EMBEDDING_KEY` | Embedding用APIキー | Yes（Ollama以外） |
| `OLLAMA_BASE_URL` | OllamaのURL | Ollama使用時 |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAIエンドポイント | Azure使用時 |
| `GRAPHRAG_DOMAIN_DICTIONARY` | 辞書ファイルパス | No（自動検出） |
| `GRAPHRAG_ROOT` | プロジェクトルート | No（自動検出） |
| `GRAPHRAG_SCI_MODEL` | scispaCyモデル名 | No (default: en_core_sci_lg) |
| `GRAPHRAG_JA_MODEL` | GiNZAモデル名 | No (default: ja_ginza) |

## トラブルシューティング

### en_core_sci_lg のインストールに失敗する

```bash
# 手動インストール
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

### Ollama bge-m3 が見つからない

```bash
ollama pull bge-m3
```

### WSLからWindows上のOllamaに接続する

```bash
# Windows側のIPアドレスを確認
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | head -1 | awk '{print $2}')
# install.sh で Ollama URL に http://$WIN_IP:11434/v1 を指定
```

## 参考資料

- [GraphRAG (Microsoft)](https://github.com/microsoft/graphrag)
- [NLPエッジ最適化パッチ PR #2273](https://github.com/microsoft/graphrag/pull/2273) — 本インストーラーの修正パッチを Microsoft GraphRAG に提出
- [GiNZA日本語チャンク分割最適化](https://qiita.com/hisaho/items/89a49e156b61609e5664)
- [ドメイン辞書統合によるLazy GraphRAG最適化](https://qiita.com/hisaho/items/d8a8ed7d2022b9e60dc5)
- [scispaCy](https://github.com/allenai/scispacy)

## ライセンス

MIT License
