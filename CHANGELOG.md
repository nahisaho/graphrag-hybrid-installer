# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-08

### Added
- **ストップワードレンマタイズ** (`patch_stopword_lemma.py`)
  - NLTK SnowballStemmer によるステムベースの停止語マッチング
  - `investigation` → `investigated`, `investigating`, `investigations` を自動除外
  - `base.py` に `is_excluded_noun()` メソッドを追加
  - `syntactic_parsing_extractor.py` のフィルタを stem ベースに変更
- **PERSON NER 強化** (`patch_person_ner.py`)
  - `en_core_web_sm` による補助的な PERSON エンティティ抽出
  - 人名エンティティは Top-K フィルタをバイパス（`priority_entities`）
  - 遅延読み込み: `en_core_web_sm` 未インストール時は自動スキップ
  - `build_noun_graph.py` の完全置換パッチ（Top-K + PERSON NER 統合）

### Changed
- `install.sh`: `patch_person_ner.py` を優先適用、`patch_noun_graph.py` にフォールバック
- `install.sh`: `patch_stopword_lemma.py` を自動適用

## [0.1.0] - 2026-03-08

### Added
- インタラクティブインストーラー (`install.sh`)
  - OpenAI / Azure OpenAI / Ollama マルチプロバイダー対応
  - scispaCy + GiNZA + ドメイン辞書によるハイブリッド NLP 抽出
  - MCP Server 設定（Claude Desktop / VS Code Copilot 対応）
  - ドメイン辞書ビルダー (`build_domain_dictionary.py`)
- NLP エッジ最適化パッチ (`patch_noun_graph.py`)
  - Top-K エンティティ制限（K=17）
  - 共起フィルタ（min_co_occurrence=2）
  - 学術ストップワード 48 語
- settings.yaml テンプレート (`templates/settings_template.yaml`)
- ハイブリッドエクストラクター (`hybrid_extractor.py`)
- PR [#2273](https://github.com/microsoft/graphrag/pull/2273) を microsoft/graphrag に提出

### Changed
- Top-K デフォルト値を K=15 → K=17 に変更
  - 80 論文実験で K=17 が品質・コストの Pareto 最適解と実証
  - Standard モードの 90% 同等品質を 20% のコストで実現
  - 研究者名の抽出能力が大幅に向上（K=15: 0名 → K=17: 2〜5名）

### Performance
- 80 論文での NLP リレーション: 535,286 → 7,179（98.7% 削減）
- インデックス構築コスト: Standard 比 80% 削減（~$4.50 → ~$0.92）
- 構築時間: 22.7 分 → 13.8 分（39% 短縮）

## [0.0.1] - 2026-03-08

### Added
- 初期リリース（K=15 デフォルト）
- graphrag-hybrid-installer の基本構成 11 ファイル

[0.2.0]: https://github.com/nahisaho/graphrag-hybrid-installer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nahisaho/graphrag-hybrid-installer/compare/af232a0...v0.1.0
[0.0.1]: https://github.com/nahisaho/graphrag-hybrid-installer/releases/tag/af232a0
