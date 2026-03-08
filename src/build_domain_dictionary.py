"""
ドメイン辞書構築スクリプト
==========================
入力ファイル（Markdownテキスト）からspaCy NLPを使用して
ドメイン固有の用語を抽出し、GraphRAG互換のJSON辞書を生成する。

使用方法:
  python3 build_domain_dictionary.py --input-dir ./input --output domain_dictionary.json
  python3 build_domain_dictionary.py --input-dir ./input --output domain_dictionary.json --model en_core_sci_lg
  python3 build_domain_dictionary.py --input-dir ./input --output domain_dictionary.json --categories-csv papers_classification.csv
"""

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def has_japanese(text: str) -> bool:
    for ch in text[:200]:
        name = unicodedata.name(ch, "")
        if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return True
    return False


def load_spacy_model(model_name: str):
    """spaCyモデルをロード"""
    import spacy
    try:
        return spacy.load(model_name, exclude=["lemmatizer"])
    except OSError:
        logger.error(
            "Model '%s' not found. Install it first:\n"
            "  pip install %s", model_name, model_name
        )
        sys.exit(1)


def extract_terms_from_text(nlp, text: str, min_length: int = 3) -> list[str]:
    """テキストからnoun chunksとnamed entitiesを抽出"""
    doc = nlp(text[:500_000])
    terms = set()

    for chunk in doc.noun_chunks:
        cleaned = []
        for token in chunk:
            if token.pos_ in ("DET", "PRON", "PUNCT", "ADP", "AUX", "CCONJ", "SCONJ"):
                continue
            if token.is_space or token.is_punct:
                continue
            cleaned.append(token.text)
        if cleaned:
            term = " ".join(cleaned).strip()
            if len(term) >= min_length and len(term.split()) <= 5:
                terms.add(term.lower())

    for ent in doc.ents:
        if ent.label_ not in ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"):
            term = ent.text.strip()
            if len(term) >= min_length:
                terms.add(term.lower())

    return list(terms)


def load_categories(csv_path: str) -> dict[str, list[str]]:
    """分類CSVからカテゴリ→ファイルのマッピングを作成"""
    categories = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) >= 2:
                filename = parts[0].strip()
                category = parts[1].strip()
                categories[category].append(filename)
    return dict(categories)


def build_dictionary(
    input_dir: str,
    output_path: str,
    model_name: str = "en_core_web_sm",
    ja_model_name: str | None = None,
    categories_csv: str | None = None,
    max_files: int | None = None,
    min_frequency: int = 2,
    max_terms_per_category: int = 100,
):
    """ドメイン辞書を構築"""
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.md"))
    if not files:
        files = sorted(input_path.glob("*.txt"))
    if not files:
        logger.error("No .md or .txt files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d input files in %s", len(files), input_dir)

    # カテゴリ分類がある場合はそれを使用
    file_categories = {}
    if categories_csv and os.path.exists(categories_csv):
        cats = load_categories(categories_csv)
        for cat, cat_files in cats.items():
            for fname in cat_files:
                base = Path(fname).stem
                file_categories[base] = cat
        logger.info("Loaded %d categories from %s", len(cats), categories_csv)

    # NLPモデルのロード
    nlp_en = load_spacy_model(model_name)
    nlp_ja = None
    if ja_model_name:
        nlp_ja = load_spacy_model(ja_model_name)

    # ファイルごとに用語を抽出
    category_terms: dict[str, Counter] = defaultdict(Counter)
    processed = 0

    for fpath in files:
        if max_files and processed >= max_files:
            break

        try:
            text = fpath.read_text(encoding="utf-8")
        except Exception:
            continue

        if not text.strip():
            continue

        # カテゴリ判定
        stem = fpath.stem
        category = file_categories.get(stem, "general")

        # 言語判定とモデル選択
        if has_japanese(text) and nlp_ja:
            terms = extract_terms_from_text(nlp_ja, text)
        else:
            terms = extract_terms_from_text(nlp_en, text)

        for term in terms:
            category_terms[category][term] += 1

        processed += 1
        if processed % 100 == 0:
            logger.info("Processed %d / %d files", processed, len(files))

    logger.info("Processed %d files total", processed)

    # 辞書構築
    result = {
        "version": "2.0",
        "source": f"NLP extraction ({model_name})",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domains_covered": sorted(category_terms.keys()),
        "total_unique_terms": 0,
        "categories": {},
    }

    all_terms = set()
    for cat_name in sorted(category_terms.keys()):
        counter = category_terms[cat_name]

        # 頻度でフィルタリング
        filtered = {
            term: count for term, count in counter.items()
            if count >= min_frequency
            and len(term) >= 3
            and not re.match(r'^[\d\s\.\-\+]+$', term)
        }

        # 上位N件を選択
        top_terms = dict(
            sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:max_terms_per_category]
        )

        all_terms.update(top_terms.keys())

        result["categories"][cat_name] = {
            "name": cat_name,
            "term_count": len(top_terms),
            "terms": top_terms,
        }

    result["total_unique_terms"] = len(all_terms)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        "Dictionary saved: %s (%d unique terms, %d categories)",
        output_path, len(all_terms), len(result["categories"]),
    )

    # サマリ出力
    print("\n" + "=" * 60)
    print(f"📚 Domain Dictionary Summary")
    print("=" * 60)
    print(f"Total unique terms: {len(all_terms)}")
    print(f"Categories: {len(result['categories'])}")
    for cat_name, cat_data in result["categories"].items():
        top3 = list(cat_data["terms"].items())[:3]
        top3_str = ", ".join(f"{t}({c})" for t, c in top3)
        print(f"  {cat_name}: {cat_data['term_count']} terms — {top3_str}")
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build domain dictionary from input files for GraphRAG"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing input files (.md or .txt)"
    )
    parser.add_argument(
        "--output", default="domain_dictionary.json",
        help="Output JSON file path (default: domain_dictionary.json)"
    )
    parser.add_argument(
        "--model", default="en_core_web_sm",
        help="spaCy model for English (default: en_core_web_sm, recommended: en_core_sci_lg)"
    )
    parser.add_argument(
        "--ja-model", default=None,
        help="spaCy model for Japanese (e.g., ja_ginza). If set, auto-detects language."
    )
    parser.add_argument(
        "--categories-csv", default=None,
        help="CSV file with filename,category columns for categorized extraction"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Max number of files to process (for testing)"
    )
    parser.add_argument(
        "--min-frequency", type=int, default=2,
        help="Minimum term frequency to include (default: 2)"
    )
    parser.add_argument(
        "--max-terms", type=int, default=100,
        help="Maximum terms per category (default: 100)"
    )
    args = parser.parse_args()

    build_dictionary(
        input_dir=args.input_dir,
        output_path=args.output,
        model_name=args.model,
        ja_model_name=args.ja_model,
        categories_csv=args.categories_csv,
        max_files=args.max_files,
        min_frequency=args.min_frequency,
        max_terms_per_category=args.max_terms,
    )


if __name__ == "__main__":
    main()
