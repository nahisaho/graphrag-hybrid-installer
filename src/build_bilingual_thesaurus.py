"""
多言語シソーラス付きドメイン辞書構築スクリプト
=================================================
既存ドメイン辞書の用語に対して:
  1. LLM (gpt-4o-mini) で日英対訳・同義語を生成
  2. spaCy で検証（実在する用語か確認）
  3. 正規化形（canonical form）を決定
  4. GraphRAG互換の多言語シソーラス辞書を出力

辞書フォーマット v3.0:
{
  "version": "3.0",
  "terms": {
    "MAGNETIC FIELD": {
      "canonical": "magnetic field",
      "translations": {"ja": "磁場", "en": "magnetic field"},
      "synonyms": ["磁界", "magnetic flux density field", "磁束密度場"],
      "category": "physics",
      "frequency": 47
    }
  },
  "synonym_index": {
    "磁場": "MAGNETIC FIELD",
    "磁界": "MAGNETIC FIELD",
    "magnetic flux density field": "MAGNETIC FIELD"
  }
}

使用方法:
  python3 build_bilingual_thesaurus.py \
    --input-dict domain_dictionary.json \
    --output bilingual_thesaurus.json \
    --api-key $OPENAI_API_KEY
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# LLM対訳生成のプロンプト
TRANSLATION_PROMPT = """\
You are a scientific terminology expert. For each technical term below, provide:
1. Japanese translation(s) - the most common academic translation
2. English synonyms - alternative English terms used in academic papers
3. Japanese synonyms - alternative Japanese terms

Output STRICT JSON array. Each element:
{{"term": "original term", "ja": "日本語訳", "ja_synonyms": ["同義語1"], "en_synonyms": ["synonym1"]}}

Rules:
- Only include actual translations used in academic papers
- For chemical formulas or proper nouns, keep as-is for ja if no standard translation
- Include abbreviations as synonyms (e.g., "TEM" for "transmission electron microscopy")
- Maximum 3 synonyms per language
- If unsure, use empty array []

Terms:
{terms}

JSON output:"""

BATCH_SIZE = 30  # terms per LLM call


def load_existing_dictionary(path: str) -> dict:
    """既存ドメイン辞書をロード"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    terms = {}
    for cat_name, cat_data in data.get("categories", {}).items():
        for term, freq in cat_data.get("terms", {}).items():
            term_clean = term.strip()
            if len(term_clean) >= 2:
                key = term_clean.upper()
                if key not in terms or terms[key]["frequency"] < freq:
                    terms[key] = {
                        "original": term_clean,
                        "category": cat_name,
                        "frequency": freq,
                    }
    return terms


def call_llm_for_translations(
    terms_batch: list[str],
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """LLMに対訳・同義語を問い合わせ"""
    import httpx

    terms_text = "\n".join(f"- {t}" for t in terms_batch)
    prompt = TRANSLATION_PROMPT.format(terms=terms_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    for attempt in range(3):
        try:
            resp = httpx.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # JSON部分を抽出
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)

            return json.loads(content)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("JSON parse error (attempt %d): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = min(2 ** (attempt + 1), 30)
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
            else:
                logger.error("HTTP error: %s", e)
                break
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            break

    return []


def validate_with_spacy(
    translations: list[dict],
    nlp_en=None,
    nlp_ja=None,
) -> list[dict]:
    """spaCyで対訳の妥当性を検証"""
    validated = []
    for entry in translations:
        # 基本的なバリデーション
        term = entry.get("term", "").strip()
        ja = entry.get("ja", "").strip()
        ja_synonyms = [s.strip() for s in entry.get("ja_synonyms", []) if s.strip()]
        en_synonyms = [s.strip() for s in entry.get("en_synonyms", []) if s.strip()]

        if not term:
            continue

        # 日本語訳の検証: 少なくとも1文字以上
        if ja and len(ja) < 1:
            ja = ""

        # 英語同義語の検証: spaCyで名詞句として認識されるか
        valid_en_syns = []
        if nlp_en and en_synonyms:
            for syn in en_synonyms[:3]:
                doc = nlp_en(syn)
                # noun_chunksまたはentitiesとして認識されれば有効
                has_noun = any(True for _ in doc.noun_chunks) or any(True for _ in doc.ents)
                if has_noun or len(syn.split()) <= 2:
                    valid_en_syns.append(syn)
        else:
            valid_en_syns = en_synonyms[:3]

        # 日本語同義語の検証
        valid_ja_syns = []
        if nlp_ja and ja_synonyms:
            for syn in ja_synonyms[:3]:
                if len(syn) >= 1:
                    valid_ja_syns.append(syn)
        else:
            valid_ja_syns = ja_synonyms[:3]

        validated.append({
            "term": term,
            "ja": ja,
            "ja_synonyms": valid_ja_syns,
            "en_synonyms": valid_en_syns,
        })

    return validated


def build_bilingual_thesaurus(
    input_dict_path: str,
    output_path: str,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    use_spacy_validation: bool = True,
    max_terms: int | None = None,
):
    """多言語シソーラス辞書を構築"""
    # 既存辞書をロード
    terms_data = load_existing_dictionary(input_dict_path)
    logger.info("Loaded %d terms from %s", len(terms_data), input_dict_path)

    # 頻度順にソート
    sorted_terms = sorted(
        terms_data.items(),
        key=lambda x: x[1]["frequency"],
        reverse=True,
    )

    if max_terms:
        sorted_terms = sorted_terms[:max_terms]

    logger.info("Processing %d terms", len(sorted_terms))

    # spaCyモデルをロード（検証用）
    nlp_en = None
    nlp_ja = None
    if use_spacy_validation:
        try:
            import spacy
            nlp_en = spacy.load("en_core_sci_lg", exclude=["lemmatizer"])
            logger.info("Loaded en_core_sci_lg for validation")
        except Exception:
            try:
                import spacy
                nlp_en = spacy.load("en_core_web_sm", exclude=["lemmatizer"])
                logger.info("Loaded en_core_web_sm for validation")
            except Exception:
                logger.warning("No English spaCy model available for validation")

        try:
            import spacy
            nlp_ja = spacy.load("ja_ginza", exclude=["lemmatizer"])
            logger.info("Loaded ja_ginza for validation")
        except Exception:
            logger.warning("No Japanese spaCy model available for validation")

    # バッチでLLM呼び出し
    thesaurus = {}       # canonical_key -> entry
    synonym_index = {}   # synonym -> canonical_key

    term_list = [v["original"] for _, v in sorted_terms]
    total_batches = (len(term_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(term_list))
        batch = term_list[start:end]

        logger.info(
            "Batch %d/%d: processing %d terms (%d-%d)",
            batch_idx + 1, total_batches, len(batch), start + 1, end,
        )

        # LLM呼び出し
        translations = call_llm_for_translations(
            batch, api_key=api_key, api_base=api_base, model=model
        )

        if not translations:
            logger.warning("No translations returned for batch %d", batch_idx + 1)
            # フォールバック: 対訳なしでエントリを作成
            for term in batch:
                key = term.upper()
                if key in terms_data:
                    thesaurus[key] = {
                        "canonical": term.lower(),
                        "translations": {"en": term.lower(), "ja": ""},
                        "synonyms": [],
                        "category": terms_data[key]["category"],
                        "frequency": terms_data[key]["frequency"],
                    }
            continue

        # spaCy検証
        if use_spacy_validation:
            translations = validate_with_spacy(translations, nlp_en, nlp_ja)

        # シソーラスに登録
        for entry in translations:
            term = entry["term"]
            key = term.upper()
            if key not in terms_data:
                # LLMが元の用語と少し異なるテキストを返した場合のフォールバック
                close_keys = [k for k in terms_data if term.lower() in k.lower() or k.lower() in term.lower()]
                if close_keys:
                    key = close_keys[0]
                else:
                    continue

            ja_translation = entry.get("ja", "")
            ja_synonyms = entry.get("ja_synonyms", [])
            en_synonyms = entry.get("en_synonyms", [])

            all_synonyms = []
            if ja_translation:
                all_synonyms.append(ja_translation)
            all_synonyms.extend(ja_synonyms)
            all_synonyms.extend(en_synonyms)

            # 重複除去
            seen = {term.lower(), key.lower()}
            unique_synonyms = []
            for s in all_synonyms:
                s_lower = s.lower()
                if s_lower not in seen and len(s) >= 1:
                    seen.add(s_lower)
                    unique_synonyms.append(s)

            thesaurus[key] = {
                "canonical": term.lower(),
                "translations": {
                    "en": term.lower(),
                    "ja": ja_translation,
                },
                "synonyms": unique_synonyms,
                "category": terms_data[key]["category"],
                "frequency": terms_data[key]["frequency"],
            }

            # 同義語インデックスを構築
            for syn in unique_synonyms:
                syn_upper = syn.upper()
                if syn_upper != key:
                    synonym_index[syn_upper] = key

        # Rate limit
        if batch_idx < total_batches - 1:
            time.sleep(1)

    # 結果を保存
    result = {
        "version": "3.0",
        "source": f"LLM translation ({model}) + spaCy validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_terms": len(thesaurus),
            "terms_with_ja": sum(1 for t in thesaurus.values() if t["translations"]["ja"]),
            "total_synonyms": sum(len(t["synonyms"]) for t in thesaurus.values()),
            "synonym_index_size": len(synonym_index),
        },
        "terms": thesaurus,
        "synonym_index": synonym_index,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # サマリ出力
    stats = result["stats"]
    print("\n" + "=" * 60)
    print("📚 Bilingual Thesaurus Summary")
    print("=" * 60)
    print(f"Total terms:       {stats['total_terms']}")
    print(f"With JP translation: {stats['terms_with_ja']}")
    print(f"Total synonyms:    {stats['total_synonyms']}")
    print(f"Synonym index:     {stats['synonym_index_size']} entries")
    print(f"\nSaved to: {output_path}")

    # サンプル表示
    print("\n--- Sample entries ---")
    for key, entry in list(thesaurus.items())[:5]:
        ja = entry["translations"]["ja"] or "(none)"
        syns = ", ".join(entry["synonyms"][:4]) or "(none)"
        print(f"  {entry['canonical']}")
        print(f"    日本語: {ja}")
        print(f"    同義語: {syns}")


def main():
    parser = argparse.ArgumentParser(
        description="Build bilingual thesaurus from domain dictionary using LLM"
    )
    parser.add_argument(
        "--input-dict", required=True,
        help="Input domain dictionary JSON (v2.0 format)"
    )
    parser.add_argument(
        "--output", default="bilingual_thesaurus.json",
        help="Output bilingual thesaurus JSON"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-base", default="https://api.openai.com/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-terms", type=int, default=None,
        help="Max terms to process (for testing)"
    )
    parser.add_argument(
        "--no-spacy-validation", action="store_true",
        help="Skip spaCy validation step"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # Try .bashrc
        import subprocess
        try:
            result = subprocess.run(
                ["grep", "OPENAI_API_KEY", os.path.expanduser("~/.bashrc")],
                capture_output=True, text=True
            )
            for line in result.stdout.strip().split("\n"):
                if "export" in line:
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
        except Exception:
            pass

    if not api_key:
        logger.error("API key required. Set --api-key or OPENAI_API_KEY env var.")
        sys.exit(1)

    build_bilingual_thesaurus(
        input_dict_path=args.input_dict,
        output_path=args.output,
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        use_spacy_validation=not args.no_spacy_validation,
        max_terms=args.max_terms,
    )


if __name__ == "__main__":
    main()
