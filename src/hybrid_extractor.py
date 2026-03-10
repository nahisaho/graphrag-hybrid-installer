"""
ハイブリッドNounPhraseExtractor for GraphRAG
==============================================
Qiita記事のアプローチに基づく実装:
  1. scispaCy (en_core_sci_lg) — ベース抽出（高カバレッジ・科学用語）
  2. GiNZA (ja_ginza) — 日本語テキスト対応
  3. ドメイン辞書 — 補完・ブースト（複合名詞・最新用語）

参考:
  - https://qiita.com/hisaho/items/89a49e156b61609e5664
  - https://qiita.com/hisaho/items/d8a8ed7d2022b9e60dc5
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

import spacy
from spacy.tokens.span import Span
from spacy.util import filter_spans

from graphrag.index.operations.build_noun_graph.np_extractors.base import (
    BaseNounPhraseExtractor,
)

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  エンティティ・ストップワード（v0.5.0）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDF→Markdown変換のアーティファクトおよびメタデータ由来のノイズを除去する。
# _post_filter_entities() で最終フィルタリングに使用。
ENTITY_STOPWORDS: set[str] = {
    # ドキュメントメタデータ / Markdownアーティファクト
    "CONTENT", "DOCUMENT", "FORMAT", "MARKDOWN FORMAT", "TEXT",
    "ORIGINAL", "EXTRACTED TEXT", "ADDITIONAL INFORMATION",
    "IMAGE",
    # 図表・構造要素
    "FIGURE", "FIGURES", "FIGURE FIGURE", "FIG", "FIG.",
    "TABLE", "TABLES", "NUMBER", "SECTION",
    # セクションヘッダー
    "REFERENCES", "INTRODUCTION", "EXPERIMENTAL", "DISCUSSION",
    "CONCLUSION", "CONCLUSIONS", "SUMMARY", "ABSTRACT",
    "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS",
    # 参考文献・学術誌略称（単語）
    "CROSSREF", "VOL", "VOLUME", "ISSN", "DOI",
    "REV", "LETT", "PHYS", "MATER", "SOC", "SCI", "APPL", "CHEM",
    "JPN", "RES", "NUCL", "TRANS", "PROC", "MAGN", "ENG", "TECH",
    "INT", "SUPPL", "MECH", "THERM", "VAC", "MED", "DENT",
    "ACTA", "PHILOS", "METALL", "ELECTROCHEM", "SURF", "CRYST",
    "BIOL", "OXID", "COHESION", "NON-CRYST",
    # 参考文献・学術誌略称（複合）
    "REV. LETT", "REV. B", "J. PHYS", "J. APPL", "J. CHEM",
    "J. MATER", "IEEE TRANS", "PHYS. REV", "J. SOC",
    "PHYS. REV. LETT", "PHYS. REV. B",
    # ページ・番号マーカー
    "P.", "PP", "NO.", "EQ.", "J.",
    # 出版社所在地
    "NEW YORK", "LONDON", "TOKYO",
    # 汎用英単語（POS除外を補完）
    "AND", "OF", "THE", "HERE", "WITH", "AT", "FOR", "FROM",
    "SUCH", "ALSO", "HOWEVER", "THEREFORE", "THUS",
    "RESULTS", "INVESTIGATED", "STUDY", "AUTHORS", "WORK",
    "CASE", "EFFECT", "DATA", "MEASURED", "METHODS",
    "SCIENCE", "NATURE", "JAPANESE",
    # 日本語汎用語
    "こと", "もの", "ため", "それ", "これ",
}

# 数値のみ / スペース区切り1文字パターンを検出する正規表現
_RE_NUMERIC_ONLY = re.compile(r"^[\d.\s%°,]+$")
_RE_SPACE_SINGLE_CHARS = re.compile(r"^([A-Z] )+[A-Z]$")  # "T O", "S N" etc.
# 学術誌パターン: "J. Xxx", "Rev. Xxx", "Phys. Rev." 等
_RE_JOURNAL_PATTERN = re.compile(
    r"^(J\.|REV\.|PHYS\.|PROC\.|TRANS\.|ANN\.|BULL\.)\s", re.IGNORECASE
)


def _has_japanese(text: str) -> bool:
    """テキストに日本語文字が含まれるか判定"""
    for ch in text:
        name = unicodedata.name(ch, "")
        if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return True
    return False


class HybridNounPhraseExtractor(BaseNounPhraseExtractor):
    """
    scispaCy + GiNZA + ドメイン辞書のハイブリッドExtractor

    処理フロー:
      入力テキスト
          │
          ├─ 日本語判定 ──→ ja_ginza で NER + noun_chunks
          │
          └─ 英語/混合 ──→ en_core_sci_lg で NER + noun_chunks
          │
          └─ ドメイン辞書マッチング（複合名詞の補完）
          │
          └─ 重複除去・フィルタリング → 結果
    """

    # サポートするNLPモード
    MODE_HYBRID = "hybrid"         # scispaCy + GiNZA
    MODE_SCISPACY = "scispacy"     # scispaCy only
    MODE_GINZA = "ginza"           # GiNZA only

    def __init__(
        self,
        sci_model_name: str = "en_core_sci_lg",
        ja_model_name: str = "ja_ginza",
        nlp_mode: str = "hybrid",
        dictionary_path: str | None = None,
        dictionary_boost_factor: float = 2.0,
        max_word_length: int = 20,
        include_named_entities: bool = True,
        exclude_entity_tags: list[str] | None = None,
        exclude_pos_tags: list[str] | None = None,
        exclude_nouns: list[str] | None = None,
        word_delimiter: str = " ",
        min_term_length: int = 2,
    ):
        model_label = {
            self.MODE_HYBRID: f"{sci_model_name}+{ja_model_name}",
            self.MODE_SCISPACY: sci_model_name,
            self.MODE_GINZA: ja_model_name,
        }.get(nlp_mode, f"{sci_model_name}+{ja_model_name}")

        super().__init__(
            model_name=model_label,
            max_word_length=max_word_length,
            exclude_nouns=exclude_nouns or [],
            word_delimiter=word_delimiter,
        )

        self.nlp_mode = nlp_mode
        self.sci_model_name = sci_model_name
        self.ja_model_name = ja_model_name
        self.include_named_entities = include_named_entities
        self.exclude_entity_tags = exclude_entity_tags or [
            "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
        ]
        self.exclude_pos_tags = exclude_pos_tags or [
            "DET", "PRON", "INTJ", "X", "ADP", "AUX", "CCONJ", "SCONJ", "PUNCT"
        ]
        self.dictionary_boost_factor = dictionary_boost_factor
        self.min_term_length = min_term_length

        self._nlp_sci = None
        self._nlp_ja = None

        if nlp_mode in (self.MODE_HYBRID, self.MODE_SCISPACY):
            logger.info("Loading scispaCy model: %s", sci_model_name)
            self._nlp_sci = self.load_spacy_model(sci_model_name, exclude=["lemmatizer"])

        if nlp_mode in (self.MODE_HYBRID, self.MODE_GINZA):
            logger.info("Loading GiNZA model: %s", ja_model_name)
            self._nlp_ja = self.load_spacy_model(ja_model_name, exclude=["lemmatizer"])

        self.dictionary: set[str] = set()
        self.dictionary_multiword: set[str] = set()
        self.synonym_to_canonical: dict[str, str] = {}  # synonym → canonical key
        if dictionary_path:
            self._load_dictionary(dictionary_path)

    def _load_dictionary(self, path: str):
        """ドメイン辞書をロード（v2.0 / v3.0 両対応）"""
        dict_path = Path(path)
        if not dict_path.exists():
            logger.warning("Domain dictionary not found: %s", path)
            return

        with open(dict_path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "2.0")

        if version.startswith("3"):
            # v3.0: 多言語シソーラス辞書
            for key, entry in data.get("terms", {}).items():
                canonical = entry.get("canonical", key.lower())
                # canonical形を辞書に追加
                if len(canonical) >= self.min_term_length:
                    self.dictionary.add(canonical)
                    if len(canonical.split()) >= 2:
                        self.dictionary_multiword.add(canonical)

                # 日本語訳を辞書に追加
                ja = entry.get("translations", {}).get("ja", "")
                if ja and len(ja) >= self.min_term_length:
                    self.dictionary.add(ja)
                    self.synonym_to_canonical[ja.upper()] = key

                # 全同義語を辞書に追加 + canonical へのマッピング
                for syn in entry.get("synonyms", []):
                    syn_lower = syn.lower().strip()
                    if len(syn_lower) >= self.min_term_length:
                        self.dictionary.add(syn_lower)
                        if len(syn_lower.split()) >= 2:
                            self.dictionary_multiword.add(syn_lower)
                        self.synonym_to_canonical[syn.upper()] = key

            # synonym_index も取り込み
            for syn, canonical_key in data.get("synonym_index", {}).items():
                self.synonym_to_canonical[syn.upper()] = canonical_key

            logger.info(
                "Loaded v3.0 bilingual thesaurus: %d terms, %d synonyms mapped",
                len(self.dictionary),
                len(self.synonym_to_canonical),
            )
        else:
            # v2.0: カテゴリ別辞書（従来形式）
            for category in data.get("categories", {}).values():
                for term in category.get("terms", {}):
                    term_lower = term.lower().strip()
                    if len(term_lower) >= self.min_term_length:
                        self.dictionary.add(term_lower)
                        if len(term_lower.split()) >= 2:
                            self.dictionary_multiword.add(term_lower)

            logger.info(
                "Loaded v2.0 domain dictionary: %d terms (%d multi-word)",
                len(self.dictionary),
                len(self.dictionary_multiword),
            )

    def extract(self, text: str) -> list[str]:
        """ハイブリッド抽出"""
        if not text or not text.strip():
            return []

        seen_terms: set[str] = set()
        results: list[str] = []

        has_ja = _has_japanese(text)

        # ---- Step 1: NLPモデルによる抽出 ----
        if has_ja and self._nlp_ja:
            ja_terms = self._extract_with_model(self._nlp_ja, text)
            for term in ja_terms:
                upper = term.upper()
                if upper not in seen_terms:
                    seen_terms.add(upper)
                    results.append(term)

            if self._nlp_sci:
                en_parts = re.findall(r'[A-Za-z][A-Za-z\s\-]{3,}[A-Za-z]', text)
                if en_parts:
                    en_text = " ".join(en_parts)
                    sci_terms = self._extract_with_model(self._nlp_sci, en_text)
                    for term in sci_terms:
                        upper = term.upper()
                        if upper not in seen_terms:
                            seen_terms.add(upper)
                            results.append(term)
        elif self._nlp_sci:
            sci_terms = self._extract_with_model(self._nlp_sci, text)
            for term in sci_terms:
                upper = term.upper()
                if upper not in seen_terms:
                    seen_terms.add(upper)
                    results.append(term)
        elif self._nlp_ja:
            ja_terms = self._extract_with_model(self._nlp_ja, text)
            for term in ja_terms:
                upper = term.upper()
                if upper not in seen_terms:
                    seen_terms.add(upper)
                    results.append(term)

        # ---- Step 2: ドメイン辞書マッチング ----
        if self.dictionary_multiword:
            dict_terms = self._dictionary_matching(text, seen_terms)
            results.extend(dict_terms)

        # ---- Step 3: 同義語を正規化形に統合 ----
        if self.synonym_to_canonical:
            results = self._normalize_synonyms(results)

        # ---- Step 4: エンティティ・ストップワードフィルタ ----
        results = self._post_filter_entities(results)

        return results

    def _normalize_synonyms(self, terms: list[str]) -> list[str]:
        """同義語を正規化形（canonical）に統合する

        例: "磁場" → "MAGNETIC FIELD", "磁界" → "MAGNETIC FIELD"
        これによりグラフ上で同一ノードに統合される。

        完全一致に加えて、抽出された用語の中に同義語が部分的に
        含まれている場合も正規化形を追加する。
        日本語はトークン間にスペースが入るため、スペース除去での比較も行う。
        """
        normalized = []
        seen = set()

        for term in terms:
            upper = term.upper()
            upper_nospace = upper.replace(" ", "")

            # 完全一致で正規化
            canonical_key = self.synonym_to_canonical.get(upper)
            if canonical_key:
                if canonical_key not in seen:
                    seen.add(canonical_key)
                    normalized.append(canonical_key)
                continue

            # 部分一致: 抽出用語の中に同義語が含まれているか
            # 日本語対応: スペース除去した形でも比較
            matched = False
            for syn, can_key in self.synonym_to_canonical.items():
                syn_nospace = syn.replace(" ", "")
                if (syn in upper
                        or syn_nospace in upper_nospace
                        or upper in syn
                        or upper_nospace in syn_nospace):
                    if can_key not in seen:
                        seen.add(can_key)
                        normalized.append(can_key)
                    matched = True
                    break

            # マッチしなかった場合はそのまま追加
            if not matched and upper not in seen:
                seen.add(upper)
                normalized.append(upper)

        return normalized

    def _post_filter_entities(self, terms: list[str]) -> list[str]:
        """エンティティレベルのノイズ除去フィルタ（v0.6.0）"""
        filtered = []
        for term in terms:
            upper = term.upper().strip()

            # ストップワード完全一致
            if upper in ENTITY_STOPWORDS:
                continue

            # 数値のみ（"10", "200", "1983", "0.08" 等）
            if _RE_NUMERIC_ONLY.match(upper):
                continue

            # スペース区切り1文字パターン（"T O", "S N", "M S" 等 = PDF断片）
            if _RE_SPACE_SINGLE_CHARS.match(upper):
                continue

            # 1文字以下（意味のない単一文字）
            stripped = upper.replace(" ", "")
            if len(stripped) <= 1:
                continue

            # "> FIGURE" のようなMarkdownアーティファクト
            if upper.startswith(">"):
                continue

            # 学術誌パターン: "J. Phys", "Rev. Lett", "Proc. Natl" 等
            if _RE_JOURNAL_PATTERN.match(upper):
                continue

            filtered.append(term)

        return filtered

    def _extract_with_model(self, nlp, text: str) -> list[str]:
        """spaCyモデルで名詞句を抽出"""
        try:
            doc = nlp(text[:1_000_000])
        except Exception:
            logger.warning("NLP processing failed, falling back to simple extraction")
            return self._fallback_extract(text)

        filtered = set()

        if self.include_named_entities:
            entities = [
                ent for ent in doc.ents
                if ent.label_ not in self.exclude_entity_tags
            ]
            spans = entities + list(doc.noun_chunks)
            spans = filter_spans(spans)

            missing = [
                ent for ent in entities
                if not any(ent.text in span.text for span in spans)
            ]
            spans.extend(missing)

            for span in spans:
                cleaned = self._clean_span(span)
                if cleaned:
                    filtered.add(cleaned)
        else:
            for chunk in doc.noun_chunks:
                cleaned = self._clean_span(chunk)
                if cleaned:
                    filtered.add(cleaned)

        return list(filtered)

    def _clean_span(self, span: Span) -> str | None:
        """スパンからノイズトークンを除去"""
        cleaned_tokens = [
            token for token in span
            if token.pos_ not in self.exclude_pos_tags
            and not self.is_excluded_noun(token.text)
            and not token.is_space
            and not token.is_punct
        ]

        if not cleaned_tokens:
            return None

        cleaned_texts = [t.text for t in cleaned_tokens]
        result = self.word_delimiter.join(cleaned_texts).replace("\n", "").strip()

        if len(result) < self.min_term_length:
            return None
        if all(len(w) > self.max_word_length for w in cleaned_texts):
            return None

        return result.upper()

    def _dictionary_matching(self, text: str, seen_terms: set[str]) -> list[str]:
        """ドメイン辞書からの追加マッチング"""
        text_lower = text.lower()
        new_terms = []

        for dict_term in self.dictionary_multiword:
            upper = dict_term.upper()
            if upper not in seen_terms and dict_term in text_lower:
                seen_terms.add(upper)
                new_terms.append(upper)

        return new_terms

    def _fallback_extract(self, text: str) -> list[str]:
        """NLP処理失敗時のフォールバック"""
        patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        return [p.upper() for p in patterns if len(p) >= self.min_term_length]

    def __str__(self) -> str:
        return (
            f"hybrid_{self.nlp_mode}_{self.sci_model_name}_{self.ja_model_name}_"
            f"dict{len(self.dictionary)}_{self.max_word_length}_"
            f"{self.include_named_entities}_{self.exclude_entity_tags}_"
            f"{self.exclude_pos_tags}_{self.exclude_nouns}_{self.word_delimiter}"
        )
