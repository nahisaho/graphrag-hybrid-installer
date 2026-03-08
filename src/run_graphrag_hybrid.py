"""
GraphRAG Hybrid CLI ラッパー
============================
NounPhraseExtractorFactory にハイブリッドExtractorをMonkey-Patchし、
GraphRAG CLI をそのまま実行する。

使用方法:
  python3 run_graphrag_hybrid.py index --method fast --skip-validation
  python3 run_graphrag_hybrid.py query -m local "your question"
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Monkey-patch the Factory ----
from graphrag.config.enums import NounPhraseExtractorType
from graphrag.config.models.extract_graph_nlp_config import TextAnalyzerConfig
from graphrag.index.operations.build_noun_graph.np_extractors.factory import (
    NounPhraseExtractorFactory,
)

# hybrid_extractor は同じ src/ ディレクトリにある
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from hybrid_extractor import HybridNounPhraseExtractor

_original_get_np_extractor = NounPhraseExtractorFactory.get_np_extractor


def _resolve_config_path(key: str, default: str) -> str | None:
    """環境変数 or デフォルトパスからファイルパスを解決"""
    path = os.environ.get(key, default)
    if path and os.path.exists(path):
        return path
    return None


@classmethod
def patched_get_np_extractor(cls, config: TextAnalyzerConfig):
    """
    model_name に 'hybrid', 'scispacy', 'ginza' を含む場合、
    HybridNounPhraseExtractor を返す。
    """
    model_name = (config.model_name or "").lower()

    if config.extractor_type == NounPhraseExtractorType.Syntactic and model_name in (
        "hybrid", "scispacy", "ginza"
    ):
        # 設定ファイルからパスを解決
        project_root = os.environ.get("GRAPHRAG_ROOT", os.getcwd())
        dict_path = _resolve_config_path(
            "GRAPHRAG_DOMAIN_DICTIONARY",
            os.path.join(project_root, "domain_dictionary.json"),
        )

        sci_model = os.environ.get("GRAPHRAG_SCI_MODEL", "en_core_sci_lg")
        ja_model = os.environ.get("GRAPHRAG_JA_MODEL", "ja_ginza")

        logger.info(
            "Creating HybridNounPhraseExtractor (mode=%s, dict=%s)",
            model_name,
            dict_path,
        )
        return HybridNounPhraseExtractor(
            sci_model_name=sci_model,
            ja_model_name=ja_model,
            nlp_mode=model_name,
            dictionary_path=dict_path,
            max_word_length=config.max_word_length,
            include_named_entities=config.include_named_entities,
            exclude_entity_tags=config.exclude_entity_tags,
            exclude_pos_tags=config.exclude_pos_tags,
            exclude_nouns=config.exclude_nouns,
            word_delimiter=config.word_delimiter,
        )
    return _original_get_np_extractor.__func__(cls, config)


NounPhraseExtractorFactory.get_np_extractor = patched_get_np_extractor
logger.info("Patched NounPhraseExtractorFactory with HybridNounPhraseExtractor")


if __name__ == "__main__":
    from graphrag.cli.main import app
    app()
