"""
GraphRAG MCP Server
====================
Anthropic MCP (Model Context Protocol) サーバーとして
Microsoft GraphRAG v3 の検索機能を公開する。

4つの検索モード:
  - local_search  : エンティティ特化の詳細検索
  - global_search : テーマ横断的な概要検索
  - drift_search  : ハイブリッド探索 (local + global)
  - basic_search  : テキストユニットのみの基本検索

使用方法:
  # stdio モード (Claude Desktop / VS Code 用)
  python3 graphrag_mcp_server.py

  # HTTP モード (リモート接続用)
  python3 graphrag_mcp_server.py --transport http --port 8765

環境変数:
  GRAPHRAG_ROOT  : GraphRAG プロジェクトルート (default: スクリプトの親ディレクトリ)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graphrag-mcp")

# ---------------------------------------------------------------------------
# Monkey-patch: HybridNounPhraseExtractor の登録
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from graphrag.config.enums import NounPhraseExtractorType
    from graphrag.config.models.extract_graph_nlp_config import TextAnalyzerConfig
    from graphrag.index.operations.build_noun_graph.np_extractors.factory import (
        NounPhraseExtractorFactory,
    )
    from hybrid_extractor import HybridNounPhraseExtractor

    _original_get_np_extractor = NounPhraseExtractorFactory.get_np_extractor

    def _resolve_config_path(key: str, default: str) -> str | None:
        path = os.environ.get(key, default)
        if path and os.path.exists(path):
            return path
        return None

    @classmethod
    def patched_get_np_extractor(cls, config: TextAnalyzerConfig):
        model_name = (config.model_name or "").lower()
        if config.extractor_type == NounPhraseExtractorType.Syntactic and model_name in (
            "hybrid", "scispacy", "ginza"
        ):
            project_root = os.environ.get("GRAPHRAG_ROOT", os.getcwd())
            dict_path = _resolve_config_path(
                "GRAPHRAG_DOMAIN_DICTIONARY",
                os.path.join(project_root, "domain_dictionary.json"),
            )
            sci_model = os.environ.get("GRAPHRAG_SCI_MODEL", "en_core_sci_lg")
            ja_model = os.environ.get("GRAPHRAG_JA_MODEL", "ja_ginza")
            logger.info("Creating HybridNounPhraseExtractor (mode=%s)", model_name)
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
except ImportError as e:
    logger.warning("HybridNounPhraseExtractor not available: %s", e)

# ---------------------------------------------------------------------------
# GraphRAG imports
# ---------------------------------------------------------------------------
from graphrag.cli.query import (
    _resolve_output_files,
    load_config,
    run_basic_search,
    run_drift_search,
    run_global_search,
    run_local_search,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "GraphRAG Hybrid MCP Server",
    instructions=(
        "Microsoft GraphRAG のナレッジグラフに対して検索を実行します。"
        "local_search はエンティティ特化、global_search はテーマ横断、"
        "drift_search はハイブリッド、basic_search はテキストユニットのみです。"
    ),
)


def _get_root_dir() -> Path:
    """GRAPHRAG_ROOT 環境変数、またはスクリプトの親ディレクトリを返す"""
    root = os.environ.get("GRAPHRAG_ROOT", os.path.dirname(SCRIPT_DIR))
    return Path(root)


def _load_graphrag_config() -> GraphRagConfig:
    return load_config(root_dir=_get_root_dir())


def _capture_search_output(search_fn, **kwargs):
    """GraphRAG の検索関数は結果を print するため、stdout をキャプチャ"""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = search_fn(**kwargs)
        captured = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    if result and isinstance(result, tuple) and len(result) == 2:
        response, context_data = result
        return response, context_data
    return captured, {}


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def graphrag_local_search(
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> str:
    """エンティティ特化のローカル検索。特定の人物・組織・材料・手法などの
    詳細情報を取得する場合に使用します。

    Args:
        query: 検索クエリ（日本語/英語）
        community_level: コミュニティ階層レベル (default: 2)
        response_type: 回答形式 (default: "Multiple Paragraphs")
    """
    root_dir = _get_root_dir()
    logger.info("Local search: %s", query)
    try:
        response, context = _capture_search_output(
            run_local_search,
            data_dir=None,
            root_dir=root_dir,
            community_level=community_level,
            response_type=response_type,
            streaming=False,
            query=query,
            verbose=False,
        )
        return str(response)
    except Exception as e:
        logger.error("Local search failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
def graphrag_global_search(
    query: str,
    community_level: int = 2,
    dynamic_community_selection: bool = True,
    response_type: str = "Multiple Paragraphs",
) -> str:
    """テーマ横断的なグローバル検索。全体的な傾向・パターン・
    主要テーマの分析に使用します。

    Args:
        query: 検索クエリ（日本語/英語）
        community_level: コミュニティ階層レベル (default: 2)
        dynamic_community_selection: 動的コミュニティ選択 (default: True)
        response_type: 回答形式 (default: "Multiple Paragraphs")
    """
    root_dir = _get_root_dir()
    logger.info("Global search: %s", query)
    try:
        response, context = _capture_search_output(
            run_global_search,
            data_dir=None,
            root_dir=root_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=False,
            query=query,
            verbose=False,
        )
        return str(response)
    except Exception as e:
        logger.error("Global search failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
def graphrag_drift_search(
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> str:
    """DRIFT検索（ハイブリッド探索）。ローカルとグローバルの両方の
    特性を組み合わせた高品質な検索に使用します。

    Args:
        query: 検索クエリ（日本語/英語）
        community_level: コミュニティ階層レベル (default: 2)
        response_type: 回答形式 (default: "Multiple Paragraphs")
    """
    root_dir = _get_root_dir()
    logger.info("DRIFT search: %s", query)
    try:
        response, context = _capture_search_output(
            run_drift_search,
            data_dir=None,
            root_dir=root_dir,
            community_level=community_level,
            response_type=response_type,
            streaming=False,
            query=query,
            verbose=False,
        )
        return str(response)
    except Exception as e:
        logger.error("DRIFT search failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
def graphrag_basic_search(
    query: str,
    response_type: str = "Multiple Paragraphs",
) -> str:
    """基本検索。テキストユニットのみを使用するシンプルな検索です。
    インデックスが不完全な場合やクイック検索に使用します。

    Args:
        query: 検索クエリ（日本語/英語）
        response_type: 回答形式 (default: "Multiple Paragraphs")
    """
    root_dir = _get_root_dir()
    logger.info("Basic search: %s", query)
    try:
        response, context = _capture_search_output(
            run_basic_search,
            data_dir=None,
            root_dir=root_dir,
            response_type=response_type,
            streaming=False,
            query=query,
            verbose=False,
        )
        return str(response)
    except Exception as e:
        logger.error("Basic search failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
def graphrag_index_status() -> str:
    """GraphRAG インデックスの状態を確認します。
    エンティティ数、リレーション数、コミュニティ数などを返します。
    """
    root_dir = _get_root_dir()
    logger.info("Checking index status at: %s", root_dir)
    try:
        config = _load_graphrag_config()
        dataframe_dict = _resolve_output_files(
            config=config,
            output_list=["entities", "relationships", "communities", "text_units"],
            optional_list=["community_reports", "covariates"],
        )

        status = {
            "root_dir": str(root_dir),
            "entities": len(dataframe_dict.get("entities", [])),
            "relationships": len(dataframe_dict.get("relationships", [])),
            "communities": len(dataframe_dict.get("communities", [])),
            "text_units": len(dataframe_dict.get("text_units", [])),
            "community_reports": (
                len(dataframe_dict["community_reports"])
                if dataframe_dict.get("community_reports") is not None
                else 0
            ),
            "covariates": (
                len(dataframe_dict["covariates"])
                if dataframe_dict.get("covariates") is not None
                else 0
            ),
        }
        return json.dumps(status, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Index status check failed: %s", e)
        return json.dumps({"error": str(e), "root_dir": str(root_dir)})


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------

@mcp.resource("graphrag://status")
def resource_status() -> str:
    """GraphRAG インデックスの統計情報"""
    return graphrag_index_status()


@mcp.resource("graphrag://entities")
def resource_entities() -> str:
    """抽出されたエンティティ一覧（上位100件）"""
    try:
        config = _load_graphrag_config()
        dataframe_dict = _resolve_output_files(
            config=config,
            output_list=["entities"],
        )
        entities_df = dataframe_dict["entities"]
        if "name" in entities_df.columns:
            top = entities_df.nlargest(100, "rank") if "rank" in entities_df.columns else entities_df.head(100)
            cols = [c for c in ["name", "type", "description", "rank"] if c in top.columns]
            return top[cols].to_json(orient="records", force_ascii=False, indent=2)
        return entities_df.head(100).to_json(orient="records", force_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GraphRAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port (only for http transport, default: 8765)",
    )
    args = parser.parse_args()

    root_dir = _get_root_dir()
    logger.info("GraphRAG MCP Server starting (root=%s, transport=%s)", root_dir, args.transport)

    if args.transport == "http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
