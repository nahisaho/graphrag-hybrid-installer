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
# 2-Layer Search imports
# ---------------------------------------------------------------------------
try:
    from two_layer_search import (
        IndexCache,
        create_provider,
        search_relevant_papers,
        run_two_layer_search,
        PROVIDERS,
        DEFAULT_CACHE_DIR,
        DEFAULT_LANCEDB_DIR,
        DEFAULT_PAPERS_DIR,
        DEFAULT_PROMPTS_DIR,
        DEFAULT_ENV_FILE,
        DEFAULT_WORKSPACE_BASE,
        PROJECT_ROOT,
    )
    TWO_LAYER_AVAILABLE = True
    logger.info("2-layer search module loaded")
except ImportError as e:
    TWO_LAYER_AVAILABLE = False
    logger.warning("2-layer search not available: %s", e)

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
        "two_layer_query は大規模論文コレクション（数千〜数万本）から"
        "クエリ駆動で関連論文を抽出し、オンデマンドで GraphRAG を構築して回答します。"
    ),
)

# マルチユーザー並行制御: 2層検索の同時ビルド数を制限
_two_layer_semaphore = asyncio.Semaphore(
    int(os.environ.get("GRAPHRAG_MAX_CONCURRENT_BUILDS", "3"))
)
_cache_lock = asyncio.Lock()


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
# 2-Layer Search MCP Tools (マルチユーザー対応)
# ---------------------------------------------------------------------------

def _get_two_layer_provider_name() -> str:
    """環境変数からデフォルトの embedding プロバイダーを取得"""
    return os.environ.get("GRAPHRAG_EMBEDDING_PROVIDER", "openai")


def _get_lancedb_dir(provider_name: str) -> Path:
    """プロバイダーに応じた LanceDB ディレクトリを返す"""
    env_dir = os.environ.get("GRAPHRAG_LANCEDB_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(PROVIDERS.get(provider_name, {}).get(
        "lancedb_default", str(DEFAULT_LANCEDB_DIR)
    )) if TWO_LAYER_AVAILABLE else DEFAULT_LANCEDB_DIR


@mcp.tool()
async def two_layer_query(
    query: str,
    top_k: int = 10,
    search_type: str = "local",
    provider: str = "",
) -> str:
    """大規模論文コレクションからクエリ駆動で関連論文を抽出し、
    オンデマンドで GraphRAG インデックスを構築して回答します。
    数千〜数万本の論文に対応する2層アーキテクチャ検索です。

    Layer 1: Embedding 類似検索で上位 top_k 論文を高速抽出（< 3秒）
    Layer 2: サブセットに GraphRAG インデックスを構築 → クエリ実行

    Args:
        query: 検索クエリ（日本語/英語）
        top_k: Layer 1 で抽出する論文数 (default: 10, 推奨: 10-50)
        search_type: GraphRAG 検索方法 "local"|"global"|"drift" (default: "local")
        provider: Embedding プロバイダー "openai"|"ollama" (default: 環境変数)
    """
    if not TWO_LAYER_AVAILABLE:
        return "Error: 2層検索モジュールが利用できません。two_layer_search.py を確認してください。"

    provider_name = provider or _get_two_layer_provider_name()
    lancedb_dir = _get_lancedb_dir(provider_name)
    logger.info(
        "2-layer query: %s (top_k=%d, search=%s, provider=%s)",
        query, top_k, search_type, provider_name,
    )

    # セマフォで同時ビルド数を制限（マルチユーザー保護）
    async with _two_layer_semaphore:
        try:
            # settings.yaml のデフォルト
            settings_src = PROJECT_ROOT / "settings.yaml"
            if not settings_src.exists():
                settings_src = None

            # ブロッキング処理を別スレッドで実行
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: run_two_layer_search(
                    query=query,
                    search_type=search_type,
                    top_k=top_k,
                    provider_name=provider_name,
                    lancedb_dir=lancedb_dir,
                    papers_dir=DEFAULT_PAPERS_DIR,
                    prompts_dir=DEFAULT_PROMPTS_DIR,
                    env_file=DEFAULT_ENV_FILE,
                    settings_src=settings_src,
                    workspace_base=DEFAULT_WORKSPACE_BASE,
                    cache_dir=DEFAULT_CACHE_DIR,
                    use_cache=True,
                ),
            )
            return answer
        except Exception as e:
            logger.error("2-layer query failed: %s", e)
            return f"Error: {e}"


@mcp.tool()
async def two_layer_quick_search(
    query: str,
    top_k: int = 20,
    provider: str = "",
) -> str:
    """Layer 1 のみの高速 Embedding 検索。関連論文のリストを距離付きで返します。
    GraphRAG 構築は行わないため、数秒以内で結果を取得できます。

    Args:
        query: 検索クエリ（日本語/英語）
        top_k: 取得する論文数 (default: 20)
        provider: Embedding プロバイダー "openai"|"ollama" (default: 環境変数)
    """
    if not TWO_LAYER_AVAILABLE:
        return "Error: 2層検索モジュールが利用できません。"

    provider_name = provider or _get_two_layer_provider_name()
    lancedb_dir = _get_lancedb_dir(provider_name)
    logger.info("Layer 1 quick search: %s (top_k=%d)", query, top_k)

    try:
        import lancedb as _lancedb

        loop = asyncio.get_event_loop()

        def _do_search():
            emb_provider = create_provider(provider_name)
            paper_ids, query_emb = search_relevant_papers(
                query, emb_provider, lancedb_dir, top_k,
            )
            # 距離情報付きで取得
            db = _lancedb.connect(str(lancedb_dir))
            table = db.open_table("papers")
            results = table.search(query_emb).limit(top_k * 5).to_pandas()
            paper_scores = results.groupby("paper_id")["_distance"].min().sort_values()
            top_papers = paper_scores.head(top_k)

            lines = [f"Layer 1 検索結果: {len(top_papers)} 論文\n"]
            for i, (pid, dist) in enumerate(top_papers.items(), 1):
                lines.append(f"  {i:3d}. {pid}  (距離: {dist:.4f})")
            return "\n".join(lines)

        return await loop.run_in_executor(None, _do_search)
    except Exception as e:
        logger.error("Quick search failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def two_layer_cache_status() -> str:
    """2層検索のキャッシュ状態・統計を確認します。
    キャッシュエントリ数、TTL、類似度閾値、アクセス回数などを返します。
    """
    if not TWO_LAYER_AVAILABLE:
        return "Error: 2層検索モジュールが利用できません。"

    logger.info("Cache status check")
    try:
        async with _cache_lock:
            loop = asyncio.get_event_loop()

            def _do_status():
                cache = IndexCache(DEFAULT_CACHE_DIR)
                stats = cache.get_stats()
                entries = cache.list_entries()

                result = {
                    "statistics": stats,
                    "entries": [
                        {
                            "query": e["query"][:80],
                            "paper_count": e["paper_count"],
                            "access_count": e["access_count"],
                            "remaining_hours": e["remaining_hours"],
                        }
                        for e in entries
                    ],
                }
                return json.dumps(result, ensure_ascii=False, indent=2)

            return await loop.run_in_executor(None, _do_status)
    except Exception as e:
        logger.error("Cache status failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def two_layer_cache_clear() -> str:
    """2層検索のキャッシュを全て削除します。
    ディスク上のワークスペースディレクトリも削除されます。
    """
    if not TWO_LAYER_AVAILABLE:
        return "Error: 2層検索モジュールが利用できません。"

    logger.info("Cache clear requested")
    try:
        async with _cache_lock:
            loop = asyncio.get_event_loop()

            def _do_clear():
                cache = IndexCache(DEFAULT_CACHE_DIR)
                entries = cache.list_entries()
                count = len(entries)
                cache.clear_all()
                return count

            count = await loop.run_in_executor(None, _do_clear)
            return f"キャッシュを削除しました（{count} エントリ）"
    except Exception as e:
        logger.error("Cache clear failed: %s", e)
        return f"Error: {e}"


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
