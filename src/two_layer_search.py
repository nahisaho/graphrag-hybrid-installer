#!/usr/bin/env python3
"""
2層アーキテクチャ検索パイプライン v1.1.0
==========================================
Layer 1: Embedding 検索で関連論文サブセットを高速抽出
Layer 2: サブセットに対して GraphRAG (FastGraphRAG) インデックスを構築 → クエリ

使用方法:
    # ── Layer 1 インデックス構築（初回のみ） ──
    python two_layer_search.py build-index
    python two_layer_search.py build-index --provider openai --limit 1000

    # ── Layer 1 のみの高速検索 ──
    python two_layer_search.py search "Ti合金の疲労特性" --top-k 20

    # ── 2層検索（Layer 1 + Layer 2 GraphRAG） ──
    python two_layer_search.py query "Ti-Nb-Ta-Zr合金の疲労特性と微細構造の関係"
    python two_layer_search.py query "材料強度に影響する因子" --search-type global
    python two_layer_search.py query "水素脆化のメカニズム" --top-k 200

    # ── キャッシュ管理 ──
    python two_layer_search.py cache-list
    python two_layer_search.py cache-clear

環境変数:
    GRAPHRAG_API_KEY   OpenAI API キー
    OLLAMA_HOST        Ollama サーバー URL (デフォルト: 自動検出)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import lancedb
import numpy as np


# ─── パス設定 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: src/ の親ディレクトリ。
# 環境変数 GRAPHRAG_ROOT があればそちらを優先。
PROJECT_ROOT = Path(os.environ.get("GRAPHRAG_ROOT", str(SCRIPT_DIR.parent)))

# デフォルトパス (CLI で上書き可能)
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "input"
DEFAULT_PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_WORKSPACE_BASE = PROJECT_ROOT / "graphrag_workspaces"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "graphrag_cache"

# Layer 1 embedding インデックス
DEFAULT_LANCEDB_DIR = PROJECT_ROOT / "lancedb_index"
TABLE_NAME = "papers"

# Layer 2 GraphRAG 構築パラメータ
FASTGRAPHRAG_CHUNK_SIZE = 100   # FastGraphRAG 推奨: 50-100 tokens
FASTGRAPHRAG_CHUNK_OVERLAP = 10

# キャッシュ設定
CACHE_TTL_HOURS = 24
SIMILARITY_THRESHOLD = 0.85


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Embedding プロバイダー（build_embedding_index.py から再利用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sys.path.insert(0, str(SCRIPT_DIR))
from build_embedding_index import (
    OllamaProvider,
    OpenAIProvider,
    create_provider,
    PROVIDERS,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  インデックスキャッシュ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """コサイン類似度を計算"""
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class IndexCache:
    """類似クエリの GraphRAG インデックスを再利用するキャッシュ"""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR,
                 ttl_hours: int = CACHE_TTL_HOURS,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.similarity_threshold = similarity_threshold
        self.manifest_file = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_file.exists():
            return json.loads(self.manifest_file.read_text(encoding="utf-8"))
        return {"entries": []}

    def _save_manifest(self):
        self.manifest_file.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def find_cached_index(self, query_embedding: list[float]) -> dict | None:
        """類似クエリのキャッシュ済みインデックスを検索"""
        now = datetime.now()
        valid_entries = []

        for entry in self.manifest["entries"]:
            created = datetime.fromisoformat(entry["created_at"])
            if now - created > timedelta(hours=self.ttl_hours):
                continue
            index_path = Path(entry["index_path"])
            if not index_path.exists():
                continue
            valid_entries.append(entry)

        # 有効なエントリの中で最も類似度が高いものを検索
        best_entry = None
        best_similarity = 0.0

        for entry in valid_entries:
            sim = cosine_similarity(query_embedding, entry["query_embedding"])
            if sim >= self.similarity_threshold and sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_entry:
            print(f"  キャッシュヒット! 類似度: {best_similarity:.3f}")
            print(f"  元クエリ: {best_entry['query'][:60]}...")
            return best_entry

        return None

    def register_index(self, query: str, query_embedding: list[float],
                       index_path: Path, paper_ids: list[str]):
        """新しいインデックスをキャッシュに登録"""
        entry = {
            "query": query,
            "query_embedding": (query_embedding.tolist()
                                if hasattr(query_embedding, "tolist")
                                else list(query_embedding)),
            "index_path": str(index_path),
            "paper_count": len(paper_ids),
            "created_at": datetime.now().isoformat(),
        }
        self.manifest["entries"].append(entry)
        self._save_manifest()
        print(f"  キャッシュ登録: {len(paper_ids)} 論文, {index_path}")

    def clean_expired(self):
        """期限切れエントリを削除"""
        now = datetime.now()
        valid = []
        removed = 0
        for entry in self.manifest["entries"]:
            created = datetime.fromisoformat(entry["created_at"])
            if now - created > timedelta(hours=self.ttl_hours):
                idx_path = Path(entry["index_path"])
                if idx_path.exists():
                    shutil.rmtree(idx_path, ignore_errors=True)
                removed += 1
            else:
                valid.append(entry)
        self.manifest["entries"] = valid
        self._save_manifest()
        return removed

    def clear_all(self):
        """全キャッシュを削除"""
        for entry in self.manifest["entries"]:
            idx_path = Path(entry["index_path"])
            if idx_path.exists():
                shutil.rmtree(idx_path, ignore_errors=True)
        self.manifest["entries"] = []
        self._save_manifest()

    def list_entries(self) -> list[dict]:
        """有効なキャッシュエントリ一覧を返す"""
        now = datetime.now()
        result = []
        for entry in self.manifest["entries"]:
            created = datetime.fromisoformat(entry["created_at"])
            remaining = timedelta(hours=self.ttl_hours) - (now - created)
            if remaining.total_seconds() > 0:
                result.append({
                    "query": entry["query"],
                    "paper_count": entry["paper_count"],
                    "created_at": entry["created_at"],
                    "remaining_hours": round(remaining.total_seconds() / 3600, 1),
                    "index_path": entry["index_path"],
                })
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 1: Embedding 検索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def search_relevant_papers(
    query: str,
    provider: OllamaProvider | OpenAIProvider,
    lancedb_dir: Path,
    top_k: int,
) -> tuple[list[str], list[float]]:
    """
    Layer 1: クエリに関連する論文を embedding 検索で取得。
    Returns: (paper_ids, query_embedding)
    """
    db = lancedb.connect(str(lancedb_dir))
    table = db.open_table(TABLE_NAME)

    query_embedding = provider.embed([query])[0]

    # チャンクレベルで検索して論文単位で集約
    results = table.search(query_embedding).limit(top_k * 5).to_pandas()
    paper_scores = results.groupby("paper_id")["_distance"].min().sort_values()
    selected_papers = paper_scores.head(top_k).index.tolist()

    return selected_papers, query_embedding


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 2: ワークスペース準備 + GraphRAG 構築
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _generate_workspace_settings(
    workspace_dir: Path,
    settings_src: Path | None = None,
) -> Path:
    """
    settings.yaml を workspace に生成。
    元の settings.yaml をベースに、FastGraphRAG 用の chunk_size を適用。
    """
    if settings_src and settings_src.exists():
        import yaml
        with open(settings_src, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # FastGraphRAG 推奨のチャンクサイズに変更
    if "chunking" not in cfg:
        cfg["chunking"] = {}
    cfg["chunking"]["size"] = FASTGRAPHRAG_CHUNK_SIZE
    cfg["chunking"]["overlap"] = FASTGRAPHRAG_CHUNK_OVERLAP

    # input_storage を workspace の input ディレクトリに向ける
    cfg["input_storage"] = {"type": "file", "base_dir": "input"}

    settings_path = workspace_dir / "settings.yaml"
    import yaml
    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)

    return settings_path


def _find_domain_dictionary(project_root: Path) -> Path | None:
    """ドメイン辞書ファイルを検索"""
    candidates = [
        project_root / "domain_dictionary.json",
        project_root / "data" / "comprehensive_domain_dictionary.json",
        project_root / "data" / "domain_dictionary.json",
    ]
    env_path = os.environ.get("GRAPHRAG_DOMAIN_DICTIONARY")
    if env_path:
        candidates.insert(0, Path(env_path))
    for p in candidates:
        if p.exists():
            return p
    return None


def prepare_workspace(
    workspace_dir: Path,
    paper_ids: list[str],
    papers_dir: Path,
    prompts_dir: Path,
    env_file: Path,
    settings_src: Path | None = None,
) -> Path:
    """
    GraphRAG ワークスペースを準備する:
    1. 論文をコピー
    2. settings.yaml を生成
    3. prompts/ をコピー
    4. .env をコピー
    5. ドメイン辞書をコピー
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    input_dir = workspace_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 既存ファイルクリア
    for f in input_dir.glob("*.md"):
        f.unlink()

    # 論文コピー
    copied = 0
    for paper_id in paper_ids:
        src = papers_dir / f"{paper_id}.md"
        if src.exists():
            shutil.copy2(src, input_dir / src.name)
            copied += 1
    print(f"  論文コピー: {copied}/{len(paper_ids)} 本")

    # settings.yaml 生成
    _generate_workspace_settings(workspace_dir, settings_src)
    print(f"  settings.yaml 生成完了 (chunk_size={FASTGRAPHRAG_CHUNK_SIZE})")

    # prompts/ コピー
    ws_prompts = workspace_dir / "prompts"
    if ws_prompts.exists():
        shutil.rmtree(ws_prompts)
    if prompts_dir.exists():
        shutil.copytree(prompts_dir, ws_prompts)
        print(f"  prompts/ コピー完了")

    # .env コピー
    if env_file.exists():
        shutil.copy2(env_file, workspace_dir / ".env")
        print(f"  .env コピー完了")

    # ドメイン辞書コピー
    dict_src = _find_domain_dictionary(PROJECT_ROOT)
    if dict_src:
        shutil.copy2(dict_src, workspace_dir / "domain_dictionary.json")
        print(f"  ドメイン辞書コピー完了: {dict_src.name}")

    return workspace_dir


def _make_subprocess_env(workspace_dir: Path) -> dict[str, str]:
    """サブプロセス用の環境変数を構築（GRAPHRAG_ROOT を設定）"""
    env = os.environ.copy()
    env["GRAPHRAG_ROOT"] = str(workspace_dir)
    # ドメイン辞書パスを明示的に設定
    dict_path = workspace_dir / "domain_dictionary.json"
    if dict_path.exists():
        env["GRAPHRAG_DOMAIN_DICTIONARY"] = str(dict_path)
    return env


def build_graphrag_index(workspace_dir: Path) -> bool:
    """GraphRAG FastGraphRAG インデックスを構築"""
    run_script = SCRIPT_DIR / "run_graphrag_hybrid.py"

    cmd = [
        sys.executable, str(run_script),
        "index",
        "--root", str(workspace_dir),
        "--method", "fast",
    ]

    print(f"\n  GraphRAG index 構築開始...")
    print(f"  コマンド: {' '.join(cmd)}")

    env = _make_subprocess_env(workspace_dir)

    result = subprocess.run(
        cmd,
        cwd=str(workspace_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )

    if result.returncode != 0:
        print(f"\n  ERROR: GraphRAG index 構築失敗")
        print(f"  stderr: {result.stderr[-500:]}")
        return False

    # 構築結果の最後の行を表示
    stdout_lines = result.stdout.strip().split("\n")
    for line in stdout_lines[-5:]:
        print(f"  {line}")

    print(f"  GraphRAG index 構築完了")
    return True


def query_graphrag(workspace_dir: Path, query: str,
                   search_type: str = "local") -> str:
    """GraphRAG クエリ実行"""
    run_script = SCRIPT_DIR / "run_graphrag_hybrid.py"

    cmd = [
        sys.executable, str(run_script),
        "query",
        "--root", str(workspace_dir),
        "--method", search_type,
        query,
    ]

    print(f"\n  GraphRAG query 実行中 (method={search_type})...")

    env = _make_subprocess_env(workspace_dir)

    result = subprocess.run(
        cmd,
        cwd=str(workspace_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout
    )

    if result.returncode != 0:
        return f"ERROR: GraphRAG query 失敗\n{result.stderr[-500:]}"

    return result.stdout


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  メインパイプライン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_two_layer_search(
    query: str,
    search_type: str = "local",
    top_k: int = 100,
    provider_name: str = "ollama",
    lancedb_dir: Path = DEFAULT_LANCEDB_DIR,
    papers_dir: Path = DEFAULT_PAPERS_DIR,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    env_file: Path = DEFAULT_ENV_FILE,
    settings_src: Path | None = None,
    workspace_base: Path = DEFAULT_WORKSPACE_BASE,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> str:
    """
    2層アーキテクチャによる検索パイプライン

    1. Layer 1: Embedding で上位 top_k 論文を高速抽出
    2. キャッシュチェック: 類似クエリのインデックスがあれば再利用
    3. Layer 2: サブセット → ワークスペース準備 → GraphRAG 構築
    4. GraphRAG クエリ実行
    """
    t_start = time.time()
    print("=" * 60)
    print(f"2層検索パイプライン")
    print(f"=" * 60)
    print(f"  クエリ:     {query}")
    print(f"  検索方法:   {search_type}")
    print(f"  サブセット: {top_k} 論文")
    print(f"  プロバイダー: {provider_name}")
    print()

    # ── Layer 1: Embedding 検索 ──
    print("── Layer 1: Embedding 検索 ──")
    provider = create_provider(provider_name)
    paper_ids, query_embedding = search_relevant_papers(
        query, provider, lancedb_dir, top_k,
    )
    print(f"  {len(paper_ids)} 論文を抽出 ({time.time() - t_start:.1f}秒)")

    if not paper_ids:
        return "ERROR: 関連論文が見つかりませんでした。"

    # ── キャッシュチェック ──
    cache = IndexCache(cache_dir) if use_cache else None
    workspace_dir = None

    if cache:
        print("\n── キャッシュチェック ──")
        cache.clean_expired()
        cached = cache.find_cached_index(query_embedding)
        if cached:
            workspace_dir = Path(cached["index_path"])

    # ── Layer 2: GraphRAG 構築 ──
    if workspace_dir is None:
        print("\n── Layer 2: GraphRAG ワークスペース準備 ──")
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_dir = workspace_base / f"ws_{timestamp}_{query_hash}"

        prepare_workspace(
            workspace_dir=workspace_dir,
            paper_ids=paper_ids,
            papers_dir=papers_dir,
            prompts_dir=prompts_dir,
            env_file=env_file,
            settings_src=settings_src,
        )

        print("\n── Layer 2: GraphRAG インデックス構築 ──")
        success = build_graphrag_index(workspace_dir)
        if not success:
            return "ERROR: GraphRAG インデックス構築に失敗しました。"

        # キャッシュ登録
        if cache:
            cache.register_index(query, query_embedding,
                                 workspace_dir, paper_ids)
    else:
        print(f"  キャッシュ済みインデックスを使用: {workspace_dir}")

    # ── GraphRAG クエリ実行 ──
    print("\n── GraphRAG クエリ実行 ──")
    answer = query_graphrag(workspace_dir, query, search_type)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  総所要時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
    print(f"{'=' * 60}")

    return answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(
        description="2層アーキテクチャ検索: Embedding → GraphRAG",
    )
    parser.add_argument("--provider", choices=["ollama", "openai"],
                        default="ollama",
                        help="Embedding プロバイダー (デフォルト: ollama)")
    parser.add_argument("--lancedb-dir", type=Path, default=None,
                        help="Layer 1 LanceDB パス")
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR,
                        help="Markdown 論文ディレクトリ")
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR,
                        help="プロンプトディレクトリ")
    parser.add_argument("--settings", type=Path, default=None,
                        help="ベース settings.yaml パス")
    parser.add_argument("--workspace-base", type=Path,
                        default=DEFAULT_WORKSPACE_BASE,
                        help="ワークスペース親ディレクトリ")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="キャッシュディレクトリ")
    parser.add_argument("--no-cache", action="store_true",
                        help="キャッシュを無効化")

    sub = parser.add_subparsers(dest="command", help="サブコマンド")

    # ── build-index サブコマンド ──
    p_build = sub.add_parser("build-index",
                             help="Layer 1 Embedding インデックスを構築")
    p_build.add_argument("--limit", type=int, default=None,
                         help="処理する論文数の上限")

    # ── search サブコマンド（Layer 1 のみ） ──
    p_search = sub.add_parser("search",
                              help="Layer 1 Embedding 検索のみ（高速）")
    p_search.add_argument("query_text", type=str, help="検索クエリ")
    p_search.add_argument("--top-k", type=int, default=20,
                          help="取得する論文数 (デフォルト: 20)")

    # ── query サブコマンド ──
    p_query = sub.add_parser("query", help="2層検索を実行")
    p_query.add_argument("query_text", type=str, help="検索クエリ")
    p_query.add_argument("--search-type", choices=["local", "global", "drift"],
                         default="local", help="GraphRAG 検索方法")
    p_query.add_argument("--top-k", type=int, default=100,
                         help="Layer 1 で取得する論文数 (デフォルト: 100)")

    # ── cache-list サブコマンド ──
    sub.add_parser("cache-list", help="キャッシュ一覧を表示")

    # ── cache-clear サブコマンド ──
    sub.add_parser("cache-clear", help="全キャッシュを削除")

    args = parser.parse_args()

    if args.command == "build-index":
        # Layer 1 インデックス構築を build_embedding_index.py に委譲
        from build_embedding_index import build_embedding_index
        cfg = PROVIDERS[args.provider]
        lancedb_dir = args.lancedb_dir or Path(cfg["lancedb_default"])
        provider = create_provider(args.provider)
        build_embedding_index(
            papers_dir=args.papers_dir,
            lancedb_dir=lancedb_dir,
            provider=provider,
            limit=args.limit,
        )

    elif args.command == "search":
        # Layer 1 のみの高速検索
        cfg = PROVIDERS[args.provider]
        lancedb_dir = args.lancedb_dir or Path(cfg["lancedb_default"])
        provider = create_provider(args.provider)

        print(f"── Layer 1: Embedding 検索 ──")
        print(f"  クエリ: {args.query_text}")
        print(f"  top-k: {args.top_k}")
        print()

        paper_ids, _ = search_relevant_papers(
            args.query_text, provider, lancedb_dir, args.top_k,
        )

        # 距離情報付きで表示
        db = lancedb.connect(str(lancedb_dir))
        table = db.open_table(TABLE_NAME)
        query_emb = provider.embed([args.query_text])[0]
        results = table.search(query_emb).limit(args.top_k * 5).to_pandas()
        paper_scores = results.groupby("paper_id")["_distance"].min().sort_values()
        top_papers = paper_scores.head(args.top_k)

        print(f"  {len(top_papers)} 論文ヒット:")
        print(f"  {'#':>3}  {'距離':>8}  {'論文ID'}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*40}")
        for i, (pid, dist) in enumerate(top_papers.items(), 1):
            print(f"  {i:3d}  {dist:8.4f}  {pid}")

    elif args.command == "query":
        cfg = PROVIDERS[args.provider]
        lancedb_dir = args.lancedb_dir or Path(cfg["lancedb_default"])

        # settings.yaml のデフォルト: プロジェクトルートの settings.yaml
        settings_src = args.settings
        if settings_src is None:
            default_settings = PROJECT_ROOT / "settings.yaml"
            if default_settings.exists():
                settings_src = default_settings

        answer = run_two_layer_search(
            query=args.query_text,
            search_type=args.search_type,
            top_k=args.top_k,
            provider_name=args.provider,
            lancedb_dir=lancedb_dir,
            papers_dir=args.papers_dir,
            prompts_dir=args.prompts_dir,
            settings_src=settings_src,
            workspace_base=args.workspace_base,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
        )
        print("\n── 回答 ──")
        print(answer)

    elif args.command == "cache-list":
        cache = IndexCache(args.cache_dir)
        entries = cache.list_entries()
        if not entries:
            print("キャッシュは空です。")
        else:
            print(f"キャッシュエントリ: {len(entries)} 件")
            print("-" * 60)
            for i, e in enumerate(entries, 1):
                print(f"  {i}. {e['query'][:60]}...")
                print(f"     論文数: {e['paper_count']}, "
                      f"残り: {e['remaining_hours']}h, "
                      f"パス: {e['index_path']}")

    elif args.command == "cache-clear":
        cache = IndexCache(args.cache_dir)
        cache.clear_all()
        print("全キャッシュを削除しました。")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
