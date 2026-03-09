#!/usr/bin/env python3
"""
Layer 1: 論文 embedding インデックスを構築・検索する。

プロバイダー:
  ollama   — Ollama bge-m3 (ローカル/Windows, 無料)
  openai   — OpenAI text-embedding-3-small ($0.02/1M tokens)

使用方法:
    # Ollama (デフォルト)
    python build_embedding_index.py build --papers-dir markdown --limit 1000

    # OpenAI
    python build_embedding_index.py --provider openai build --papers-dir markdown --limit 1000

    # 検索
    python build_embedding_index.py search --query "Ti合金の疲労特性"
    python build_embedding_index.py --provider openai search --query "fatigue"

環境変数:
    OLLAMA_HOST      Ollama サーバー URL (デフォルト: 自動検出)
    OPENAI_API_KEY   OpenAI API キー (--provider openai 時に必要)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import lancedb
import pyarrow as pa
import requests


# ─── プロバイダー設定 ─────────────────────────────────────────
PROVIDERS = {
    "ollama": {
        "model": "bge-m3",
        "dim": 1024,
        "batch_size": 32,
        "lancedb_default": "lancedb_index",
        "cost_per_1m_tokens": 0.0,
    },
    "openai": {
        "model": "text-embedding-3-small",
        "dim": 1536,
        "batch_size": 64,
        "lancedb_default": "lancedb_openai",
        "cost_per_1m_tokens": 0.02,
    },
}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
TABLE_NAME = "papers"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Embedding プロバイダー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EmbeddingProvider:
    def __init__(self, model: str, dim: int, batch_size: int,
                 cost_per_1m: float):
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.cost_per_1m = cost_per_1m
        self._total_tokens = 0

    @property
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def cost(self) -> float:
        return self._total_tokens / 1_000_000 * self.cost_per_1m


class OllamaProvider(EmbeddingProvider):
    def __init__(self, model: str, dim: int, batch_size: int):
        super().__init__(model, dim, batch_size, cost_per_1m=0.0)
        self.host = self._detect_host()

    @property
    def provider_name(self) -> str:
        return f"ollama ({self.host})"

    @staticmethod
    def _detect_host() -> str:
        env = os.environ.get("OLLAMA_HOST")
        if env:
            return env.rstrip("/")
        for host in ["http://localhost:11434"]:
            try:
                r = requests.get(f"{host}/api/tags", timeout=3)
                if r.status_code == 200:
                    return host
            except requests.ConnectionError:
                pass
        resolv = Path("/etc/resolv.conf")
        if resolv.exists():
            for line in resolv.read_text().splitlines():
                if line.strip().startswith("nameserver"):
                    ip = line.split()[1]
                    host = f"http://{ip}:11434"
                    try:
                        r = requests.get(f"{host}/api/tags", timeout=3)
                        if r.status_code == 200:
                            return host
                    except requests.ConnectionError:
                        pass
        print("ERROR: Ollama サーバーに接続できません。", file=sys.stderr)
        sys.exit(1)

    def embed(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t if t.strip() else "empty" for t in texts]
        try:
            resp = requests.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": cleaned},
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except requests.HTTPError:
            results = []
            for t in cleaned:
                try:
                    r = requests.post(
                        f"{self.host}/api/embed",
                        json={"model": self.model, "input": [t[:8000]]},
                        timeout=120,
                    )
                    r.raise_for_status()
                    results.append(r.json()["embeddings"][0])
                except Exception:
                    results.append([0.0] * self.dim)
            return results


class OpenAIProvider(EmbeddingProvider):
    def __init__(self, model: str, dim: int, batch_size: int):
        super().__init__(model, dim, batch_size, cost_per_1m=0.02)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            for env_file in [Path(".env"), Path.home() / ".env"]:
                if env_file.exists():
                    for line in env_file.read_text().splitlines():
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if api_key:
                    break
        if not api_key:
            print("ERROR: OPENAI_API_KEY が設定されていません。", file=sys.stderr)
            sys.exit(1)
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return f"openai ({self.model})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        # text-embedding-3-small の上限は 8192 tokens ≈ 6000 words
        cleaned = [t[:24000] if t.strip() else "empty" for t in texts]
        try:
            resp = self._client.embeddings.create(model=self.model, input=cleaned)
        except Exception as e:
            # バッチ内に長すぎるテキストがある場合、1件ずつフォールバック
            results = []
            for t in cleaned:
                try:
                    r = self._client.embeddings.create(
                        model=self.model, input=[t[:12000]])
                    self._total_tokens += r.usage.total_tokens
                    results.append(r.data[0].embedding)
                except Exception:
                    results.append([0.0] * self.dim)
            return results
        self._total_tokens += resp.usage.total_tokens
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


def create_provider(name: str, model: str | None = None) -> EmbeddingProvider:
    cfg = PROVIDERS[name]
    m = model or cfg["model"]
    dim = cfg["dim"]
    bs = cfg["batch_size"]
    if name == "ollama":
        return OllamaProvider(m, dim, bs)
    elif name == "openai":
        return OpenAIProvider(m, dim, bs)
    else:
        raise ValueError(f"Unknown provider: {name}")


# ─── テキスト前処理 ───────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ─── メイン処理 ───────────────────────────────────────────────
def build_embedding_index(
    papers_dir: Path,
    lancedb_dir: Path,
    provider: EmbeddingProvider,
    limit: int | None = None,
) -> dict:
    print(f"Provider:  {provider.provider_name}")
    print(f"Model:     {provider.model}")
    print(f"Dimension: {provider.dim}")

    md_files = sorted(papers_dir.glob("*.md"))
    if limit:
        md_files = md_files[:limit]
    total_files = len(md_files)
    print(f"対象論文:   {total_files} 本")

    # LanceDB
    lancedb_dir = Path(lancedb_dir)
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lancedb_dir))
    schema = pa.schema([
        pa.field("paper_id", pa.string()),
        pa.field("chunk_id", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), provider.dim)),
    ])
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    table = db.create_table(TABLE_NAME, schema=schema)

    total_chunks = 0
    total_words = 0
    t_start = time.time()
    batch_texts: list[str] = []
    batch_meta: list[dict] = []

    def flush_batch():
        nonlocal total_chunks
        if not batch_texts:
            return
        embeddings = provider.embed(batch_texts)
        records = [
            {
                "paper_id": meta["paper_id"],
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "source_file": meta["source_file"],
                "vector": emb,
            }
            for meta, emb in zip(batch_meta, embeddings)
        ]
        table.add(records)
        total_chunks += len(records)
        batch_texts.clear()
        batch_meta.clear()

    for idx, md_file in enumerate(md_files):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        words = text.split()
        total_words += len(words)
        chunks = chunk_text(text)
        if not chunks:
            continue
        for ci, chunk in enumerate(chunks):
            batch_texts.append(chunk)
            batch_meta.append({
                "paper_id": md_file.stem,
                "chunk_id": ci,
                "text": chunk,
                "source_file": str(md_file),
            })
            if len(batch_texts) >= provider.batch_size:
                flush_batch()
        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_files - idx - 1) / rate if rate > 0 else 0
            cost_now = provider.cost()
            cost_str = f"  cost=${cost_now:.4f}" if cost_now > 0 else ""
            print(f"  [{idx+1}/{total_files}] "
                  f"chunks={total_chunks}  "
                  f"{rate:.1f} papers/s  "
                  f"ETA {eta:.0f}s{cost_str}")

    flush_batch()
    elapsed = time.time() - t_start
    estimated_tokens = int(total_words * 1.3)
    actual_cost = provider.cost()

    stats = {
        "provider": provider.provider_name,
        "papers": total_files,
        "chunks": total_chunks,
        "total_words": total_words,
        "estimated_tokens": estimated_tokens,
        "api_reported_tokens": provider.total_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "model": provider.model,
        "embedding_dim": provider.dim,
        "chunk_size": CHUNK_SIZE,
        "cost_usd": round(actual_cost, 6),
        "lancedb_dir": str(lancedb_dir),
    }

    # レポート
    print("\n" + "=" * 60)
    print("Layer 1 Embedding Index - 構築完了")
    print("=" * 60)
    print(f"  プロバイダー:      {provider.provider_name}")
    print(f"  論文数:           {stats['papers']:,}")
    print(f"  チャンク数:        {stats['chunks']:,}")
    print(f"  総ワード数:        {stats['total_words']:,}")
    print(f"  推定トークン数:     {stats['estimated_tokens']:,}")
    if provider.total_tokens > 0:
        print(f"  API報告トークン数:  {provider.total_tokens:,}")
    print(f"  所要時間:          {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
    print(f"  処理速度:          {total_files/elapsed:.1f} papers/s")
    print(f"  コスト:            ${actual_cost:.4f}")
    print()
    factor = 100_000 / total_files if total_files > 0 else 1
    tokens_100k = (provider.total_tokens if provider.total_tokens > 0
                   else estimated_tokens) * factor
    cost_100k = tokens_100k / 1_000_000 * provider.cost_per_1m
    print("── 10万本への外挿 ──")
    print(f"  推定チャンク数:     {int(total_chunks * factor):,}")
    print(f"  推定トークン数:     {int(tokens_100k):,}")
    print(f"  推定コスト:        ${cost_100k:.2f}")
    print(f"  推定時間:          {elapsed * factor / 60:.0f}分 "
          f"({elapsed * factor / 3600:.1f}時間)")
    print("=" * 60)

    stats_file = lancedb_dir / "build_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n統計情報: {stats_file}")
    return stats


# ─── 検索テスト ───────────────────────────────────────────────
def search_test(lancedb_dir: Path, query: str, top_k: int,
                provider: EmbeddingProvider):
    db = lancedb.connect(str(lancedb_dir))
    table = db.open_table(TABLE_NAME)
    query_emb = provider.embed([query])[0]
    results = table.search(query_emb).limit(top_k * 5).to_pandas()
    paper_scores = results.groupby("paper_id")["_distance"].min().sort_values()
    top_papers = paper_scores.head(top_k)

    print(f"\n── 検索結果: '{query}' (top {top_k}) [{provider.provider_name}] ──")
    for rank, (paper_id, dist) in enumerate(top_papers.items(), 1):
        best = results[results["paper_id"] == paper_id].sort_values("_distance").iloc[0]
        snippet = best["text"][:120].replace("\n", " ")
        print(f"  {rank}. [{dist:.4f}] {paper_id}")
        print(f"     {snippet}...")


# ─── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: 論文 embedding インデックス構築 (Ollama / OpenAI)")
    parser.add_argument("--provider", choices=["ollama", "openai"],
                        default="ollama",
                        help="Embedding プロバイダー (デフォルト: ollama)")
    parser.add_argument("--model", default=None,
                        help="モデル名 (省略時はプロバイダーのデフォルト)")
    sub = parser.add_subparsers(dest="command", help="サブコマンド")

    p_build = sub.add_parser("build", help="インデックスを構築")
    p_build.add_argument("--papers-dir", type=Path, required=True,
                         help="Markdown 論文のディレクトリ")
    p_build.add_argument("--lancedb-dir", type=Path, default=None,
                         help="LanceDB 格納先 (デフォルト: プロバイダー依存)")
    p_build.add_argument("--limit", type=int, default=None,
                         help="処理する論文数の上限")

    p_search = sub.add_parser("search", help="検索テスト")
    p_search.add_argument("--lancedb-dir", type=Path, default=None,
                          help="LanceDB 格納先")
    p_search.add_argument("--query", type=str, required=True,
                          help="検索クエリ")
    p_search.add_argument("--top-k", type=int, default=10,
                          help="返す論文数 (デフォルト: 10)")

    args = parser.parse_args()
    cfg = PROVIDERS[args.provider]
    lancedb_dir = (args.lancedb_dir if hasattr(args, 'lancedb_dir') and args.lancedb_dir
                   else Path(cfg["lancedb_default"]))
    prov = create_provider(args.provider, args.model)

    if args.command == "build":
        build_embedding_index(
            papers_dir=args.papers_dir,
            lancedb_dir=lancedb_dir,
            provider=prov,
            limit=args.limit,
        )
    elif args.command == "search":
        search_test(lancedb_dir=lancedb_dir, query=args.query,
                    top_k=args.top_k, provider=prov)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
