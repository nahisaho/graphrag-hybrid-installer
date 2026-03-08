"""
GraphRAG NLP Edge Extraction Optimizer
======================================
Patches graphrag's build_noun_graph.py to fix the "relationship explosion"
problem in Lazy (Fast) mode.

Problem:
  The original _extract_edges() uses itertools.combinations() to create
  ALL PAIRS of entities within each text chunk. With ~60 entities/chunk,
  this generates ~1,744 pairs/chunk → 65-70x more relationships than
  Standard mode → higher community_reports LLM costs.

Solution (3 strategies):
  1. Top-K entity limit (K=15): Only pair the K most frequent entities
     per chunk, reducing O(N²) → O(K²).
  2. Min co-occurrence filter (≥2): Keep only edges that appear in ≥2
     text chunks, removing coincidental co-occurrences.
  3. Academic stopword exclusion: Configured via settings.yaml
     exclude_nouns (handled separately).

Results (20 papers, gpt-4o-mini):
  - Relationships: 120,287 → 2,660 (97.8% reduction)
  - Cost: $0.929 → $0.097 (89.6% reduction)
  - Quality: Maintained or improved vs Standard

Usage:
  python3 patch_noun_graph.py [--max-k 15] [--min-cooccurrence 2] [--dry-run]

Reference:
  https://github.com/microsoft/graphrag
  graphrag/index/operations/build_noun_graph/build_noun_graph.py
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import textwrap

logger = logging.getLogger(__name__)

DEFAULT_MAX_K = 15
DEFAULT_MIN_COOCCURRENCE = 2


def find_target_file() -> str:
    """Locate build_noun_graph.py in the installed graphrag package."""
    spec = importlib.util.find_spec("graphrag")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "graphrag package not found. Install it first: pip install graphrag"
        )
    graphrag_root = os.path.dirname(spec.origin)
    target = os.path.join(
        graphrag_root,
        "index", "operations", "build_noun_graph", "build_noun_graph.py",
    )
    if not os.path.exists(target):
        raise FileNotFoundError(f"Target file not found: {target}")
    return target


def is_already_patched(filepath: str) -> bool:
    """Check if the file already contains our patch."""
    with open(filepath, "r") as f:
        content = f.read()
    return "max_entities_per_chunk" in content


def generate_patched_code(max_k: int, min_cooccurrence: int) -> str:
    """Generate the patched _extract_edges function."""
    return textwrap.dedent(f'''\
def _extract_edges(
    title_to_ids: dict[str, list[str]],
    nodes_df: pd.DataFrame,
    normalize_edge_weights: bool = True,
    max_entities_per_chunk: int = {max_k},
    min_co_occurrence: int = {min_cooccurrence},
) -> pd.DataFrame:
    """Build co-occurrence edges between noun phrases.

    Nodes that appear in the same text unit are connected.
    To control relationship explosion, two filters are applied:
      1. Top-K: Only the K most frequent entities per chunk are paired.
      2. Min co-occurrence: Edges must appear in ≥M text units.

    Returns edges with schema [source, target, weight, text_unit_ids].
    """
    if not title_to_ids:
        return pd.DataFrame(
            columns=["source", "target", "weight", "text_unit_ids"],
        )

    # Build global frequency map for Top-K selection
    entity_freq: dict[str, int] = {{
        t: len(ids) for t, ids in title_to_ids.items()
    }}

    text_unit_to_titles: dict[str, list[str]] = defaultdict(list)
    for title, tu_ids in title_to_ids.items():
        for tu_id in tu_ids:
            text_unit_to_titles[tu_id].append(title)

    edge_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tu_id, titles in text_unit_to_titles.items():
        unique_titles = sorted(set(titles))
        if len(unique_titles) < 2:
            continue
        # Top-K: keep only the most frequent entities per chunk
        if max_entities_per_chunk > 0 and len(unique_titles) > max_entities_per_chunk:
            unique_titles = sorted(
                unique_titles,
                key=lambda t: entity_freq.get(t, 0),
                reverse=True,
            )[:max_entities_per_chunk]
            unique_titles.sort()
        for pair in combinations(unique_titles, 2):
            edge_map[pair].append(tu_id)

    # Min co-occurrence filter
    records = [
        {{
            "source": src,
            "target": tgt,
            "weight": len(tu_ids),
            "text_unit_ids": tu_ids,
        }}
        for (src, tgt), tu_ids in edge_map.items()
        if min_co_occurrence <= 1 or len(tu_ids) >= min_co_occurrence
    ]

    logger.info(
        "Edge extraction: %d raw pairs -> %d after top-%d & co-occurrence>=%d",
        len(edge_map),
        len(records),
        max_entities_per_chunk,
        min_co_occurrence,
    )

    edges_df = pd.DataFrame(
        records,
        columns=["source", "target", "weight", "text_unit_ids"],
    )

    if normalize_edge_weights and not edges_df.empty:
        edges_df = calculate_pmi_edge_weights(nodes_df, edges_df)

    return edges_df
''')


def apply_patch(filepath: str, max_k: int, min_cooccurrence: int, dry_run: bool = False) -> bool:
    """Apply the patch to build_noun_graph.py."""
    with open(filepath, "r") as f:
        content = f.read()

    # Find the original _extract_edges function
    marker_start = "def _extract_edges("
    idx_start = content.find(marker_start)
    if idx_start < 0:
        logger.error("Cannot find _extract_edges() in %s", filepath)
        return False

    # Find the end of the function (next top-level def or end of file)
    remaining = content[idx_start:]
    lines = remaining.split("\n")
    func_lines = [lines[0]]
    for line in lines[1:]:
        # A new top-level definition or class at column 0 marks end
        if line and not line[0].isspace() and (line.startswith("def ") or line.startswith("class ") or line.startswith("async def ")):
            break
        func_lines.append(line)

    old_func = "\n".join(func_lines)
    new_func = generate_patched_code(max_k, min_cooccurrence)

    if dry_run:
        print(f"Would patch: {filepath}")
        print(f"  max_entities_per_chunk = {max_k}")
        print(f"  min_co_occurrence = {min_cooccurrence}")
        print(f"  Old function: {len(old_func)} chars")
        print(f"  New function: {len(new_func)} chars")
        return True

    # Create backup
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        logger.info("Backup created: %s", backup)

    new_content = content[:idx_start] + new_func + "\n"
    with open(filepath, "w") as f:
        f.write(new_content)

    logger.info("Patched: %s (K=%d, min_cooccurrence=%d)", filepath, max_k, min_cooccurrence)
    return True


def restore(filepath: str) -> bool:
    """Restore the original file from backup."""
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        logger.error("No backup found: %s", backup)
        return False
    shutil.copy2(backup, filepath)
    logger.info("Restored original: %s", filepath)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch GraphRAG build_noun_graph.py to optimize NLP edge extraction"
    )
    parser.add_argument(
        "--max-k", type=int, default=DEFAULT_MAX_K,
        help=f"Max entities per chunk for Top-K pairing (default: {DEFAULT_MAX_K})"
    )
    parser.add_argument(
        "--min-cooccurrence", type=int, default=DEFAULT_MIN_COOCCURRENCE,
        help=f"Min co-occurrence count to keep an edge (default: {DEFAULT_MIN_COOCCURRENCE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without modifying files"
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore original build_noun_graph.py from backup"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        target = find_target_file()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.restore:
        ok = restore(target)
        sys.exit(0 if ok else 1)

    if is_already_patched(target):
        if args.dry_run:
            print(f"Already patched: {target}")
            return
        logger.info("Re-applying patch with new parameters...")

    ok = apply_patch(target, args.max_k, args.min_cooccurrence, args.dry_run)
    if ok and not args.dry_run:
        print(f"✅ Patched: {target}")
        print(f"   Top-K = {args.max_k}, Min co-occurrence = {args.min_cooccurrence}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
