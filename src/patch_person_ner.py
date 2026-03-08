"""
GraphRAG PERSON NER Enhancement Patch
======================================
Patches graphrag's build_noun_graph.py to add supplementary PERSON entity
extraction using spaCy's en_core_web_sm NER model.

Problem:
  In Lazy/NLP mode, person names are poorly extracted because:
  1. scispaCy (en_core_sci_lg) is trained on biomedical text with limited
     PERSON NER capability
  2. The frequency-based Top-K filter may exclude low-frequency person names

Solution:
  1. After standard NLP entity extraction, run a supplementary PERSON NER pass
     using en_core_web_sm (general English model with good PERSON NER)
  2. Person entities bypass Top-K filtering as "priority entities"
  3. This ensures researcher names are always included in the graph

Note:
  This patch REPLACES the entire build_noun_graph.py with an integrated version
  that includes Top-K (v0.1.0), co-occurrence filter (v0.1.0), and PERSON NER
  (v0.2.0). It subsumes patch_noun_graph.py — run this one alone for v0.2.0.

Usage:
  python3 patch_person_ner.py [--max-k 17] [--min-cooccurrence 2] [--dry-run]
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

DEFAULT_MAX_K = 17
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
    """Check if the file already contains our PERSON NER patch."""
    with open(filepath, "r") as f:
        content = f.read()
    return "_extract_person_entities" in content


def generate_complete_file(max_k: int, min_cooccurrence: int) -> str:
    """Generate the complete patched build_noun_graph.py."""
    # Use raw string concatenation to avoid f-string brace issues
    return (
        '# Copyright (c) 2024 Microsoft Corporation.\n'
        '# Licensed under the MIT License\n'
        f'# Patched by graphrag-hybrid-installer v0.2.0\n'
        f'# - Top-K entity limit (K={max_k})\n'
        f'# - Min co-occurrence filter (>={min_cooccurrence})\n'
        '# - PERSON NER enhancement (en_core_web_sm)\n'
        '\n'
        '"""Graph extraction using NLP."""\n'
        '\n'
        'import logging\n'
        'from collections import defaultdict\n'
        'from itertools import combinations\n'
        '\n'
        'import pandas as pd\n'
        'from graphrag_cache import Cache\n'
        'from graphrag_storage.tables.table import Table\n'
        '\n'
        'from graphrag.graphs.edge_weights import calculate_pmi_edge_weights\n'
        'from graphrag.index.operations.build_noun_graph.np_extractors.base import (\n'
        '    BaseNounPhraseExtractor,\n'
        ')\n'
        'from graphrag.index.utils.hashing import gen_sha512_hash\n'
        '\n'
        'logger = logging.getLogger(__name__)\n'
        '\n'
        '# PERSON NER model (loaded lazily)\n'
        '_person_nlp = None\n'
        '\n'
        '\n'
        'def _get_person_nlp():\n'
        '    """Lazily load en_core_web_sm for PERSON NER."""\n'
        '    global _person_nlp\n'
        '    if _person_nlp is None:\n'
        '        try:\n'
        '            import spacy\n'
        '            _person_nlp = spacy.load("en_core_web_sm", enable=["ner"])\n'
        '            logger.info("PERSON NER model loaded: en_core_web_sm")\n'
        '        except OSError:\n'
        '            logger.warning(\n'
        '                "en_core_web_sm not available — PERSON NER disabled. "\n'
        '                "Install with: python -m spacy download en_core_web_sm"\n'
        '            )\n'
        '            _person_nlp = False  # Mark as unavailable\n'
        '    return _person_nlp if _person_nlp is not False else None\n'
        '\n'
        '\n'
        'async def build_noun_graph(\n'
        '    text_unit_table: Table,\n'
        '    text_analyzer: BaseNounPhraseExtractor,\n'
        '    normalize_edge_weights: bool,\n'
        '    cache: Cache,\n'
        ') -> tuple[pd.DataFrame, pd.DataFrame]:\n'
        '    """Build a noun graph from text units."""\n'
        '    title_to_ids = await _extract_nodes(\n'
        '        text_unit_table,\n'
        '        text_analyzer,\n'
        '        cache=cache,\n'
        '    )\n'
        '\n'
        '    # PERSON NER enhancement (v0.2.0)\n'
        '    person_title_to_ids, person_entities = await _extract_person_entities(\n'
        '        text_unit_table, cache,\n'
        '    )\n'
        '    for name, tu_ids in person_title_to_ids.items():\n'
        '        if name not in title_to_ids:\n'
        '            title_to_ids[name] = tu_ids\n'
        '        else:\n'
        '            title_to_ids[name] = list(set(title_to_ids[name] + tu_ids))\n'
        '\n'
        '    nodes_df = pd.DataFrame(\n'
        '        [\n'
        '            {\n'
        '                "title": title,\n'
        '                "frequency": len(ids),\n'
        '                "text_unit_ids": ids,\n'
        '            }\n'
        '            for title, ids in title_to_ids.items()\n'
        '        ],\n'
        '        columns=["title", "frequency", "text_unit_ids"],\n'
        '    )\n'
        '\n'
        '    edges_df = _extract_edges(\n'
        '        title_to_ids,\n'
        '        nodes_df=nodes_df,\n'
        '        normalize_edge_weights=normalize_edge_weights,\n'
        '        priority_entities=person_entities,\n'
        '    )\n'
        '    return (nodes_df, edges_df)\n'
        '\n'
        '\n'
        'async def _extract_nodes(\n'
        '    text_unit_table: Table,\n'
        '    text_analyzer: BaseNounPhraseExtractor,\n'
        '    cache: Cache,\n'
        ') -> dict[str, list[str]]:\n'
        '    """Extract noun-phrase nodes from text units.\n'
        '\n'
        '    NLP extraction is CPU-bound (spaCy/TextBlob), so threading\n'
        '    provides no benefit under the GIL. We process rows\n'
        '    sequentially, relying on the cache to skip repeated work.\n'
        '\n'
        '    Returns a mapping of noun-phrase title to text-unit ids.\n'
        '    """\n'
        '    extraction_cache = cache.child("extract_noun_phrases")\n'
        '    total = await text_unit_table.length()\n'
        '    title_to_ids: dict[str, list[str]] = defaultdict(list)\n'
        '    completed = 0\n'
        '\n'
        '    async for row in text_unit_table:\n'
        '        text_unit_id = row["id"]\n'
        '        text = row["text"]\n'
        '\n'
        '        attrs = {"text": text, "analyzer": str(text_analyzer)}\n'
        '        key = gen_sha512_hash(attrs, attrs.keys())\n'
        '        result = await extraction_cache.get(key)\n'
        '        if not result:\n'
        '            result = text_analyzer.extract(text)\n'
        '            await extraction_cache.set(key, result)\n'
        '\n'
        '        for phrase in result:\n'
        '            title_to_ids[phrase].append(text_unit_id)\n'
        '\n'
        '        completed += 1\n'
        '        if completed % 100 == 0 or completed == total:\n'
        '            logger.info(\n'
        '                "extract noun phrases progress: %d/%d",\n'
        '                completed,\n'
        '                total,\n'
        '            )\n'
        '\n'
        '    return dict(title_to_ids)\n'
        '\n'
        '\n'
        'async def _extract_person_entities(\n'
        '    text_unit_table: Table,\n'
        '    cache: Cache,\n'
        ') -> tuple[dict[str, list[str]], set[str]]:\n'
        '    """Extract PERSON entities using en_core_web_sm NER.\n'
        '\n'
        '    Runs a supplementary NER pass specifically for person name extraction.\n'
        '    Returns (title_to_ids, person_entity_set) where person_entity_set\n'
        '    contains uppercase person names for priority handling in edge extraction.\n'
        '    """\n'
        '    nlp = _get_person_nlp()\n'
        '    if nlp is None:\n'
        '        return {}, set()\n'
        '\n'
        '    person_cache = cache.child("extract_person_entities")\n'
        '    title_to_ids: dict[str, list[str]] = defaultdict(list)\n'
        '    person_names: set[str] = set()\n'
        '    total = await text_unit_table.length()\n'
        '    completed = 0\n'
        '\n'
        '    async for row in text_unit_table:\n'
        '        text_unit_id = row["id"]\n'
        '        text = row["text"]\n'
        '\n'
        '        attrs = {"text": text, "analyzer": "person_ner_v1"}\n'
        '        key = gen_sha512_hash(attrs, attrs.keys())\n'
        '        result = await person_cache.get(key)\n'
        '\n'
        '        if not result:\n'
        '            doc = nlp(text[:500000])\n'
        '            result = []\n'
        '            for ent in doc.ents:\n'
        '                if ent.label_ == "PERSON":\n'
        '                    name = ent.text.strip()\n'
        '                    if len(name) >= 2 and not name.isdigit():\n'
        '                        result.append(name.upper())\n'
        '            await person_cache.set(key, result)\n'
        '\n'
        '        for name in result:\n'
        '            title_to_ids[name].append(text_unit_id)\n'
        '            person_names.add(name)\n'
        '\n'
        '        completed += 1\n'
        '\n'
        '    if person_names:\n'
        '        logger.info(\n'
        '            "PERSON NER: %d unique persons from %d text units",\n'
        '            len(person_names),\n'
        '            total,\n'
        '        )\n'
        '\n'
        '    return dict(title_to_ids), person_names\n'
        '\n'
        '\n'
        'def _extract_edges(\n'
        '    title_to_ids: dict[str, list[str]],\n'
        '    nodes_df: pd.DataFrame,\n'
        '    normalize_edge_weights: bool = True,\n'
        f'    max_entities_per_chunk: int = {max_k},\n'
        f'    min_co_occurrence: int = {min_cooccurrence},\n'
        '    priority_entities: set[str] | None = None,\n'
        ') -> pd.DataFrame:\n'
        '    """Build co-occurrence edges between noun phrases.\n'
        '\n'
        '    Nodes that appear in the same text unit are connected.\n'
        '    Filters applied:\n'
        '      1. Top-K: Only the K most frequent entities per chunk are paired.\n'
        '         Priority entities (e.g., PERSON names) bypass this filter.\n'
        '      2. Min co-occurrence: Edges must appear in >=M text units.\n'
        '\n'
        '    Returns edges with schema [source, target, weight, text_unit_ids].\n'
        '    """\n'
        '    if not title_to_ids:\n'
        '        return pd.DataFrame(\n'
        '            columns=["source", "target", "weight", "text_unit_ids"],\n'
        '        )\n'
        '\n'
        '    if priority_entities is None:\n'
        '        priority_entities = set()\n'
        '\n'
        '    entity_freq: dict[str, int] = {\n'
        '        t: len(ids) for t, ids in title_to_ids.items()\n'
        '    }\n'
        '\n'
        '    text_unit_to_titles: dict[str, list[str]] = defaultdict(list)\n'
        '    for title, tu_ids in title_to_ids.items():\n'
        '        for tu_id in tu_ids:\n'
        '            text_unit_to_titles[tu_id].append(title)\n'
        '\n'
        '    edge_map: dict[tuple[str, str], list[str]] = defaultdict(list)\n'
        '    for tu_id, titles in text_unit_to_titles.items():\n'
        '        unique_titles = sorted(set(titles))\n'
        '        if len(unique_titles) < 2:\n'
        '            continue\n'
        '\n'
        '        priority = [t for t in unique_titles if t in priority_entities]\n'
        '        regular = [t for t in unique_titles if t not in priority_entities]\n'
        '\n'
        '        if max_entities_per_chunk > 0 and len(regular) > max_entities_per_chunk:\n'
        '            regular = sorted(\n'
        '                regular,\n'
        '                key=lambda t: entity_freq.get(t, 0),\n'
        '                reverse=True,\n'
        '            )[:max_entities_per_chunk]\n'
        '\n'
        '        selected = sorted(set(priority + regular))\n'
        '        if len(selected) < 2:\n'
        '            continue\n'
        '\n'
        '        for pair in combinations(selected, 2):\n'
        '            edge_map[pair].append(tu_id)\n'
        '\n'
        '    records = [\n'
        '        {\n'
        '            "source": src,\n'
        '            "target": tgt,\n'
        '            "weight": len(tu_ids),\n'
        '            "text_unit_ids": tu_ids,\n'
        '        }\n'
        '        for (src, tgt), tu_ids in edge_map.items()\n'
        '        if min_co_occurrence <= 1 or len(tu_ids) >= min_co_occurrence\n'
        '    ]\n'
        '\n'
        '    priority_edge_count = sum(\n'
        '        1 for (src, tgt) in edge_map\n'
        '        if src in priority_entities or tgt in priority_entities\n'
        '    )\n'
        '    logger.info(\n'
        '        "Edge extraction: %d raw -> %d filtered "\n'
        '        "(top-%d, co-occ>=%d, %d person edges)",\n'
        '        len(edge_map),\n'
        '        len(records),\n'
        '        max_entities_per_chunk,\n'
        '        min_co_occurrence,\n'
        '        priority_edge_count,\n'
        '    )\n'
        '\n'
        '    edges_df = pd.DataFrame(\n'
        '        records,\n'
        '        columns=["source", "target", "weight", "text_unit_ids"],\n'
        '    )\n'
        '\n'
        '    if normalize_edge_weights and not edges_df.empty:\n'
        '        edges_df = calculate_pmi_edge_weights(nodes_df, edges_df)\n'
        '\n'
        '    return edges_df\n'
    )


def apply_patch(filepath: str, max_k: int, min_cooccurrence: int,
                dry_run: bool = False) -> bool:
    """Replace build_noun_graph.py with the integrated patched version."""
    if dry_run:
        print(f"Would replace: {filepath}")
        print(f"  Top-K = {max_k}, Min co-occurrence = {min_cooccurrence}")
        print("  + PERSON NER extraction via en_core_web_sm")
        print("  + Priority entities bypass Top-K filtering")
        return True

    backup = filepath + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        logger.info("Backup created: %s", backup)

    new_content = generate_complete_file(max_k, min_cooccurrence)

    with open(filepath, "w") as f:
        f.write(new_content)

    logger.info(
        "Patched: %s (K=%d, min_cooccurrence=%d, PERSON NER enabled)",
        filepath, max_k, min_cooccurrence,
    )
    return True


def restore(filepath: str) -> bool:
    """Restore the original file from backup."""
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        logger.error("No backup found: %s", filepath)
        return False
    shutil.copy2(backup, filepath)
    logger.info("Restored original: %s", filepath)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch GraphRAG build_noun_graph.py with PERSON NER + Top-K"
    )
    parser.add_argument(
        "--max-k", type=int, default=DEFAULT_MAX_K,
        help=f"Max entities per chunk for Top-K (default: {DEFAULT_MAX_K})"
    )
    parser.add_argument(
        "--min-cooccurrence", type=int, default=DEFAULT_MIN_COOCCURRENCE,
        help=f"Min co-occurrence to keep edge (default: {DEFAULT_MIN_COOCCURRENCE})"
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
        logger.info("Re-applying patch with PERSON NER...")

    ok = apply_patch(target, args.max_k, args.min_cooccurrence, args.dry_run)
    if ok and not args.dry_run:
        print(f"✅ Patched: {target}")
        print(f"   Top-K = {args.max_k}, Min co-occurrence = {args.min_cooccurrence}")
        print("   PERSON NER = enabled (en_core_web_sm)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
