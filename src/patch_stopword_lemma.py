"""
GraphRAG Stopword Lemmatization Patch
=====================================
Patches graphrag's noun phrase extractors to use stem-based stopword matching.

Problem:
  The original stopword matching uses exact uppercase string comparison:
    token.text.upper() not in self.exclude_nouns
  This means 'investigation' in exclude_nouns does NOT filter out
  'investigated', 'investigating', 'investigations', etc.

Solution:
  Adds stem-based matching using NLTK SnowballStemmer:
    1. At init time, compute stems for all stopwords
    2. During filtering, stem each token and check against stopword stems
    3. Falls back to exact matching if NLTK is unavailable

Results (80 papers, K=17):
  - 'investigation' also filters: investigated, investigating, investigations
  - 'figure' also filters: figures, figured, figuring (not 'fig.' — add separately)
  - 'result' also filters: results, resulted, resulting

Usage:
  python3 patch_stopword_lemma.py [--dry-run] [--restore]
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import textwrap

logger = logging.getLogger(__name__)


def find_package_dir() -> str:
    """Locate the graphrag package directory."""
    spec = importlib.util.find_spec("graphrag")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "graphrag package not found. Install it first: pip install graphrag"
        )
    return os.path.dirname(spec.origin)


def find_base_file(graphrag_root: str) -> str:
    """Locate base.py in np_extractors."""
    target = os.path.join(
        graphrag_root,
        "index", "operations", "build_noun_graph", "np_extractors", "base.py",
    )
    if not os.path.exists(target):
        raise FileNotFoundError(f"Target file not found: {target}")
    return target


def is_already_patched(filepath: str) -> bool:
    """Check if the file already contains our patch."""
    with open(filepath, "r") as f:
        content = f.read()
    return "_exclude_stems" in content


def generate_patched_init() -> str:
    """Generate the patched __init__ method for BaseNounPhraseExtractor.

    IMPORTANT: Methods must be indented with 4 spaces to stay inside the class.
    """
    # Each line has 4-space indent to remain inside the class body
    return (
        '    def __init__(\n'
        '        self,\n'
        '        model_name: str | None,\n'
        '        exclude_nouns: list[str] | None = None,\n'
        '        max_word_length: int = 15,\n'
        '        word_delimiter: str = " ",\n'
        '    ) -> None:\n'
        '        self.model_name = model_name\n'
        '        self.max_word_length = max_word_length\n'
        '        if exclude_nouns is None:\n'
        '            exclude_nouns = []\n'
        '        self.exclude_nouns = [noun.upper() for noun in exclude_nouns]\n'
        '        self.word_delimiter = word_delimiter\n'
        '\n'
        '        # Stem-based stopword matching (v0.2.0)\n'
        '        self._stemmer = None\n'
        '        self._exclude_stems: set[str] = set()\n'
        '        try:\n'
        '            from nltk.stem import SnowballStemmer\n'
        '            self._stemmer = SnowballStemmer("english")\n'
        '            self._exclude_stems = {\n'
        '                self._stemmer.stem(noun.lower()) for noun in exclude_nouns\n'
        '            }\n'
        '        except ImportError:\n'
        '            pass\n'
        '\n'
        '    def is_excluded_noun(self, text: str) -> bool:\n'
        '        """Check if a word should be excluded, with stem-based matching.\n'
        '\n'
        '        Handles morphological variants: \'investigation\' in exclude_nouns\n'
        '        also matches \'investigated\', \'investigating\', \'investigations\'.\n'
        '        """\n'
        '        if text.upper() in self.exclude_nouns:\n'
        '            return True\n'
        '        if self._stemmer is not None:\n'
        '            stem = self._stemmer.stem(text.lower())\n'
        '            if stem in self._exclude_stems:\n'
        '                return True\n'
        '        return False\n'
    )


def apply_patch_base(filepath: str, dry_run: bool = False) -> bool:
    """Patch base.py to add stem-based stopword matching."""
    with open(filepath, "r") as f:
        content = f.read()

    # Find the __init__ method (indented inside class)
    marker = "    def __init__("
    idx = content.find(marker)
    if idx < 0:
        # Try unindented (already patched but wrong)
        marker = "def __init__("
        idx = content.find(marker)
    if idx < 0:
        logger.error("Cannot find __init__ in %s", filepath)
        return False

    # Find the end of __init__ + is_excluded_noun if present
    # Look for the next method that's at class level (4-space indent)
    remaining = content[idx:]
    lines = remaining.split("\n")
    func_lines = [lines[0]]
    for line in lines[1:]:
        stripped = line.lstrip()
        indent_len = len(line) - len(stripped)
        # Stop at next class-level method (4-space indent def/@)
        if stripped and indent_len == 4 and (
            stripped.startswith("def ") or stripped.startswith("@")
        ) and "is_excluded_noun" not in stripped and "__init__" not in stripped:
            break
        # Stop at module-level definitions
        if stripped and indent_len == 0 and (
            stripped.startswith("def ") or stripped.startswith("class ")
            or stripped.startswith("@") or stripped.startswith("async def ")
        ) and "is_excluded_noun" not in stripped:
            break
        func_lines.append(line)

    old_func = "\n".join(func_lines)

    if dry_run:
        print(f"Would patch: {filepath}")
        print(f"  Old __init__: {len(old_func)} chars")
        print(f"  Adding: is_excluded_noun() method with SnowballStemmer")
        return True

    # Create backup
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        logger.info("Backup created: %s", backup)

    new_func = generate_patched_init()
    new_content = content[:idx] + new_func + "\n" + content[idx + len(old_func):]

    with open(filepath, "w") as f:
        f.write(new_content)

    logger.info("Patched __init__ + is_excluded_noun(): %s", filepath)
    return True


def apply_patch_extractor(graphrag_root: str, dry_run: bool = False) -> bool:
    """Patch syntactic_parsing_extractor.py to use is_excluded_noun()."""
    target = os.path.join(
        graphrag_root,
        "index", "operations", "build_noun_graph", "np_extractors",
        "syntactic_parsing_extractor.py",
    )
    if not os.path.exists(target):
        logger.warning("syntactic_parsing_extractor.py not found: %s", target)
        return False

    with open(target, "r") as f:
        content = f.read()

    old_pattern = "and token.text.upper() not in self.exclude_nouns"
    new_pattern = "and not self.is_excluded_noun(token.text)"

    if new_pattern in content:
        logger.info("Already patched: %s", target)
        return True

    if old_pattern not in content:
        logger.error("Cannot find stopword filter pattern in %s", target)
        return False

    if dry_run:
        print(f"Would patch: {target}")
        print(f"  Replace: {old_pattern}")
        print(f"  With:    {new_pattern}")
        return True

    # Create backup
    backup = target + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        logger.info("Backup created: %s", backup)

    new_content = content.replace(old_pattern, new_pattern)

    with open(target, "w") as f:
        f.write(new_content)

    logger.info("Patched stopword filter: %s", target)
    return True


def apply_patch_cfg_extractor(graphrag_root: str, dry_run: bool = False) -> bool:
    """Patch cfg_extractor.py to use is_excluded_noun() if applicable."""
    target = os.path.join(
        graphrag_root,
        "index", "operations", "build_noun_graph", "np_extractors",
        "cfg_extractor.py",
    )
    if not os.path.exists(target):
        return True  # Not all installations have this

    with open(target, "r") as f:
        content = f.read()

    old_pattern = "and token.text.upper() not in self.exclude_nouns"
    new_pattern = "and not self.is_excluded_noun(token.text)"

    if new_pattern in content:
        return True
    if old_pattern not in content:
        return True  # Different pattern, skip

    if dry_run:
        print(f"Would patch: {target}")
        return True

    backup = target + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(target, backup)

    new_content = content.replace(old_pattern, new_pattern)
    with open(target, "w") as f:
        f.write(new_content)

    logger.info("Patched CFG extractor: %s", target)
    return True


def restore(graphrag_root: str) -> bool:
    """Restore all patched files from backups."""
    files = [
        os.path.join(graphrag_root, "index", "operations", "build_noun_graph",
                     "np_extractors", "base.py"),
        os.path.join(graphrag_root, "index", "operations", "build_noun_graph",
                     "np_extractors", "syntactic_parsing_extractor.py"),
        os.path.join(graphrag_root, "index", "operations", "build_noun_graph",
                     "np_extractors", "cfg_extractor.py"),
    ]
    ok = True
    for f in files:
        backup = f + ".bak"
        if os.path.exists(backup):
            shutil.copy2(backup, f)
            logger.info("Restored: %s", f)
        else:
            if os.path.exists(f):
                logger.info("No backup for: %s (skipped)", f)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Patch GraphRAG extractors for stem-based stopword matching"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without modifying files"
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore original files from backup"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        graphrag_root = find_package_dir()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.restore:
        ok = restore(graphrag_root)
        sys.exit(0 if ok else 1)

    base_file = find_base_file(graphrag_root)

    if is_already_patched(base_file):
        if args.dry_run:
            print(f"Already patched: {base_file}")
            return
        logger.info("Re-applying patch...")

    ok1 = apply_patch_base(base_file, args.dry_run)
    ok2 = apply_patch_extractor(graphrag_root, args.dry_run)
    ok3 = apply_patch_cfg_extractor(graphrag_root, args.dry_run)

    ok = ok1 and ok2 and ok3
    if ok and not args.dry_run:
        print("✅ Stopword lemmatization patch applied")
        print("   Morphological variants of exclude_nouns will now be filtered")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
