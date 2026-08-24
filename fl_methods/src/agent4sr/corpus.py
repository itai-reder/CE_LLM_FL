"""Corpus generation for Agent4SR via the benchmark-aware source parser.

Builds the two line-aligned corpus files the SR (and LR) retrieval tools read at
runtime, dispatching on ``dataset`` through
:func:`src.common.source_corpus.iter_method_corpus` — the single place the
Java-vs-Python parser and corpus-id construction live. Defects4J parses ``.java``
with the canonical Java corpus-id; BugsInPy parses ``.py`` with the Python
corpus-id (``method_info_to_corpus_id`` is Java-only and mis-splits Python module
paths).

Produces two parallel line-aligned outputs per bug:
  - **method IDs**: ``<pkg-or-module>$<qualname>(<params>)`` (one per line)
  - **raw method code**: one method per physical line, line-aligned with the IDs.
    Defects4J stores the Java source space-flattened (newlines→spaces; fine for a
    brace/semicolon language). BugsInPy stores the **verbatim multi-line Python
    source** ``json.dumps``-encoded to a single physical line — significant
    indentation must survive, and ``load_corpus_codes`` ``json.loads`` it back.

The ``$`` separator divides the package/module path from the in-module qualname.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.common.config import get_sr_dir, get_src_dir
from src.common.java_parser import MethodInfo
from src.common.source_corpus import iter_method_corpus
from src.core.layout import normalize_benchmark_name

logger = logging.getLogger(__name__)


def _is_bugsinpy(dataset: str) -> bool:
    return normalize_benchmark_name(dataset) == "BIP"


def _raw_method_source(m: MethodInfo, file_cache: dict[str, list[str]]) -> str:
    """Return a method's verbatim multi-line source slice (indentation preserved).

    Python's significant indentation makes the parser's space-flattened ``content``
    unusable as a code snippet (it collapses nested blocks onto one line), so for
    BugsInPy the corpus stores the source lines re-read from the checkout. Falls back
    to the flattened ``content`` if the file can't be read. ``file_cache`` memoises the
    per-file line list so each source file is read at most once per corpus run.
    """
    lines = file_cache.get(m.file_path)
    if lines is None:
        try:
            lines = Path(m.file_path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            logger.warning(
                "Could not read %s for raw method source; using flattened content", m.file_path
            )
            lines = []
        file_cache[m.file_path] = lines
    if not lines:
        return m.content
    return "\n".join(lines[m.start_line - 1 : m.end_line])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """In-memory corpus: parallel lists of method IDs and raw code."""

    method_ids: list[str]
    raw_codes: list[str]


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


def generate_corpus(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> Corpus:
    """Generate method-level corpus from the buggy source checkout.

    Iterates over every method under the source directory via
    :func:`iter_method_corpus`, which selects the language-appropriate parser and
    corpus-id for *dataset*. The raw-code line is the method's flattened
    ``content`` (newlines already replaced with spaces).

    Returns a :class:`Corpus` with parallel ``method_ids`` and ``raw_codes``.
    """
    src_dir = get_src_dir(project, bug_id, dataset=dataset)
    is_bip = _is_bugsinpy(dataset)

    method_ids: list[str] = []
    raw_codes: list[str] = []
    file_cache: dict[str, list[str]] = {}
    for m, corpus_id in iter_method_corpus(src_dir, dataset):
        method_ids.append(corpus_id)
        # D4J keeps the parser's flattened content (byte-identical); BugsInPy preserves
        # the real multi-line source so Python indentation is not lost.
        raw_codes.append(_raw_method_source(m, file_cache) if is_bip else m.content)

    logger.info(
        "Corpus for %s-%s (%s): %d methods extracted from %s",
        project,
        bug_id,
        dataset,
        len(method_ids),
        src_dir,
    )
    return Corpus(method_ids=method_ids, raw_codes=raw_codes)


def save_corpus(
    project: str,
    bug_id: str | int,
    *,
    skip_existing: bool = True,
    dataset: str = "defects4j",
) -> Path:
    """Generate and save the corpus files to the FlexFL/SR output directory.

    Creates:
      - ``corpus_methods.txt`` — method IDs (one per line)
      - ``corpus_codes.txt`` — raw method code (one per line)

    Returns the SR output directory path.
    """
    out_dir = get_sr_dir(project, bug_id, dataset=dataset)
    methods_path = out_dir / "corpus_methods.txt"
    codes_path = out_dir / "corpus_codes.txt"

    if skip_existing and methods_path.exists() and codes_path.exists():
        logger.info("Corpus already exists for %s-%s, skipping.", project, bug_id)
        return out_dir

    corpus = generate_corpus(project, bug_id, dataset=dataset)
    methods_path.write_text("\n".join(corpus.method_ids) + "\n", encoding="utf-8")
    # BugsInPy code can be multi-line (Python indentation); JSON-encode each method to a
    # single physical line so the file stays line-aligned with corpus_methods.txt. D4J is
    # already single-line and stays byte-identical.
    if _is_bugsinpy(dataset):
        codes_text = "\n".join(json.dumps(c) for c in corpus.raw_codes)
    else:
        codes_text = "\n".join(corpus.raw_codes)
    codes_path.write_text(codes_text + "\n", encoding="utf-8")

    logger.info("Corpus saved to %s (%d methods)", out_dir, len(corpus.method_ids))
    return out_dir
