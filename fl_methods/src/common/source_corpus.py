"""Benchmark-aware source parsing for the FL corpora.

The traditional FL methods build two corpora from a checked-out source tree:

* a **statement** corpus (Blues / SBIR stage 2), and
* a **method** corpus keyed on the canonical corpus identity (BoostN).

Both are language-specific: Defects4J parses Java, BugsInPy parses Python. The
parser choice — and, for methods, the corpus-id construction — is the only
language-dependent step. Centralising the dispatch here keeps Blues/BoostN (and
future benchmarks) plugging in once, and mirrors the extraction dispatch in
:func:`src.extraction.gzoltar.run_corpus_method_extraction`.

The ``MethodInfo``/``StatementInfo`` shapes are identical across languages (the
Python parser imports them from :mod:`src.common.java_parser`); only the *values*
differ (Python ``param_types`` holds parameter names; ``class_fqn`` is the owner
module or class).
"""

from __future__ import annotations

from pathlib import Path

from src.common.java_parser import (
    MethodInfo,
    StatementInfo,
    extract_methods_from_java,
    extract_statements_from_java,
    find_java_files,
)
from src.common.method_entity import (
    method_entity_from_python_method_info,
    method_info_to_corpus_id,
)
from src.common.python_parser import (
    extract_methods_from_python,
    extract_statements_from_python,
    find_python_files,
    module_path_for_python,
)
from src.core.layout import normalize_benchmark_name


def _is_bugsinpy(dataset: str) -> bool:
    return normalize_benchmark_name(dataset) == "BIP"


def iter_statement_corpus(src_dir: Path, dataset: str) -> list[StatementInfo]:
    """Return all source statements under *src_dir* for *dataset*.

    Each ``StatementInfo`` carries the class-bearing ``stmt_id`` (``<owner>#<line>``)
    that the SBFL→method aggregation joins on. Tests are excluded.
    """
    statements: list[StatementInfo] = []
    if _is_bugsinpy(dataset):
        for py_file in find_python_files(src_dir, exclude_tests=True):
            statements.extend(extract_statements_from_python(py_file, source_root=src_dir))
    else:
        for java_file in find_java_files(src_dir, exclude_tests=True):
            statements.extend(extract_statements_from_java(java_file, source_root=src_dir))
    return statements


def iter_method_corpus(src_dir: Path, dataset: str) -> list[tuple[MethodInfo, str]]:
    """Return ``(method_info, corpus_id)`` pairs for every method under *src_dir*.

    The corpus id is the canonical ``<pkg-or-module>$<qualname>(<params>)`` shape.
    For Python the Java ``method_info_to_corpus_id`` cannot be reused (it mis-splits
    multi-segment module paths / nested classes), so the id is built via
    :func:`method_entity_from_python_method_info`, which needs the module string.
    """
    pairs: list[tuple[MethodInfo, str]] = []
    if _is_bugsinpy(dataset):
        for py_file in find_python_files(src_dir, exclude_tests=True):
            module = module_path_for_python(py_file, src_dir)
            for m in extract_methods_from_python(py_file, module=module, source_root=src_dir):
                corpus_id = method_entity_from_python_method_info(
                    m, module, src_root=src_dir
                ).corpus_id
                pairs.append((m, corpus_id))
    else:
        for java_file in find_java_files(src_dir, exclude_tests=True):
            for m in extract_methods_from_java(java_file):
                pairs.append((m, method_info_to_corpus_id(m)))
    return pairs
