"""Per-test coverage helpers built on GZoltar's ``sfl/sfl/txt/`` outputs.

The :mod:`src.evaluation` package and :func:`src.extraction.faults.save_first_fault_lines`
both need to answer: *which methods does test T cover?* This module centralises
the parsing so neither caller has to re-derive the matrix/spectra mapping.

Inputs (per bug, under ``processed_dir``):

* ``trigger_tests`` — D4J raw failing-tests file. One or more chunks separated
  by ``--- <pkg.Class>::<method>`` headers. The **order** of chunks is the
  trigger order — the first chunk is what FlexFL's LR planner sees.
* ``sfl/sfl/txt/tests.csv`` — ``name,outcome,runtime,stacktrace``. ``name``
  uses ``pkg.Class#method`` (note ``#`` not ``::``).
* ``sfl/sfl/txt/matrix.txt`` — space-separated; one row per test (same order
  as ``tests.csv``), N+1 tokens per row: N statement-coverage bits then a
  trailing ``+`` (pass) / ``-`` (fail) outcome marker.
* ``sfl/sfl/txt/spectra.csv`` — header ``name``; row i (0-based, excluding
  header) is column i in the matrix; format ``pkg$Class#method(qualifiedParams):line``.

The spectra parameter list is fully qualified (``java.lang.String``) while
corpus identities use simple names (``String``); we sidestep that mismatch
entirely by keying lookups on ``(class_fqn_dotted, line)`` via the
``MethodEntity`` line index from :mod:`src.common.method_entity`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.common.config import (
    MATRIX_FILE,
    SPECTRA_FILE,
    TESTS_FILE,
    coverage_subdir,
)
from src.common.method_entity import (
    MethodEntity,
    _class_fqn_from_corpus_id,
    build_entity_line_index,
)
from src.core.layout import normalize_benchmark_name

logger = logging.getLogger(__name__)


def parse_trigger_test_names(trigger_tests_path: Path) -> list[str]:
    """Return the failing-test names in trigger order, in ``tests.csv`` format.

    Parses every ``--- <pkg.Class>::<method>`` header chunk in the
    ``trigger_tests`` file, converts ``::`` to ``#`` so the names match the
    ``name`` column in ``sfl/sfl/txt/tests.csv``, and preserves their order.

    Returns an empty list if the file does not exist or is empty (callers
    treat that as "no trigger tests" — universe is empty).
    """
    if not trigger_tests_path.exists():
        return []

    names: list[str] = []
    for line in trigger_tests_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("--- "):
            continue
        header = line[4:].strip()
        if "::" not in header:
            continue
        class_fqn, method = header.split("::", 1)
        names.append(f"{class_fqn}#{method}")
    return names


def parse_failing_test_names(failing_tests_path: Path) -> list[str]:
    """Return trigger-test names (``tests.csv`` form) from ``failing_tests.txt``.

    BugsInPy's trigger-name source — one ``<module>$<Class>::<method>`` id per line;
    ``::`` is converted to ``#`` to match the ``name`` column in ``tests.csv``.
    (BugsInPy deprecates the extensionless ``trigger_tests`` file D4J uses.)
    """
    if not failing_tests_path.exists():
        return []

    names: list[str] = []
    for line in failing_tests_path.read_text(encoding="utf-8").splitlines():
        ident = line.strip()
        if not ident or "::" not in ident:
            continue
        class_fqn, method = ident.split("::", 1)
        names.append(f"{class_fqn}#{method}")
    return names


def read_test_row_indices(tests_csv: Path, names: list[str]) -> list[int]:
    """Look up matrix-row indices for each test name.

    Returns 0-based indices into the data rows of ``tests.csv`` (i.e. the
    same row index used in ``matrix.txt``). Raises :class:`KeyError` if any
    requested name is absent — extraction is expected to have produced both
    files in lockstep.
    """
    with tests_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        index: dict[str, int] = {}
        for i, row in enumerate(reader):
            name = row.get("name", "").strip()
            if name and name not in index:
                index[name] = i

    indices: list[int] = []
    for name in names:
        if name not in index:
            raise KeyError(f"test {name!r} not found in {tests_csv}")
        indices.append(index[name])
    return indices


def covered_columns(matrix_path: Path, row_idx: int) -> set[int]:
    """Return the 0-based column indices covered by the test at ``row_idx``.

    Reads exactly one matrix row, splits on whitespace, drops the trailing
    pass/fail outcome marker (``+`` / ``-``), and collects column indices
    where the value is ``"1"``.

    Raises :class:`IndexError` if the matrix has fewer rows than ``row_idx + 1``.
    """
    with matrix_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i != row_idx:
                continue
            tokens = line.split()
            if not tokens:
                return set()
            # Drop the trailing outcome marker. Defensive: if the row is all
            # bits with no outcome marker we still strip the last token.
            tokens = tokens[:-1]
            return {col for col, value in enumerate(tokens) if value == "1"}
    raise IndexError(f"row {row_idx} not found in {matrix_path}")


def _parse_spectra_id(s: str) -> tuple[str, int] | None:
    """Parse ``pkg$Class#method(params):line`` -> ``(dotted_class_fqn, line)``.

    Returns ``None`` (and logs at debug level) when the spectra id doesn't
    parse — keeps the caller-side loop robust against any future GZoltar
    format drift.
    """
    line_sep = s.rfind(":")
    if line_sep < 0:
        return None
    line_str = s[line_sep + 1 :].strip()
    if not line_str.isdigit():
        return None
    line_num = int(line_str)
    prefix = s[:line_sep]
    method_sep = prefix.find("#")
    if method_sep >= 0:
        # GZoltar (Java) shape: ``pkg$Class#method(params):line``.
        class_part = prefix[:method_sep]
        return class_part.replace("$", "."), line_num
    # FauxPy/Python shape: ``<module>$<qualname>(params):line`` (no ``#``). The
    # prefix is a corpus id; recover its statement-owner FQN (class-bearing).
    if "$" not in prefix:
        return None
    return _class_fqn_from_corpus_id(prefix), line_num


def covered_method_entities(
    spectra_path: Path,
    col_indices: set[int],
    entities: list[MethodEntity],
) -> set[MethodEntity]:
    """Map covered statement columns to a deduplicated method-entity set.

    For each spectra row at ``col_indices``, parse the GZoltar statement id,
    look it up in the entity line index (smallest enclosing method wins for
    nested scopes), and accumulate unique entities.

    The spectra file is single-column but unquoted, so its rows can contain
    commas inside parameter lists. We read line-by-line rather than via
    :mod:`csv` to avoid splitting on those commas.
    """
    if not col_indices:
        return set()

    line_index = build_entity_line_index(entities)
    found: set[MethodEntity] = set()
    unmatched = 0

    with spectra_path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            return set()
        for col, raw_line in enumerate(fh):
            if col not in col_indices:
                continue
            stmt_id = raw_line.strip()
            if not stmt_id:
                continue
            parsed = _parse_spectra_id(stmt_id)
            if parsed is None:
                continue
            candidates = line_index.get(parsed)
            if not candidates:
                unmatched += 1
                continue
            chosen = min(candidates, key=lambda e: e.end_line - e.start_line)
            found.add(chosen)

    if unmatched:
        logger.debug(
            "covered_method_entities: %d/%d statements had no matching method entity",
            unmatched,
            len(col_indices),
        )
    return found


def compute_universe(
    processed_dir: Path,
    entities: list[MethodEntity],
    *,
    first_only: bool,
    dataset: str = "defects4j",
) -> set[MethodEntity]:
    """Return the candidate-method universe for a bug.

    Universe = methods covered by at least one of the bug's triggering tests
    (default), or by the *first* triggering test only when ``first_only=True``.

    The coverage location is resolved per benchmark (GZoltar ``sfl/sfl/txt`` for
    D4J; ``FauxPy/coverage`` for BugsInPy) via :func:`config.coverage_subdir`.

    Returns an empty set when ``trigger_tests`` is missing/empty, when the
    coverage files are missing, or (for ``first_only``) when the first test
    covers nothing.
    """
    coverage_dir = processed_dir / coverage_subdir(dataset)
    tests_csv = coverage_dir / TESTS_FILE
    matrix_path = coverage_dir / MATRIX_FILE
    spectra_path = coverage_dir / SPECTRA_FILE

    for required in (tests_csv, matrix_path, spectra_path):
        if not required.exists():
            logger.warning("coverage input missing: %s; universe is empty", required)
            return set()

    # Trigger names: BugsInPy reads failing_tests.txt (the extensionless trigger_tests
    # file is deprecated for BIP); D4J reads its raw trigger_tests dump.
    if normalize_benchmark_name(dataset) == "BIP":
        names = parse_failing_test_names(processed_dir / "failing_tests.txt")
    else:
        names = parse_trigger_test_names(processed_dir / "trigger_tests")
    if not names:
        return set()

    if first_only:
        names = names[:1]

    try:
        row_indices = read_test_row_indices(tests_csv, names)
    except KeyError as exc:
        logger.warning("%s; universe is empty", exc)
        return set()

    columns: set[int] = set()
    for idx in row_indices:
        columns |= covered_columns(matrix_path, idx)

    return covered_method_entities(spectra_path, columns, entities)
