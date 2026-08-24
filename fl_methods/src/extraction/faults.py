"""Diff-based fault ground truth extraction.

Compares buggy vs. fixed source files to identify changed lines, producing:

* ``faults.txt`` — one ``pkg.Class lineNum`` per line (legacy flat format).
* ``faults.csv`` — ``path,line,signature`` per row, where ``signature`` is the
  corpus_id of the enclosing method (empty when no method maps to that line).
  This makes downstream coverage computation a single CSV read.
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from src.common.coverage import compute_universe
from src.common.method_entity import (
    MethodEntity,
    build_entity_line_index,
    load_method_entities,
)
from src.core.layout import normalize_benchmark_name
from src.extraction.d4j import D4JRepo

if TYPE_CHECKING:
    from src.extraction.bugsinpy import BugsInPyRepo

logger = logging.getLogger(__name__)

FAULTS_CSV_HEADER = ("path", "line", "signature")
FAULTS_FIRST_TXT = "faults_first.txt"
FAULTS_FIRST_CSV = "faults_first.csv"


def diff_start_lines(file1: Path, file2: Path) -> list[int]:
    """Extract the starting line numbers of changes from ``diff`` output.

    Parameters
    ----------
    file1:
        Path to the buggy source file.
    file2:
        Path to the fixed source file.

    Returns
    -------
    List of 1-based line numbers in the *buggy* file where changes start.
    """
    exist1, exist2 = file1.exists(), file2.exists()

    # Case 0: neither file exists
    if not exist1 and not exist2:
        raise FileNotFoundError(f"Both files do not exist:\n  file1: {file1}\n  file2: {file2}")

    # Case 1: only fixed file exists (new file added in fix → no buggy lines)
    if not exist1:
        logger.warning(
            "Buggy file %s does not exist; "
            "assuming all lines in %s are added (no buggy modifications).",
            file1,
            file2,
        )
        return []

    # Case 2: only buggy file exists (file deleted in fix → no modifications)
    if not exist2:
        logger.warning(
            "Fixed file %s does not exist; assuming all lines in %s are deleted.",
            file2,
            file1,
        )
        return []

    # Case 3: both files exist — run diff
    result = subprocess.run(
        ["diff", str(file1), str(file2)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"diff failed (rc={result.returncode}):\n"
            f"  file1: {file1}\n  file2: {file2}\n"
            f"  stderr: {result.stderr}"
        )

    changed_lines: list[int] = []
    for line in result.stdout.splitlines():
        # Lines like "10c14", "5,7d3", "12a15,18"
        # Extract the left-side (buggy) first line number.
        line_range = line.split("a")[0].split("c")[0].split("d")[0].strip()
        first_line = line_range.split(",")[0].strip()
        if first_line.isdigit():
            changed_lines.append(int(first_line))

    return changed_lines


def save_fault_lines(
    repo: D4JRepo | BugsInPyRepo,
    *,
    skip_existing: bool = True,
    dataset: str = "defects4j",
) -> Path:
    """Identify faulty lines and write ``faults.txt`` plus ``faults.csv``.

    ``faults.txt`` contains one ``pkg.Class lineNum`` per line (legacy format).
    ``faults.csv`` contains one ``path,line,signature`` row per fault, where
    ``signature`` is the corpus_id of the smallest enclosing method (or empty
    when no method maps to that line — a warning is logged for each empty row).

    For BugsInPy the diff comes from the bug's ``bug_patch.txt`` (no fixed
    checkout); for Defects4J the fixed version is checked out and diffed.

    Parameters
    ----------
    repo:
        A *buggy* repo instance (D4JRepo or BugsInPyRepo).
    skip_existing:
        If True, skip when both ``faults.txt`` and ``faults.csv`` exist.

    Returns
    -------
    Path to the written ``faults.txt``.
    """
    if normalize_benchmark_name(dataset) == "BIP":
        return _save_fault_lines_bugsinpy(cast("BugsInPyRepo", repo), skip_existing=skip_existing)

    output_file = repo.output_dir / "faults.txt"
    csv_file = repo.output_dir / "faults.csv"
    if skip_existing and output_file.exists() and csv_file.exists():
        logger.info("%s and %s already exist, skipping fault extraction.", output_file, csv_file)
        return output_file

    modified_classes = repo.get_modified_classes()

    # Load method entities once; absence is non-fatal — faults.csv will still
    # be written but every row's signature column will be empty.
    line_index = _load_line_index_or_warn(repo.output_dir)

    # Checkout the fixed version
    fixed_repo = D4JRepo(repo.project, repo.bug_id, buggy=False)
    fixed_repo.checkout()

    faulty_entries: list[str] = []
    csv_rows: list[tuple[str, int, str]] = []

    try:
        for cls in modified_classes:
            buggy_file = repo.classpath_from_class_signature(cls)
            fixed_file = fixed_repo.classpath_from_class_signature(cls)

            try:
                lines = diff_start_lines(buggy_file, fixed_file)
            except FileNotFoundError:
                logger.warning(
                    "Skipping diff for %s: one or both source files missing.",
                    cls,
                )
                continue

            for line_num in lines:
                faulty_entries.append(f"{cls} {line_num}")
                path, signature = _resolve_fault_location(cls, line_num, line_index)
                csv_rows.append((path, line_num, signature))

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(faulty_entries) + "\n" if faulty_entries else "")
        _write_faults_csv(csv_file, csv_rows)

        logger.info(
            "Wrote %d fault entries to %s and %s",
            len(faulty_entries),
            output_file,
            csv_file,
        )
    finally:
        # Always clean up fixed checkout, even if diffing fails.
        fixed_repo.remove_repo()

    return output_file


def _load_line_index_or_warn(processed_dir: Path) -> dict[tuple[str, int], list[MethodEntity]]:
    """Load method_signatures.csv into a (class, line) → entities index.

    Returns an empty dict (and logs a warning) when the signatures CSV is
    missing — in that case every faults.csv row will have an empty signature.
    """
    try:
        entities = load_method_entities(processed_dir)
    except FileNotFoundError:
        logger.warning(
            "method_signatures.csv not found in %s; faults.csv signatures will be empty.",
            processed_dir,
        )
        return {}
    return build_entity_line_index(entities)


def _resolve_fault_location(
    class_fqn: str,
    line_num: int,
    line_index: dict[tuple[str, int], list[MethodEntity]],
) -> tuple[str, str]:
    """Return ``(path, signature)`` for one fault entry.

    Path comes from the smallest enclosing method's ``MethodEntity.path`` when
    available (matches ``method_signatures.csv`` convention). When no
    method maps to the line, ``path`` is derived from the FQCN and ``signature``
    is empty (with a warning).
    """
    candidates = line_index.get((class_fqn, line_num), [])
    if candidates:
        # Pick the innermost (smallest line range) for nested-method scenarios.
        chosen = min(candidates, key=lambda e: e.end_line - e.start_line)
        return chosen.path, chosen.corpus_id

    logger.warning(
        "No method maps to %s:%d; faults.csv signature left empty.",
        class_fqn,
        line_num,
    )
    fallback_path = class_fqn.replace(".", "/") + ".java"
    return fallback_path, ""


def _write_faults_csv(path: Path, rows: list[tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FAULTS_CSV_HEADER)
        for src_path, line_num, signature in rows:
            writer.writerow([src_path, line_num, signature])


# ---------------------------------------------------------------------------
# BugsInPy fault extraction (diff from bug_patch.txt; no fixed checkout)
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+")
_DIFF_GIT_RE = re.compile(r"diff --git a/(.+?) b/(.+)$")


def _parse_patch_buggy_lines(patch_text: str) -> dict[str, list[int]]:
    """Map each changed ``.py`` file to its buggy-side changed line numbers.

    Parses a unified diff, recording the buggy-file line number of every removed
    (``-``) line. For a pure-insertion hunk (no removals) records the hunk's
    anchor (its buggy start line) so omission bugs still get a fault line.
    """
    changes: dict[str, list[int]] = defaultdict(list)
    current_file: str | None = None
    buggy_lineno = 0
    hunk_anchor = 0
    hunk_had_minus = False

    def flush_hunk() -> None:
        if current_file and not hunk_had_minus and hunk_anchor:
            changes[current_file].append(hunk_anchor)

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush_hunk()
            hunk_had_minus = True  # suppress a stray flush before the first hunk
            match = _DIFF_GIT_RE.match(line.strip())
            current_file = match.group(2).strip() if match else None
            if current_file and not current_file.endswith(".py"):
                current_file = None
        elif line.startswith("@@"):
            flush_hunk()
            match = _HUNK_RE.search(line)
            if match:
                buggy_lineno = int(match.group(1))
                hunk_anchor = buggy_lineno
                hunk_had_minus = False
        elif current_file is None or line.startswith(("---", "+++")):
            continue
        elif line.startswith("-"):
            changes[current_file].append(buggy_lineno)
            buggy_lineno += 1
            hunk_had_minus = True
        elif line.startswith("+"):
            continue  # added line: consumes no buggy line
        else:
            buggy_lineno += 1  # context line
    flush_hunk()
    return dict(changes)


def _build_path_line_index(
    entities: list[MethodEntity],
) -> dict[tuple[str, int], list[MethodEntity]]:
    """Build a ``(src-relative path, line) -> [MethodEntity]`` index."""
    index: dict[tuple[str, int], list[MethodEntity]] = defaultdict(list)
    for e in entities:
        for line in range(e.start_line, e.end_line + 1):
            index[(e.path, line)].append(e)
    return index


def _match_src_path(patch_path: str, known_paths: set[str]) -> str:
    """Resolve a repo-root-relative patch path to a method_signatures src path.

    Tries progressively shorter suffixes so a build-output import root (e.g.
    ansible's ``pythonpath`` ``ansible/build/lib`` vs the source ``lib/`` prefix)
    reconciles with the entity paths. Falls back to the patch path unchanged.
    """
    parts = patch_path.split("/")
    for i in range(len(parts)):
        candidate = "/".join(parts[i:])
        if candidate in known_paths:
            return candidate
    return patch_path


def _save_fault_lines_bugsinpy(repo: BugsInPyRepo, *, skip_existing: bool) -> Path:
    """BugsInPy fault extraction from ``bug_patch.txt`` (no fixed checkout)."""
    output_file = repo.output_dir / "faults.txt"
    csv_file = repo.output_dir / "faults.csv"
    if skip_existing and output_file.exists() and csv_file.exists():
        logger.info("%s and %s already exist, skipping fault extraction.", output_file, csv_file)
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)
    patch_text = repo.get_patch_text()
    if not patch_text:
        logger.warning(
            "No bug_patch.txt for %s-%s; writing empty faults.", repo.project, repo.bug_id
        )
        output_file.write_text("")
        _write_faults_csv(csv_file, [])
        return output_file

    try:
        entities = load_method_entities(repo.output_dir)
    except FileNotFoundError:
        logger.warning(
            "method_signatures.csv not found in %s; faults.csv signatures will be empty.",
            repo.output_dir,
        )
        entities = []
    path_index = _build_path_line_index(entities)
    known_paths = {e.path for e in entities}

    faulty_entries: list[str] = []
    csv_rows: list[tuple[str, int, str]] = []
    for patch_path, lines in _parse_patch_buggy_lines(patch_text).items():
        rel = _match_src_path(patch_path, known_paths)
        module = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
        for line_num in sorted(set(lines)):
            candidates = path_index.get((rel, line_num), [])
            if candidates:
                chosen = min(candidates, key=lambda e: e.end_line - e.start_line)
                faulty_entries.append(f"{chosen.class_fqn_dotted} {line_num}")
                csv_rows.append((chosen.path, line_num, chosen.corpus_id))
            else:
                faulty_entries.append(f"{module} {line_num}")
                csv_rows.append((rel, line_num, ""))

    output_file.write_text("\n".join(faulty_entries) + "\n" if faulty_entries else "")
    _write_faults_csv(csv_file, csv_rows)
    logger.info(
        "Wrote %d fault entries to %s and %s (BugsInPy)",
        len(faulty_entries),
        output_file,
        csv_file,
    )
    return output_file


def save_first_fault_lines(
    repo: D4JRepo | BugsInPyRepo,
    *,
    skip_existing: bool = True,
    dataset: str = "defects4j",
) -> Path:
    """Filter ``faults.{csv,txt}`` to faults reachable from the first triggering test.

    FlexFL's LR planner only sees the first triggering test in its
    initializing prompt. To compare LR fairly under that assumption, this
    filter writes a parallel ground-truth pair restricted to faults whose
    enclosing method is covered by the first triggering test.

    Outputs (sibling to ``faults.{csv,txt}``):

    * ``faults_first.csv`` — same ``path,line,signature`` schema; rows
      retained only when ``signature`` (corpus_id) is in the first-test
      universe. Always written; may be header-only when no fault is reachable.
    * ``faults_first.txt`` — ``pkg.Class lineNum`` rows for the same retained
      faults (mapped back via the ``faults.txt`` line). Empty file when no
      faults are retained.

    Requires that ``save_fault_lines`` has already produced ``faults.csv`` and
    that GZoltar coverage outputs are present.
    """
    output_dir = repo.output_dir
    csv_out = output_dir / FAULTS_FIRST_CSV
    txt_out = output_dir / FAULTS_FIRST_TXT
    if skip_existing and csv_out.exists() and txt_out.exists():
        logger.info("%s and %s already exist, skipping.", csv_out, txt_out)
        return csv_out

    faults_csv = output_dir / "faults.csv"
    faults_txt = output_dir / "faults.txt"
    if not faults_csv.exists():
        raise FileNotFoundError(f"{faults_csv} missing; run save_fault_lines first.")

    try:
        entities = load_method_entities(output_dir)
    except FileNotFoundError:
        logger.warning(
            "method_signatures.csv missing in %s; "
            "writing header-only faults_first.csv and empty faults_first.txt.",
            output_dir,
        )
        _write_faults_csv(csv_out, [])
        txt_out.write_text("", encoding="utf-8")
        return csv_out

    universe = compute_universe(output_dir, entities, first_only=True, dataset=dataset)
    universe_ids = {e.corpus_id for e in universe}

    kept_csv_rows: list[tuple[str, int, str]] = []
    kept_indices: list[int] = []
    with faults_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            sig = (row.get("signature") or "").strip()
            if not sig or sig not in universe_ids:
                continue
            path = (row.get("path") or "").strip()
            line_raw = (row.get("line") or "").strip()
            if not line_raw.isdigit():
                continue
            kept_csv_rows.append((path, int(line_raw), sig))
            kept_indices.append(idx)

    _write_faults_csv(csv_out, kept_csv_rows)

    # faults.txt and faults.csv are written in lockstep by save_fault_lines:
    # csv row i and txt line i describe the same fault. Filter by index.
    kept_txt_lines: list[str] = []
    if faults_txt.exists() and kept_indices:
        keep_set = set(kept_indices)
        txt_lines = [ln for ln in faults_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept_txt_lines = [ln for i, ln in enumerate(txt_lines) if i in keep_set]

    txt_out.write_text(
        "\n".join(kept_txt_lines) + ("\n" if kept_txt_lines else ""),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d first-test fault entries to %s and %s",
        len(kept_csv_rows),
        csv_out,
        txt_out,
    )
    return csv_out
