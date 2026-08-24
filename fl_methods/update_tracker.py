"""Static-analysis CLI for populating ``tracker.json`` from on-disk state.

Inspects the outputs already present in
``data/<Benchmark>/processed/<Project>/<BugId>/`` (Defects4J or BugsInPy)
and reconstructs a ``tracker.json`` describing which pipeline steps have
completed, plus coverage counts.

This is a *backfill* tool: it never runs container/pipeline work.  It reads
existing files, infers completion from their presence and content, and may
repair derived local artifacts such as ``failing_tests.txt`` from raw
``failing_tests``.

Usage examples::

    # Single bug
    python -m fl_methods.update_tracker -p Chart -v 1

    # Range of bugs
    python -m fl_methods.update_tracker -p Chart --start 1 --end 10

    # All bugs for a project
    python -m fl_methods.update_tracker -p Chart

    # All projects
    python -m fl_methods.update_tracker --all-projects

    # Dry-run (print tracker changes; may still repair derived local artifacts)
    python -m fl_methods.update_tracker -p Chart -v 1 --dry-run

    # No backup (overwrite existing tracker.json without renaming)
    python -m fl_methods.update_tracker -p Chart -v 1 --no-backup
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import (
    BOOSTN_CSV,
    BOOSTN_JSON,
    BOOSTN_SUBDIR,
    FLEXFL_SR_SUBDIR,
    MATRIX_FILE,
    OCHIAI_RANKING_FILE,
    OCHIAI_SUBDIR,
    SBFL_JSON,
    SBFL_STMT_SUSPS,
    SBIR_JSON,
    SBIR_STMT_SUSPS,
    SBIR_SUBDIR,
    SPECTRA_FILE,
    TESTS_FILE,
    coverage_subdir,
    get_benchmark_processed_root,
    ranking_subdir,
)
from src.common.tracker import (
    TRACKER_FILENAME,
    Tracker,
    _empty_tracker,
    get_or_assign_sr_model_id,
    mark_completed,
    record_error,
    record_warning,
    save_tracker,
    update_coverage,
)
from src.core.layout import normalize_benchmark_name
from src.extraction.d4j import EXPORT_PROPERTIES, write_parsed_failing_tests
from src.extraction.validation import validate_extraction_outputs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FILE_TO_STEP: maps a validation issue's ``file`` to a tracker step.
# Used to route issues from ``validate_extraction_outputs`` into the right
# tracker step's warnings/errors bucket.
# ---------------------------------------------------------------------------

_FILE_TO_STEP: dict[str, str] = {}

# Properties → "properties"
for _prop in EXPORT_PROPERTIES:
    _FILE_TO_STEP[_prop] = "properties"

# Method signatures → "signatures" (BIP validation reports on this file)
_FILE_TO_STEP["method_signatures.csv"] = "signatures"

# Test enumeration → "relevant_tests"
for _tf in ("all_tests.txt", "relevant_tests.txt", "junit_tests.txt"):
    _FILE_TO_STEP[_tf] = "relevant_tests"

# Failing tests → "failing_tests"
for _ft in ("failing_tests", "failing_tests.txt"):
    _FILE_TO_STEP[_ft] = "failing_tests"

# GZoltar outputs → "gzoltar"
for _gf in (SPECTRA_FILE, MATRIX_FILE, TESTS_FILE, OCHIAI_RANKING_FILE):
    _FILE_TO_STEP[_gf] = "gzoltar"

# Faults → "faults"
_FILE_TO_STEP["faults.txt"] = "faults"
_FILE_TO_STEP["faults.csv"] = "faults"

# First-test-filtered faults → "faults_first"
_FILE_TO_STEP["faults_first.txt"] = "faults_first"
_FILE_TO_STEP["faults_first.csv"] = "faults_first"

# Bug report → "bug_report"
_FILE_TO_STEP["bug_report.json"] = "bug_report"

# Cleaned trigger test → "trigger_test_processed"
_FILE_TO_STEP["trigger_test_clean.txt"] = "trigger_test_processed"


def file_to_step(filename: str) -> str:
    """Resolve a validation-issue filename to a tracker step name."""
    return _FILE_TO_STEP.get(filename, "unknown")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _non_empty_file(path: Path) -> bool:
    """Return True if *path* exists and has non-whitespace content."""
    if not path.exists():
        return False
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False


def _infer_extraction(
    processed_dir: Path,
    tracker: Tracker,
    *,
    dataset: str = "defects4j",
) -> None:
    """Infer extraction step completion and route validation issues."""
    is_d4j = normalize_benchmark_name(dataset) == "D4J"

    # properties (D4J-only tier-1: `defects4j export` files; BIP writes none)
    if is_d4j and all(_non_empty_file(processed_dir / p) for p in EXPORT_PROPERTIES):
        mark_completed(tracker, "extraction", "properties")

    # signatures
    corpus_csv = processed_dir / "method_signatures.csv"
    if corpus_csv.exists():
        try:
            header = corpus_csv.read_text(encoding="utf-8").splitlines()[0]
            if "corpus_id" in header:
                mark_completed(tracker, "extraction", "signatures")
        except Exception:
            pass

    # relevant_tests
    relevant = processed_dir / "relevant_tests.txt"
    if _non_empty_file(relevant):
        mark_completed(tracker, "extraction", "relevant_tests")

    # failing_tests (D4J repairs the parsed file from the raw dump; BIP has
    # no raw `failing_tests` — extraction writes failing_tests.txt directly)
    failing = processed_dir / "failing_tests.txt"
    if is_d4j:
        raw_failing = processed_dir / "failing_tests"
        if raw_failing.exists():
            if not _non_empty_file(failing):
                write_parsed_failing_tests(processed_dir)
            if _non_empty_file(failing):
                mark_completed(tracker, "extraction", "failing_tests")
    elif _non_empty_file(failing):
        mark_completed(tracker, "extraction", "failing_tests")

    # coverage (GZoltar sfl/sfl/txt for D4J; FauxPy coverage/reports for BIP)
    cov_dir = processed_dir / coverage_subdir(dataset)
    rank_dir = processed_dir / ranking_subdir(dataset)
    coverage_files = [
        cov_dir / SPECTRA_FILE,
        cov_dir / MATRIX_FILE,
        cov_dir / TESTS_FILE,
        rank_dir / OCHIAI_RANKING_FILE,
    ]
    if all(_non_empty_file(f) for f in coverage_files):
        mark_completed(tracker, "extraction", "gzoltar")

    # faults
    faults_txt = processed_dir / "faults.txt"
    faults_csv = processed_dir / "faults.csv"
    if faults_txt.exists() and faults_csv.exists():
        # Verify faults.txt parses
        txt_ok = True
        for line in faults_txt.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2 or not parts[1].isdigit():
                txt_ok = False
                break
        if txt_ok:
            mark_completed(tracker, "extraction", "faults")

    # faults_first (header-only csv counts as complete; this step is just a filter)
    faults_first_csv = processed_dir / "faults_first.csv"
    faults_first_txt = processed_dir / "faults_first.txt"
    if faults_first_csv.exists() and faults_first_txt.exists():
        mark_completed(tracker, "extraction", "faults_first")

    # bug_report
    br_path = processed_dir / "bug_report.json"
    if br_path.exists():
        try:
            data = json.loads(br_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "error" not in data:
                title = data.get("title", "")
                desc = data.get("description", "")
                if (
                    isinstance(title, str)
                    and title.strip()
                    and isinstance(desc, str)
                    and desc.strip()
                ):
                    mark_completed(tracker, "extraction", "bug_report")
        except json.JSONDecodeError:
            pass

    # trigger_test_processed (BIP: a *blank* trigger_test_clean.txt is a valid
    # outcome — non-mirrorable failure — so existence means the step ran)
    trigger_clean = processed_dir / "trigger_test_clean.txt"
    trigger_ok = _non_empty_file(trigger_clean) if is_d4j else trigger_clean.exists()
    if trigger_ok:
        mark_completed(tracker, "extraction", "trigger_test_processed")

    # Route validation issues
    issues = validate_extraction_outputs(
        processed_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
        dataset=dataset,
    )
    for issue in issues:
        step = file_to_step(issue["file"])
        if issue["severity"] == "error":
            record_error(tracker, "extraction", step, issue["message"])
        else:
            record_warning(tracker, "extraction", step, issue["message"])


def _infer_fl(processed_dir: Path, tracker: Tracker) -> None:
    """Infer FL step completion."""
    # ochiai
    ochiai_dir = processed_dir / OCHIAI_SUBDIR
    if _non_empty_file(ochiai_dir / SBFL_STMT_SUSPS) and _non_empty_file(ochiai_dir / SBFL_JSON):
        mark_completed(tracker, "fl", "ochiai")

    # boostn
    boostn_dir = processed_dir / BOOSTN_SUBDIR
    if _non_empty_file(boostn_dir / BOOSTN_CSV) and _non_empty_file(boostn_dir / BOOSTN_JSON):
        mark_completed(tracker, "fl", "boostn")

    # sbir
    sbir_dir = processed_dir / SBIR_SUBDIR
    if _non_empty_file(sbir_dir / SBIR_STMT_SUSPS) and _non_empty_file(sbir_dir / SBIR_JSON):
        mark_completed(tracker, "fl", "sbir")

    # top15
    rankings_dir = processed_dir / FLEXFL_SR_SUBDIR / "rankings"
    if _non_empty_file(rankings_dir / "top15.txt") and _non_empty_file(rankings_dir / "top15.csv"):
        mark_completed(tracker, "fl", "top15")


def _infer_sr(processed_dir: Path, tracker: Tracker) -> None:
    """Infer SR model entries from Agent4SR result directories."""
    agent_dir = processed_dir / FLEXFL_SR_SUBDIR / "Agent4SR"
    if not agent_dir.is_dir():
        return

    default_input = ["bug_report", "ochiai", "boostn", "sbir"]

    for slug_dir in sorted(agent_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        sr_result = slug_dir / "sr_result.json"
        if not sr_result.exists():
            continue

        try:
            data = json.loads(sr_result.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            record_warning(
                tracker,
                "extraction",
                "unknown",
                f"Could not read {sr_result}",
            )
            continue

        model = data.get("model", "")
        temperature = float(data.get("temperature", 0.0))
        iterations = int(data.get("iterations", 0))
        base_url = data.get("base_url", "")
        input_keys = data.get("input", default_input)

        if not base_url:
            record_warning(
                tracker,
                "fl",
                "top15",
                f"base_url unknown for Agent4SR dir {slug_dir.name}",
            )

        if model:
            get_or_assign_sr_model_id(
                tracker,
                model=model,
                temperature=temperature,
                iterations=iterations,
                base_url=base_url,
                input_keys=input_keys,
            )


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------


def compute_coverage(processed_dir: Path, tracker: Tracker) -> None:
    """Compute and store coverage counts from pre-computed extraction artifacts.

    Reads ``faults.csv``, ``failing_tests.txt``, ``relevant_tests.txt``, and
    the various ranking files to count how many faulty elements appear in each.
    """
    s_faults: dict[str, int] = {}
    m_faults: dict[str, int] = {}
    r_tests: dict[str, int] = {}
    f_tests: dict[str, int] = {}

    # --- Statement faults ---
    faults_csv = processed_dir / "faults.csv"
    stmt_fault_keys: set[tuple[str, int]] = set()
    method_fault_ids: set[str] = set()
    if faults_csv.exists():
        try:
            with faults_csv.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    path = (row.get("path") or "").strip()
                    line_raw = (row.get("line") or "").strip()
                    sig = (row.get("signature") or "").strip()
                    if path and line_raw.isdigit():
                        stmt_fault_keys.add((path, int(line_raw)))
                    if sig:
                        method_fault_ids.add(sig)
        except Exception:
            pass

    s_faults["count"] = len(stmt_fault_keys)
    m_faults["count"] = len(method_fault_ids)

    # --- Per-source statement coverage ---
    # Ochiai stmt ranking
    ochiai_stmt = processed_dir / OCHIAI_SUBDIR / SBFL_STMT_SUSPS
    if ochiai_stmt.exists() and stmt_fault_keys:
        s_faults["ochiai"] = _count_stmt_hits(ochiai_stmt, stmt_fault_keys)

    # SBIR stmt ranking
    sbir_stmt = processed_dir / SBIR_SUBDIR / SBIR_STMT_SUSPS
    if sbir_stmt.exists() and stmt_fault_keys:
        s_faults["sbir"] = _count_stmt_hits(sbir_stmt, stmt_fault_keys)

    # --- Per-source method coverage ---
    rankings_dir = processed_dir / FLEXFL_SR_SUBDIR / "rankings"
    for source, csv_name in [
        ("ochiai", "ochiai.csv"),
        ("sbir", "sbir.csv"),
        ("boostn", "boostn.csv"),
        ("top15", "top15.csv"),
    ]:
        csv_path = rankings_dir / csv_name
        if csv_path.exists() and method_fault_ids:
            m_faults[source] = _count_method_hits_ranking(csv_path, method_fault_ids)

    # Agent4SR model coverage
    for model_id in tracker.get("sr", {}):
        agent_dir = processed_dir / FLEXFL_SR_SUBDIR / "Agent4SR"
        # Find matching dir: try model_id directly, or search slug dirs
        top5_path = agent_dir / model_id / "top5.txt"
        if top5_path.exists() and method_fault_ids:
            m_faults[model_id] = _count_method_hits_top5(top5_path, method_fault_ids)

    # --- Test counts ---
    relevant = processed_dir / "relevant_tests.txt"
    if relevant.exists():
        lines = [
            ln.strip() for ln in relevant.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        r_tests["count"] = len(lines)

    failing = processed_dir / "failing_tests.txt"
    if failing.exists():
        lines = [
            ln.strip() for ln in failing.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        f_tests["count"] = len(lines)

    update_coverage(
        tracker,
        s_faults=s_faults if s_faults else None,
        m_faults=m_faults if m_faults else None,
        r_tests=r_tests if r_tests else None,
        f_tests=f_tests if f_tests else None,
    )


def _count_stmt_hits(stmt_csv: Path, fault_keys: set[tuple[str, int]]) -> int:
    """Count how many fault (path, line) keys appear in a statement ranking.

    Statement rankings use ``Statement,Suspiciousness`` format where the
    Statement column is ``pkg.Class#lineNum``.  Since ``faults.csv`` uses
    file-relative paths while statement rankings use dotted class FQCNs, we
    compare on (class_fqn, line) — the ``path`` in faults.csv is the
    file-relative path, not the class FQN.  The ranking uses class FQN.

    We build a set of (class_fqn, line) from the ranking and intersect with
    the fault keys.
    """
    # faults.csv paths are file-relative, but statement rankings use dotted
    # class FQCNs (e.g. "org.jfree.chart.ChartFactory#42").  We need to
    # build the set of ranking statements and then count how many fault
    # statements match.  The fault_keys here are (path, line) but the
    # ranking keys are (class_fqn, line).  For a proper intersection we
    # need both representations of the faults — but since faults.csv has
    # both the file path and (via statement IDs in the GZoltar output) the
    # class FQN, we simply count how many ranking statements share a line
    # with a faulty line.  In practice, faults.csv path like
    # "org/jfree/chart/ChartFactory.java" maps to class
    # "org.jfree.chart.ChartFactory".
    #
    # Instead of doing the mapping here, we use the simpler approach: read
    # the ranking into a set of (class, line) and count how many fault
    # lines are in that set.  The caller should provide fault keys as
    # (dotted_class_fqn, line) for this to work.
    ranking_stmts: set[tuple[str, int]] = set()
    try:
        with stmt_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return 0
            id_col = "Statement" if "Statement" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                stmt_id = (row.get(id_col) or "").strip()
                if "#" in stmt_id:
                    cls, line_str = stmt_id.rsplit("#", 1)
                    if line_str.isdigit():
                        ranking_stmts.add((cls, int(line_str)))
    except Exception:
        return 0
    return len(fault_keys & ranking_stmts)


def _count_method_hits_ranking(csv_path: Path, fault_corpus_ids: set[str]) -> int:
    """Count faulty corpus IDs present in a ranking CSV.

    Entity ranking CSVs (``ochiai.csv``, ``sbir.csv``, ``boostn.csv``) use
    ``;``-delimited format with a ``signature`` column.  Combined ranking CSVs
    (``top15.csv``) use ``,``-delimited format.  We auto-detect by trying
    semicolon first, falling back to comma.
    """
    found = 0
    try:
        text = csv_path.read_text(encoding="utf-8")
        # Auto-detect: if first line contains semicolons, use that
        first_line = text.split("\n", 1)[0]
        delimiter = ";" if ";" in first_line else ","
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            seen: set[str] = set()
            for row in reader:
                sig = (row.get("signature") or "").strip()
                if sig and sig not in seen:
                    seen.add(sig)
                    if sig in fault_corpus_ids:
                        found += 1
    except Exception:
        pass
    return found


def _count_method_hits_top5(top5_path: Path, fault_corpus_ids: set[str]) -> int:
    """Count faulty corpus IDs in an Agent4SR top5.txt file."""
    found = 0
    seen: set[str] = set()
    for line in top5_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        # top5.txt may use dotted form; also check dollar form
        if raw in fault_corpus_ids or raw.replace(".", "$", 1) in fault_corpus_ids:
            found += 1
    return found


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def update_single_bug(
    processed_dir: Path,
    project: str,
    bug_id: int,
    *,
    dataset: str = "defects4j",
    no_backup: bool = False,
    dry_run: bool = False,
) -> Tracker:
    """Build or refresh tracker.json for a single processed bug directory.

    Returns the constructed tracker (always; even on dry_run).
    """
    tracker = _empty_tracker()

    _infer_extraction(processed_dir, tracker, dataset=dataset)
    _infer_fl(processed_dir, tracker)
    _infer_sr(processed_dir, tracker)
    compute_coverage(processed_dir, tracker)

    if dry_run:
        logger.debug(
            "[%s-%d] dry-run: %s",
            project,
            bug_id,
            json.dumps(tracker, indent=2),
        )
        return tracker

    tracker_file = processed_dir / TRACKER_FILENAME
    if tracker_file.exists() and not no_backup:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = tracker_file.with_name(f"tracker.{stamp}.json")
        tracker_file.rename(backup)
        logger.info("[%s-%d] backed up existing tracker to %s", project, bug_id, backup.name)

    save_tracker(tracker, project, bug_id, dataset=dataset)
    logger.info("[%s-%d] wrote %s", project, bug_id, tracker_file)
    return tracker


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill tracker.json from on-disk pipeline outputs",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="defects4j",
        choices=supported_benchmarks(),
        help="Benchmark adapter key (e.g. defects4j, bugsinpy). Authoritative.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="defects4j",
        help="Deprecated legacy alias for --benchmark; honored only when --benchmark is unset.",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=None,
        help="Project name (e.g. Chart, PySnooper). Required unless --all-projects.",
    )
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Version IDs to process (space-separated). Omit to scan all.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start from this version number (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End at this version number (inclusive)",
    )
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Scan all project directories under the processed root.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Overwrite existing tracker.json without creating a backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tracker JSON to log but do not write to disk.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # --benchmark is authoritative; -d/--dataset is a deprecated alias honored
    # only when --benchmark was left at its default (matches run_evaluation.py).
    benchmark_key = args.benchmark
    if args.benchmark == "defects4j" and args.dataset != "defects4j":
        benchmark_key = args.dataset
    if benchmark_key not in supported_benchmarks():
        parser.error(
            f"Benchmark {benchmark_key!r} not supported. Supported: {supported_benchmarks()}"
        )
    args.dataset = get_benchmark_adapter(benchmark_key).benchmark_key

    if not args.all_projects and not args.project:
        parser.error("Either --project or --all-projects is required.")

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    processed_root = get_benchmark_processed_root(args.dataset)
    if not processed_root.is_dir():
        logger.error("Processed root does not exist: %s", processed_root)
        sys.exit(1)

    # Resolve project(s) and versions
    projects_versions: dict[str, list[int]] = {}

    if args.all_projects:
        for proj_dir in sorted(processed_root.iterdir()):
            if not proj_dir.is_dir() or proj_dir.name.startswith("_"):
                continue
            bug_ids = _discover_bug_ids(proj_dir)
            if bug_ids:
                projects_versions[proj_dir.name] = bug_ids
    else:
        assert args.project is not None
        proj_dir = processed_root / args.project
        if args.versions:
            bug_ids = args.versions
        elif proj_dir.is_dir():
            bug_ids = _discover_bug_ids(proj_dir)
        else:
            adapter = get_benchmark_adapter(args.dataset)
            bug_ids = adapter.list_cases(args.project)
        projects_versions[args.project] = bug_ids

    # Apply range filter
    for proj in projects_versions:
        projects_versions[proj] = _filter_range(projects_versions[proj], args.start, args.end)

    total = sum(len(vs) for vs in projects_versions.values())
    count = 0
    errors = 0

    for project, versions in sorted(projects_versions.items()):
        for bug_id in versions:
            count += 1
            bug_dir = processed_root / project / str(bug_id)
            if not bug_dir.is_dir():
                logger.warning("[%s-%d] processed dir missing, skipping", project, bug_id)
                continue
            try:
                update_single_bug(
                    bug_dir,
                    project,
                    bug_id,
                    dataset=args.dataset,
                    no_backup=args.no_backup,
                    dry_run=args.dry_run,
                )
            except Exception:
                logger.exception("[%s-%d] failed", project, bug_id)
                errors += 1

    logger.info("Done: %d/%d bugs processed, %d errors.", count, total, errors)
    if errors:
        sys.exit(1)


def _discover_bug_ids(proj_dir: Path) -> list[int]:
    """List numeric subdirectory names under a project's processed dir."""
    ids = []
    for sub in proj_dir.iterdir():
        if sub.is_dir() and sub.name.isdigit():
            ids.append(int(sub.name))
    return sorted(ids)


def _filter_range(
    bids: list[int],
    start: int | None,
    end: int | None,
) -> list[int]:
    """Filter bug IDs to the [start, end] range."""
    result = bids
    if start is not None:
        result = [b for b in result if b >= start]
    if end is not None:
        result = [b for b in result if b <= end]
    return result


if __name__ == "__main__":
    main()
