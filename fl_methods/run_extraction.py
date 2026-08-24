"""CLI entry point for the Defects4J data extraction pipeline.

Usage examples::

    # Extract all data for a single bug
    python -m fl_methods.run_extraction -p Chart -v 1

    # Extract multiple bugs
    python -m fl_methods.run_extraction -p Chart -v 1 5 10

    # Extract all bugs for a project
    python -m fl_methods.run_extraction -p Chart

    # Extract only GZoltar data (skip bug report + faults)
    python -m fl_methods.run_extraction -p Chart -v 1 --gzoltar-only

    # Extract a range of versions
    python -m fl_methods.run_extraction -p Chart --start 1 --end 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from update_tracker import compute_coverage

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import get_benchmark_repos_root, get_logs_dir
from src.common.tracker import load_tracker, save_tracker
from src.extraction.bugsinpy import BugsInPyRepo
from src.extraction.d4j import D4JRepo
from src.extraction.pipeline import ensure_bugsinpy_outputs, ensure_d4j_outputs

logger = logging.getLogger(__name__)


def process_bug(
    project: str,
    bug_id: int,
    *,
    dataset: str = "defects4j",
    skip_existing: bool = True,
    gzoltar_only: bool = False,
    bug_report_only: bool = False,
    faults_only: bool = False,
    cleanup_checkouts: bool = True,
    validate_outputs: bool = True,
    repo_factory: Any | None = None,
) -> dict[str, Any]:
    """Dispatch the per-bug extraction pipeline based on ``dataset``."""
    if dataset == "defects4j":
        return process_bug_d4j(
            project,
            bug_id,
            skip_existing=skip_existing,
            gzoltar_only=gzoltar_only,
            bug_report_only=bug_report_only,
            faults_only=faults_only,
            cleanup_checkouts=cleanup_checkouts,
            validate_outputs=validate_outputs,
            repo_factory=repo_factory,
        )
    if dataset == "bugsinpy":
        return process_bug_bugsinpy(
            project,
            bug_id,
            skip_existing=skip_existing,
            gzoltar_only=gzoltar_only,
            bug_report_only=bug_report_only,
            faults_only=faults_only,
            cleanup_checkouts=cleanup_checkouts,
            validate_outputs=validate_outputs,
            repo_factory=repo_factory,
        )
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def process_bug_d4j(
    project: str,
    bug_id: int,
    *,
    skip_existing: bool = True,
    gzoltar_only: bool = False,
    bug_report_only: bool = False,
    faults_only: bool = False,
    cleanup_checkouts: bool = True,
    validate_outputs: bool = True,
    repo_factory: Any | None = None,
) -> dict[str, Any]:
    """Run the full extraction pipeline for a single Defects4J bug.

    Steps:
        1. Checkout buggy version
        2. Compile
        3. Export D4J properties
        4. Run tests → extract test methods
        5. Run GZoltar coverage + FL report
        6. Extract diff-based fault ground truth
        7. Fetch bug report (if available)
    """
    prefix = f"[{project}-{bug_id}]"
    repo = D4JRepo(project, bug_id) if repo_factory is None else repo_factory(project, bug_id)
    issues: list[dict[str, str]] = []

    # Determine which step groups to run from the legacy --*-only flags.
    run_all = not (gzoltar_only or bug_report_only or faults_only)
    run_repo_dependent_steps = run_all or gzoltar_only or faults_only
    cleanup_status = "not-requested"

    if run_all:
        steps: tuple[str, ...] | None = None
    else:
        selected: list[str] = []
        if run_repo_dependent_steps:
            selected.append("repo_setup")
        if gzoltar_only:
            selected.extend(["signatures", "tests", "gzoltar"])
        if faults_only:
            selected.append("faults")
        if bug_report_only:
            selected.append("bug_report")
        steps = tuple(selected)

    try:
        issues = ensure_d4j_outputs(
            repo,
            project=project,
            bug_id=bug_id,
            skip_existing=skip_existing,
            steps=steps,
            validate=validate_outputs,
        )
        for issue in issues:
            level = logging.ERROR if issue["severity"] == "error" else logging.WARNING
            logger.log(
                level,
                "%s Validation %s in %s: %s",
                prefix,
                issue["severity"],
                issue["file"],
                issue["message"],
            )

        # Update coverage in tracker after extraction
        try:
            tracker = load_tracker(project, bug_id)
            compute_coverage(repo.output_dir, tracker)
            save_tracker(tracker, project, bug_id)
        except Exception:
            logger.exception("%s Coverage update failed", prefix)

        logger.info("%s Done.", prefix)
    finally:
        if cleanup_checkouts and run_repo_dependent_steps:
            try:
                repo_existed = repo.repo_dir.exists()
                repo.remove_repo()
                cleanup_status = "removed" if repo_existed else "already-missing"
            except Exception as exc:
                cleanup_status = "failed"
                msg = f"Failed to clean up buggy checkout: {exc}"
                issues.append(
                    {
                        "severity": "error",
                        "file": str(repo.repo_dir),
                        "message": msg,
                    }
                )
                logger.exception("%s %s", prefix, msg)

    return {
        "issues": issues,
        "cleanup_status": cleanup_status,
    }


def process_bug_bugsinpy(
    project: str,
    bug_id: int,
    *,
    skip_existing: bool = True,
    gzoltar_only: bool = False,
    bug_report_only: bool = False,
    faults_only: bool = False,
    cleanup_checkouts: bool = True,
    validate_outputs: bool = True,
    repo_factory: Any | None = None,
) -> dict[str, Any]:
    """Run the per-bug extraction pipeline for a single BugsInPy bug.

    Steps (mirrors the D4J flow via :func:`ensure_bugsinpy_outputs`):
        1. Checkout buggy version + compile (per-bug conda env)
        2. Extract Python method signatures (ast parser)
        3. Write trigger/failing test artifacts
        4. FauxPy coverage + Ochiai ranking (the GZoltar analogue)
        5. Diff-based fault ground truth (from bug_patch.txt)
        6. Fetch bug report (GitHub)

    No tracker coverage update (that step is D4J-specific); cleanup removes the
    root-owned checkout through the container.
    """
    prefix = f"[{project}-{bug_id}]"
    repo = BugsInPyRepo(project, bug_id) if repo_factory is None else repo_factory(project, bug_id)
    issues: list[dict[str, str]] = []

    run_all = not (gzoltar_only or bug_report_only or faults_only)
    run_repo_dependent_steps = run_all or gzoltar_only or faults_only
    cleanup_status = "not-requested"

    if run_all:
        steps: tuple[str, ...] | None = None
    else:
        selected: list[str] = []
        if run_repo_dependent_steps:
            selected.append("repo_setup")
        if gzoltar_only:
            selected.extend(["signatures", "tests", "gzoltar"])
        if faults_only:
            selected.append("faults")
        if bug_report_only:
            selected.append("bug_report")
        steps = tuple(selected)

    try:
        issues = ensure_bugsinpy_outputs(
            repo,
            project=project,
            bug_id=bug_id,
            skip_existing=skip_existing,
            steps=steps,
            validate=validate_outputs,
        )
        for issue in issues:
            level = logging.ERROR if issue["severity"] == "error" else logging.WARNING
            logger.log(
                level,
                "%s Validation %s in %s: %s",
                prefix,
                issue["severity"],
                issue["file"],
                issue["message"],
            )
        logger.info("%s Done.", prefix)
    finally:
        if cleanup_checkouts and run_repo_dependent_steps:
            try:
                repo_existed = repo.repo_dir.exists()
                repo.remove_repo()
                cleanup_status = "removed" if repo_existed else "already-missing"
            except Exception as exc:
                cleanup_status = "failed"
                msg = f"Failed to clean up buggy checkout: {exc}"
                issues.append({"severity": "error", "file": str(repo.repo_dir), "message": msg})
                logger.exception("%s %s", prefix, msg)

    return {
        "issues": issues,
        "cleanup_status": cleanup_status,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Defects4J data extraction pipeline for CEFL",
    )

    # Project/version selection
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
        required=True,
        help="Defects4J project name (e.g. Chart, Math)",
    )
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Version IDs to process (space-separated). Omit to run all versions.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start from this version number (inclusive, filters resolved versions)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End at this version number (inclusive, filters resolved versions)",
    )

    # Pipeline step selection
    parser.add_argument(
        "--gzoltar-only",
        action="store_true",
        help="Only run GZoltar coverage + report (skip faults and bug report)",
    )
    parser.add_argument(
        "--bug-report-only",
        action="store_true",
        help="Only fetch bug reports (skip GZoltar and faults)",
    )
    parser.add_argument(
        "--faults-only",
        action="store_true",
        help="Only extract fault ground truth (skip GZoltar and bug report)",
    )

    # Behaviour
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
        help="Do not remove checked-out repositories after each bug (debug mode)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-extraction output validation checks",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for run logs and JSON summaries (default: logs/<benchmark>/extraction)",
    )

    args = parser.parse_args()

    # --benchmark is authoritative; --dataset is a legacy alias. Honor a legacy
    # --dataset only when --benchmark was left at its default, so old `-d <bench>`
    # invocations keep working. The adapter's benchmark_key is the canonical dispatch
    # key (so e.g. `--benchmark bip` routes to the "bugsinpy" pipeline branch).
    benchmark_key = args.benchmark
    if args.benchmark == "defects4j" and args.dataset != "defects4j":
        benchmark_key = args.dataset
    if benchmark_key not in supported_benchmarks():
        raise NotImplementedError(
            f"Benchmark {benchmark_key!r} not supported. Supported: {supported_benchmarks()}"
        )

    skip_existing = not args.force
    adapter = get_benchmark_adapter(benchmark_key)
    dataset_key = adapter.benchmark_key

    # Resolve versions
    versions = args.versions if args.versions is not None else adapter.list_cases(args.project)
    versions = _filter_range(versions, args.start, args.end)

    projects_versions: dict[str, list[int]] = {args.project: versions}

    # Resolve the log dir against the *parsed* benchmark (an argparse default would be
    # computed at the default benchmark, before --benchmark is known).
    log_dir = args.log_dir or get_logs_dir("extraction", dataset_key)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_scope = _build_run_scope(projects_versions)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"extraction_{run_scope}_{run_stamp}.log"
    report_path = log_dir / f"extraction_{run_scope}_{run_stamp}.json"

    # Configure logging (console + file)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
        force=True,
    )

    logger.info("Run log file: %s", log_path)
    logger.info("Run report file: %s", report_path)

    # Process
    total = sum(len(vs) for vs in projects_versions.values())
    count = 0
    failures: list[str] = []
    run_results: list[dict[str, Any]] = []

    for project, versions in projects_versions.items():
        for bug_id in versions:
            count += 1
            started = time.time()
            run_all = not (args.gzoltar_only or args.bug_report_only or args.faults_only)
            run_repo_dependent_steps = run_all or args.gzoltar_only or args.faults_only
            bug_result: dict[str, Any] = {
                "project": project,
                "bug_id": bug_id,
                "status": "ok",
                "issues": [],
                "error": None,
                "cleanup_status": "not-requested",
                "duration_sec": 0.0,
            }
            logger.info(
                "Processing %s-%d (%d/%d)",
                project,
                bug_id,
                count,
                total,
            )
            try:
                details = process_bug(
                    project,
                    bug_id,
                    dataset=dataset_key,
                    skip_existing=skip_existing,
                    gzoltar_only=args.gzoltar_only,
                    bug_report_only=args.bug_report_only,
                    faults_only=args.faults_only,
                    cleanup_checkouts=not args.keep_checkouts,
                    validate_outputs=not args.no_validate,
                    repo_factory=adapter.build_repo,
                )

                issues = details["issues"]
                bug_result["issues"] = issues
                bug_result["cleanup_status"] = details["cleanup_status"]
                if any(issue["severity"] == "error" for issue in issues):
                    bug_result["status"] = "validation_failed"
                    failures.append(f"{project}-{bug_id}: validation_failed")
            except Exception as exc:
                msg = f"{project}-{bug_id}: {exc}"
                logger.error("Failed: %s", msg)
                logger.debug(traceback.format_exc())
                failures.append(msg)
                bug_result["status"] = "failed"
                bug_result["error"] = str(exc)
            finally:
                if (
                    bug_result["cleanup_status"] == "not-requested"
                    and not args.keep_checkouts
                    and run_repo_dependent_steps
                ):
                    repo_dir = get_benchmark_repos_root(dataset_key) / project / str(bug_id)
                    bug_result["cleanup_status"] = "removed" if not repo_dir.exists() else "kept"
                bug_result["duration_sec"] = round(time.time() - started, 2)
                run_results.append(bug_result)

    summary = _summarize_results(run_results)

    run_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_scope": run_scope,
        "arguments": {
            "benchmark": args.benchmark,
            "project": args.project,
            "versions": args.versions,
            "start": args.start,
            "end": args.end,
            "gzoltar_only": args.gzoltar_only,
            "bug_report_only": args.bug_report_only,
            "faults_only": args.faults_only,
            "force": args.force,
            "keep_checkouts": args.keep_checkouts,
            "validate_outputs": not args.no_validate,
        },
        "summary": summary,
        "results": run_results,
    }
    report_path.write_text(json.dumps(run_report, indent=2) + "\n")

    # Summary
    logger.info(
        "Completed %d bugs: ok=%d, validation_failed=%d, failed=%d, warnings=%d",
        summary["total"],
        summary["ok"],
        summary["validation_failed"],
        summary["failed"],
        summary["warnings"],
    )

    if failures:
        logger.warning(
            "Completed with %d/%d errors:",
            len(failures),
            total,
        )
        for err in failures:
            logger.warning("  - %s", err)
        sys.exit(1)
    else:
        logger.info("All %d bugs processed successfully.", total)


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


def _build_run_scope(projects_versions: dict[str, list[int]]) -> str:
    if not projects_versions:
        return "empty"
    if len(projects_versions) == 1:
        project = next(iter(projects_versions.keys()))
        return project.lower()
    return "multi_project"


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(results),
        "ok": 0,
        "validation_failed": 0,
        "failed": 0,
        "warnings": 0,
    }
    for result in results:
        status = result["status"]
        if status in summary:
            summary[status] += 1

        issues = result["issues"]
        warnings = sum(1 for issue in issues if issue["severity"] == "warning")
        summary["warnings"] += warnings

    return summary


if __name__ == "__main__":
    main()
