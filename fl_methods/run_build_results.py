#!/usr/bin/env python3
"""Build the slim committed results tree (and transcripts) from processed data.

Distills ``data/<Benchmark>/processed/<Project>/<BugId>/`` into
``results/<Benchmark>/<Project>/<BugId>/`` (see :mod:`src.results.build` for
the exact contents) and copies the full LLM interaction records into the
gitignored ``transcripts/`` tree. Once built, ``run_evaluation.py`` runs from
``results/`` alone — the processed data can be absent entirely.

For BugsInPy the benchmark-level ``results/BIP/_meta/`` snapshot is also
written: the extraction ``audit.csv`` and the ``bugsinpy-index.csv`` project
index, so exclusions reporting and bug enumeration work offline.

Usage::

    python run_build_results.py -p Lang -v 1
    python run_build_results.py -p Lang                    # all bugs with data
    python run_build_results.py --all-projects             # everything on disk
    python run_build_results.py --benchmark bugsinpy --all-projects
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import (
    get_benchmark_processed_root,
    get_logs_dir,
    get_results_meta_dir,
)
from src.results.build import ResultsBuildError, build_bug_results

logger = logging.getLogger(__name__)


def _processed_projects(dataset: str) -> list[str]:
    """Project dirs present under the processed root (``_*`` dirs excluded)."""
    root = get_benchmark_processed_root(dataset)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _processed_bugs(dataset: str, project: str) -> list[int]:
    """Numeric bug dirs present under one project's processed tree."""
    root = get_benchmark_processed_root(dataset) / project
    if not root.is_dir():
        return []
    return sorted(int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit())


def _valid_bip_projects() -> set[str] | None:
    """Project names from the BugsInPy index, or ``None`` when unavailable.

    Guards against stray non-benchmark dirs inside the processed tree — those
    must never be built into results/ or packaged for distribution.
    """
    try:
        from src.extraction.bugsinpy import get_bip_pids

        return set(get_bip_pids())
    except Exception:
        logger.warning("BugsInPy index unavailable; skipping project validation")
        return None


def _write_bip_meta(dataset: str) -> None:
    """Snapshot ``audit.csv`` + ``bugsinpy-index.csv`` into ``results/BIP/_meta/``."""
    meta_dir = get_results_meta_dir(dataset)
    meta_dir.mkdir(parents=True, exist_ok=True)

    audit_candidates = (
        get_logs_dir("extraction", dataset) / "audit.csv",
        get_benchmark_processed_root(dataset) / "_logs" / "extraction" / "audit.csv",
    )
    for candidate in audit_candidates:
        if candidate.exists():
            shutil.copy2(candidate, meta_dir / "audit.csv")
            logger.info("snapshotted audit: %s", candidate)
            break
    else:
        logger.warning("no extraction audit.csv found (looked in %s)", audit_candidates)

    try:
        from src.extraction.bugsinpy import _read_index_text

        (meta_dir / "bugsinpy-index.csv").write_text(_read_index_text(), encoding="utf-8")
        logger.info("snapshotted bugsinpy-index.csv")
    except Exception as exc:
        logger.warning("could not snapshot bugsinpy-index.csv: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill processed data into the committed results/ tree",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="defects4j",
        choices=supported_benchmarks(),
        help="Benchmark adapter key (e.g. defects4j, bugsinpy).",
    )
    parser.add_argument("-p", "--project", type=str, default=None, help="Project name.")
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Build every project present in the processed tree.",
    )
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Bug IDs to build. Omit to build all bugs with processed data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild bugs whose results dir already exists.",
    )
    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Skip copying full LLM records into transcripts/.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if bool(args.project) == bool(args.all_projects):
        parser.error("exactly one of -p/--project or --all-projects is required")

    dataset = get_benchmark_adapter(args.benchmark).benchmark_key
    log_dir = get_logs_dir("build_results", dataset)

    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"build_{stamp}.log"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )

    projects = [args.project] if args.project else _processed_projects(dataset)
    if dataset == "bugsinpy":
        valid = _valid_bip_projects()
        if valid is not None:
            strays = [p for p in projects if p not in valid]
            for stray in strays:
                logger.warning("skipping %r: not a BugsInPy project", stray)
            projects = [p for p in projects if p in valid]

    built = skipped = existed = 0
    failures: list[str] = []
    for project in projects:
        bugs = args.versions if args.versions is not None else _processed_bugs(dataset, project)
        for bug_id in bugs:
            try:
                outcome = build_bug_results(
                    project,
                    bug_id,
                    dataset=dataset,
                    include_transcripts=not args.no_transcripts,
                    force=args.force,
                )
            except ResultsBuildError as exc:
                logger.error("EQUIVALENCE FAILURE %s-%s: %s", project, bug_id, exc)
                failures.append(f"{project}-{bug_id}")
                continue
            except Exception as exc:
                logger.error("failed %s-%s: %s", project, bug_id, exc)
                failures.append(f"{project}-{bug_id}")
                continue
            status = outcome.get("status")
            if status == "built":
                built += 1
                logger.info(
                    "built %s-%s (%d entities, %d LR configs)",
                    project,
                    bug_id,
                    outcome.get("entities", 0),
                    outcome.get("lr_configs", 0),
                )
            elif status == "exists":
                existed += 1
            else:
                skipped += 1
                logger.info("skipped %s-%s: %s", project, bug_id, outcome.get("reason"))

    if dataset == "bugsinpy":
        _write_bip_meta(dataset)

    logger.info(
        "done: %d built, %d already existed, %d skipped, %d failed",
        built,
        existed,
        skipped,
        len(failures),
    )
    if failures:
        for f in failures:
            logger.warning("  failed: %s", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
