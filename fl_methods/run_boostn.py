#!/usr/bin/env python3
"""Run BoostN fault localization.

BoostN is an IR-based method-level FL technique that scores source methods
against a bug report using BM25 with adaptive k1/b parameters.

Usage:
    python run_boostn.py -p Lang -v 1
    python run_boostn.py -p Chart -v 1 5
    python run_boostn.py -p Lang          # all versions
"""

import argparse
import logging
import time

from update_tracker import update_single_bug

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.boostn.boostn import BoostN
from src.common.config import BOOSTN_CSV, get_boostn_dir, get_processed_dir
from src.common.tracker import TrackerStep
from src.core.layout import normalize_benchmark_name
from src.extraction.bugsinpy import BugsInPyRepo
from src.extraction.d4j import D4JRepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_boostn")


def _run_single_bug(
    project: str,
    bug_id: str,
    *,
    dataset: str = "defects4j",
    skip_existing: bool,
    keep_checkouts: bool,
) -> bool:
    """Dispatch BoostN for a single bug based on ``dataset``."""
    canonical = normalize_benchmark_name(dataset)
    if canonical == "D4J":
        return _run_single_bug_d4j(
            project,
            bug_id,
            skip_existing=skip_existing,
            keep_checkouts=keep_checkouts,
        )
    if canonical == "BIP":
        return _run_single_bug_bugsinpy(
            project,
            bug_id,
            skip_existing=skip_existing,
            keep_checkouts=keep_checkouts,
        )
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def _run_single_bug_d4j(
    project: str,
    bug_id: str,
    *,
    skip_existing: bool,
    keep_checkouts: bool,
) -> bool:
    """Run BoostN for a single Defects4J bug. Returns True on success."""
    log.info("=" * 60)
    log.info("BoostN Fault Localization")
    log.info("  Project : %s", project)
    log.info("  Version : %s", bug_id)
    log.info("=" * 60)

    processed_dir = get_processed_dir(project, bug_id, dataset="defects4j")
    inferred_tracker = update_single_bug(
        processed_dir, project, int(bug_id), dry_run=True, no_backup=True
    )
    if skip_existing and "boostn" in inferred_tracker["fl"]["completed"]:
        log.info("BoostN outputs already exist, skipping.")
        return True

    repo = D4JRepo(project, int(bug_id))

    try:
        repo.checkout(skip_existing=skip_existing)
        repo.export_property("dir.src.classes")

        t0 = time.time()
        with TrackerStep(project, bug_id, section="fl", step="boostn") as ts:
            try:
                boostn = BoostN()
                results = boostn.process_project(project, bug_id, dataset="defects4j")
                elapsed = time.time() - t0

                if results:
                    top5 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
                    log.info("Top-5 suspicious methods:")
                    for rank, (method_id, score) in enumerate(top5, 1):
                        log.info("  %d. %.4f  %s", rank, score, method_id)
                else:
                    log.warning("No results produced.")

                log.info("Completed in %.1f seconds.", elapsed)
            except Exception as exc:
                log.exception("Failed processing %s-%s", project, bug_id)
                ts.record_error(repr(exc))
                return False
        return True
    finally:
        if not keep_checkouts:
            try:
                repo.remove_repo()
            except Exception:
                log.exception(
                    "Failed to clean up checkout for %s-%s",
                    project,
                    bug_id,
                )


def _run_single_bug_bugsinpy(
    project: str,
    bug_id: str,
    *,
    skip_existing: bool,
    keep_checkouts: bool,
) -> bool:
    """Run BoostN for a single BugsInPy bug. Returns True on success.

    Mirrors the Defects4J path but uses :class:`BugsInPyRepo` (checkout + on-demand
    conda compile, no ``defects4j export``); the source import root is resolved by
    ``get_src_dir(dataset="bugsinpy")`` inside ``process_project``. The per-bug
    ``tracker.json`` / ``update_single_bug`` inference is intentionally bypassed for
    BugsInPy (its schema is D4J-shaped) — skip-existing is inferred from the output
    file directly.
    """
    log.info("=" * 60)
    log.info("BoostN Fault Localization (BugsInPy)")
    log.info("  Project : %s", project)
    log.info("  Version : %s", bug_id)
    log.info("=" * 60)

    if (
        skip_existing
        and (get_boostn_dir(project, bug_id, dataset="bugsinpy") / BOOSTN_CSV).exists()
    ):
        log.info("BoostN outputs already exist, skipping.")
        return True

    repo = BugsInPyRepo(project, int(bug_id))

    try:
        repo.checkout(skip_existing=skip_existing)
        repo.compile(skip_existing=True)

        t0 = time.time()
        results = BoostN().process_project(project, bug_id, dataset="bugsinpy")
        elapsed = time.time() - t0

        if results:
            top5 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info("Top-5 suspicious methods:")
            for rank, (method_id, score) in enumerate(top5, 1):
                log.info("  %d. %.4f  %s", rank, score, method_id)
        else:
            log.warning("No results produced.")

        log.info("Completed in %.1f seconds.", elapsed)
        return True
    except Exception:
        log.exception("Failed processing %s-%s", project, bug_id)
        return False
    finally:
        if not keep_checkouts:
            try:
                repo.remove_repo()
            except Exception:
                log.exception(
                    "Failed to clean up checkout for %s-%s",
                    project,
                    bug_id,
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BoostN method-level fault localization.")
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
        required=True,
        help="Project name (e.g., Lang, Chart for D4J; PySnooper, ansible for BugsInPy)",
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
        "--force",
        action="store_true",
        help="Re-run steps even if outputs already exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
        help="Do not remove checked-out repositories after processing",
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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    adapter = get_benchmark_adapter(benchmark_key)
    dataset_key = adapter.benchmark_key

    # Resolve versions
    versions = args.versions if args.versions is not None else adapter.list_cases(args.project)

    had_error = False
    for bug_id in versions:
        ok = _run_single_bug(
            args.project,
            str(bug_id),
            dataset=dataset_key,
            skip_existing=not args.force,
            keep_checkouts=args.keep_checkouts,
        )
        if not ok:
            had_error = True

    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
