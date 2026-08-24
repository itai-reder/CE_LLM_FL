#!/usr/bin/env python3
"""Run the full SBIR pipeline: SBFL -> Blues -> RAFL.

SBIR is a 3-stage fault localization approach:
  1. SBFL  -- Parse existing GZoltar/Ochiai spectrum-based rankings
  2. Blues -- IR-based statement scoring using BM25 on bug report vs source
  3. RAFL  -- Rank aggregation (Borda count or CE Monte Carlo)

Usage:
    python run_sbir.py -p Lang -v 1
    python run_sbir.py -p Chart -v 1 5
    python run_sbir.py -p Lang               # all versions
    python run_sbir.py -p Lang -v 1 --use_ce  # with CE Monte Carlo
"""

import argparse
import logging
import time

from update_tracker import update_single_bug

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import (
    SBFL_STMT_SUSPS,
    SBIR_STMT_SUSPS,
    get_ochiai_dir,
    get_processed_dir,
    get_sbir_dir,
)
from src.common.tracker import TrackerStep
from src.core.layout import normalize_benchmark_name
from src.extraction.bugsinpy import BugsInPyRepo
from src.extraction.d4j import D4JRepo
from src.sbir.blues import Blues
from src.sbir.rafl import RAFL
from src.sbir.sbfl import SBFL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_sbir")


def _run_single_bug(
    project: str,
    bug_id: str,
    *,
    dataset: str = "defects4j",
    skip_sbfl: bool,
    skip_blues: bool,
    use_ce: bool,
    ce_max_iter: int,
    ce_pop_size: int,
    sbfl_weight: float,
    skip_existing: bool,
    keep_checkouts: bool,
) -> bool:
    """Dispatch SBIR for a single bug based on ``dataset``."""
    canonical = normalize_benchmark_name(dataset)
    if canonical == "D4J":
        return _run_single_bug_d4j(
            project,
            bug_id,
            skip_sbfl=skip_sbfl,
            skip_blues=skip_blues,
            use_ce=use_ce,
            ce_max_iter=ce_max_iter,
            ce_pop_size=ce_pop_size,
            sbfl_weight=sbfl_weight,
            skip_existing=skip_existing,
            keep_checkouts=keep_checkouts,
        )
    if canonical == "BIP":
        return _run_single_bug_bugsinpy(
            project,
            bug_id,
            skip_sbfl=skip_sbfl,
            skip_blues=skip_blues,
            use_ce=use_ce,
            ce_max_iter=ce_max_iter,
            ce_pop_size=ce_pop_size,
            sbfl_weight=sbfl_weight,
            skip_existing=skip_existing,
            keep_checkouts=keep_checkouts,
        )
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def _run_single_bug_d4j(
    project: str,
    bug_id: str,
    *,
    skip_sbfl: bool,
    skip_blues: bool,
    use_ce: bool,
    ce_max_iter: int,
    ce_pop_size: int,
    sbfl_weight: float,
    skip_existing: bool,
    keep_checkouts: bool,
) -> bool:
    """Run SBIR for a single Defects4J bug. Returns True on success."""
    log.info("=" * 60)
    log.info("SBIR Fault Localization Pipeline")
    log.info("  Project : %s", project)
    log.info("  Version : %s", bug_id)
    log.info("  RAFL    : %s", "CE Monte Carlo" if use_ce else "Borda Count")
    log.info("  SBFL wt : %.2f", sbfl_weight)
    log.info("=" * 60)

    repo = D4JRepo(project, int(bug_id))

    try:
        repo.checkout(skip_existing=skip_existing)
        repo.export_property("dir.src.classes")

        processed_dir = get_processed_dir(project, bug_id, dataset="defects4j")
        inferred_tracker = update_single_bug(
            processed_dir, project, int(bug_id), dry_run=True, no_backup=True
        )

        skip_rafl = False
        if skip_existing:
            if "ochiai" in inferred_tracker["fl"]["completed"]:
                skip_sbfl = True
            if "sbir" in inferred_tracker["fl"]["completed"]:
                skip_blues = True
                skip_rafl = True

        t_total = time.time()

        # -- Stage 1: SBFL ---------------------------------------------------
        if not skip_sbfl:
            log.info("--- Stage 1/3: SBFL (parsing GZoltar Ochiai) ---")
            t0 = time.time()
            with TrackerStep(project, bug_id, section="fl", step="ochiai") as ts:
                try:
                    sbfl = SBFL()
                    sbfl_results = sbfl.process_project(project, bug_id, dataset="defects4j")
                    log.info(
                        "SBFL: %d statements ranked in %.1fs",
                        len(sbfl_results) if sbfl_results else 0,
                        time.time() - t0,
                    )
                except Exception as exc:
                    ts.record_error(repr(exc))
                    raise
        else:
            log.info("--- Stage 1/3: SBFL (skipped -- using existing output) ---")

        # -- Stage 2: Blues ---------------------------------------------------
        if not skip_blues:
            log.info("--- Stage 2/3: Blues (IR-based statement scoring) ---")
            t0 = time.time()
            blues = Blues()
            blues_results = blues.process_project(project, bug_id, dataset="defects4j")
            log.info(
                "Blues: %d statements scored in %.1fs",
                len(blues_results) if blues_results else 0,
                time.time() - t0,
            )
        else:
            log.info("--- Stage 2/3: Blues (skipped -- using existing output) ---")

        # -- Stage 3: RAFL ---------------------------------------------------
        if not skip_rafl:
            log.info("--- Stage 3/3: RAFL (rank aggregation) ---")
            t0 = time.time()
            with TrackerStep(project, bug_id, section="fl", step="sbir") as ts:
                try:
                    rafl = RAFL()
                    rafl_results = rafl.process_project(
                        project,
                        bug_id,
                        use_ce=use_ce,
                        ce_max_iter=ce_max_iter,
                        ce_pop_size=ce_pop_size,
                        sbfl_weight=sbfl_weight,
                        dataset="defects4j",
                    )
                    log.info(
                        "RAFL: %d statements in final ranking in %.1fs",
                        len(rafl_results) if rafl_results else 0,
                        time.time() - t0,
                    )
                except Exception as exc:
                    ts.record_error(repr(exc))
                    raise
        else:
            log.info("--- Stage 3/3: RAFL (skipped -- using existing output) ---")
            rafl_results = None

        # -- Summary ----------------------------------------------------------
        elapsed = time.time() - t_total
        log.info("=" * 60)
        if rafl_results:
            top10 = sorted(rafl_results.items(), key=lambda x: x[1], reverse=True)[:10]
            log.info("Top-10 suspicious statements (SBIR):")
            for rank, (stmt_id, score) in enumerate(top10, 1):
                log.info("  %2d. %.4f  %s", rank, score, stmt_id)
        else:
            log.warning("No final results produced.")
        log.info("Total pipeline time: %.1f seconds.", elapsed)
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


def _run_single_bug_bugsinpy(
    project: str,
    bug_id: str,
    *,
    skip_sbfl: bool,
    skip_blues: bool,
    use_ce: bool,
    ce_max_iter: int,
    ce_pop_size: int,
    sbfl_weight: float,
    skip_existing: bool,
    keep_checkouts: bool,
) -> bool:
    """Run SBIR for a single BugsInPy bug. Returns True on success.

    Mirrors the Defects4J path but uses :class:`BugsInPyRepo` (checkout + on-demand
    conda compile, no ``defects4j export``); the source import root is resolved by
    ``get_src_dir(dataset="bugsinpy")`` inside the FL stages. The per-bug
    ``tracker.json`` / ``update_single_bug`` inference is intentionally bypassed for
    BugsInPy (its schema is D4J-shaped) — skip-existing is inferred from the output
    files directly.
    """
    log.info("=" * 60)
    log.info("SBIR Fault Localization Pipeline (BugsInPy)")
    log.info("  Project : %s", project)
    log.info("  Version : %s", bug_id)
    log.info("  RAFL    : %s", "CE Monte Carlo" if use_ce else "Borda Count")
    log.info("  SBFL wt : %.2f", sbfl_weight)
    log.info("=" * 60)

    repo = BugsInPyRepo(project, int(bug_id))

    try:
        repo.checkout(skip_existing=skip_existing)
        repo.compile(skip_existing=True)

        skip_rafl = False
        if skip_existing:
            ochiai_out = get_ochiai_dir(project, bug_id, dataset="bugsinpy") / SBFL_STMT_SUSPS
            sbir_out = get_sbir_dir(project, bug_id, dataset="bugsinpy") / SBIR_STMT_SUSPS
            if ochiai_out.exists():
                skip_sbfl = True
            if sbir_out.exists():
                skip_blues = True
                skip_rafl = True

        t_total = time.time()

        # -- Stage 1: SBFL ---------------------------------------------------
        if not skip_sbfl:
            log.info("--- Stage 1/3: SBFL (parsing FauxPy Ochiai) ---")
            t0 = time.time()
            sbfl_results = SBFL().process_project(project, bug_id, dataset="bugsinpy")
            log.info(
                "SBFL: %d statements ranked in %.1fs",
                len(sbfl_results) if sbfl_results else 0,
                time.time() - t0,
            )
        else:
            log.info("--- Stage 1/3: SBFL (skipped -- using existing output) ---")

        # -- Stage 2: Blues ---------------------------------------------------
        if not skip_blues:
            log.info("--- Stage 2/3: Blues (IR-based statement scoring) ---")
            t0 = time.time()
            blues_results = Blues().process_project(project, bug_id, dataset="bugsinpy")
            log.info(
                "Blues: %d statements scored in %.1fs",
                len(blues_results) if blues_results else 0,
                time.time() - t0,
            )
        else:
            log.info("--- Stage 2/3: Blues (skipped -- using existing output) ---")

        # -- Stage 3: RAFL ---------------------------------------------------
        rafl_results = None
        if not skip_rafl:
            log.info("--- Stage 3/3: RAFL (rank aggregation) ---")
            t0 = time.time()
            rafl_results = RAFL().process_project(
                project,
                bug_id,
                use_ce=use_ce,
                ce_max_iter=ce_max_iter,
                ce_pop_size=ce_pop_size,
                sbfl_weight=sbfl_weight,
                dataset="bugsinpy",
            )
            log.info(
                "RAFL: %d statements in final ranking in %.1fs",
                len(rafl_results) if rafl_results else 0,
                time.time() - t0,
            )
        else:
            log.info("--- Stage 3/3: RAFL (skipped -- using existing output) ---")

        # -- Summary ----------------------------------------------------------
        elapsed = time.time() - t_total
        log.info("=" * 60)
        if rafl_results:
            top10 = sorted(rafl_results.items(), key=lambda x: x[1], reverse=True)[:10]
            log.info("Top-10 suspicious statements (SBIR):")
            for rank, (stmt_id, score) in enumerate(top10, 1):
                log.info("  %2d. %.4f  %s", rank, score, stmt_id)
        else:
            log.warning("No final results produced.")
        log.info("Total pipeline time: %.1f seconds.", elapsed)
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
    parser = argparse.ArgumentParser(
        description="Run the full SBIR fault localization pipeline.",
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
        "--use_ce",
        action="store_true",
        help="Use Cross-Entropy Monte Carlo for RAFL (slower, more accurate)",
    )
    parser.add_argument(
        "--ce_max_iter",
        type=int,
        default=1000,
        help="Max iterations for CE optimization (default: 1000)",
    )
    parser.add_argument(
        "--ce_pop_size",
        type=int,
        default=100,
        help="Population size for CE optimization (default: 100)",
    )
    parser.add_argument(
        "--sbfl_weight",
        type=float,
        default=0.5,
        help="SBFL weight in weighted Borda aggregation (Blues weight is 1-sbfl_weight).",
    )
    parser.add_argument(
        "--skip_sbfl",
        action="store_true",
        help="Skip SBFL step (use existing stmt-susps.txt)",
    )
    parser.add_argument(
        "--skip_blues",
        action="store_true",
        help="Skip Blues step (use existing stmt-susps-blues.txt)",
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
            skip_sbfl=args.skip_sbfl,
            skip_blues=args.skip_blues,
            use_ce=args.use_ce,
            ce_max_iter=args.ce_max_iter,
            ce_pop_size=args.ce_pop_size,
            sbfl_weight=args.sbfl_weight,
            skip_existing=not args.force,
            keep_checkouts=args.keep_checkouts,
        )
        if not ok:
            had_error = True

    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
