#!/usr/bin/env python3
"""Evaluate fault-localization rankings for one or more bugs.

Runs entirely from the slim committed ``results/`` tree (built by
``run_build_results.py``) — no processed data, containers, or network access
required. Consumes each bug's baseline rankings + SR top-20 under
``results/<Benchmark>/<Project>/<BugId>/rankings/`` and its Agent4LR
``lr.json``, and produces:

* per-bug ``results/<...>/evaluation/{baselines,baselines_first,flexfl,flexfl_first}.csv``
* cross-bug long form at ``evaluation/<Benchmark>/<slot>.csv``
* cross-bug summary at ``evaluation/<Benchmark>/<slot>_summary.csv``
* (BugsInPy only) full-corpus exclusions report at
  ``evaluation/BIP/exclusions.csv``

Usage::

    python run_evaluation.py -p Lang -v 1
    python run_evaluation.py -p Lang             # all bugs present in results/
    python run_evaluation.py -p Lang -v 1 --force
    python run_evaluation.py -p Lang -v 1 --no-summary
    python run_evaluation.py --benchmark bugsinpy -p PySnooper -v 3
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

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import (
    get_logs_dir,
    get_results_bug_dir,
    get_results_root,
)
from src.evaluation.cross_bug import (
    SLOTS,
    append_to_long_csv,
    slot_paths,
    write_summary_csv,
)
from src.evaluation.exclusions import write_exclusions_report
from src.evaluation.per_bug import (
    RESULTS_EVALUATION_SUBDIR,
    evaluate_bug_from_results,
    write_per_bug_csvs,
)
from src.evaluation.sources import DEFAULT_SR_MODEL_ID
from src.results.read import check_required_configs_results

logger = logging.getLogger(__name__)


def process_bug(
    project: str,
    bug_id: int,
    *,
    dataset: str = "defects4j",
    skip_existing: bool = True,
    require_configs: list[str] | None = None,
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
) -> dict[str, Any]:
    """Evaluate a single bug and write per-bug + long-CSV rows.

    Returns a small result dict describing what was written.

    When ``require_configs`` is non-empty, the bug is gated: each named
    Agent4LR config must be present in ``lr.json`` with a usable ``top5``.
    A failing bug is skipped before any per-bug or cross-bug write, leaving
    existing outputs untouched.

    BugsInPy bugs without a candidate universe (no resolvable SR top-20)
    are skipped before any per-bug write, with ``skip_code="no_universe"``
    in the result. Unlike the scoping skips above, this skip *is* a
    data-state assertion, so the bug's prior rows are purged from the
    cross-bug long CSVs rather than left stale. Defects4J keeps its
    historical behavior (header-only per-bug CSVs + empty long-CSV rows).
    """
    results_dir = get_results_bug_dir(project, bug_id, dataset=dataset)
    if not results_dir.is_dir():
        return {
            "project": project,
            "bug_id": bug_id,
            "status": "skipped",
            "reason": "results dir missing (run run_build_results.py first)",
            "counts": {},
        }

    if require_configs:
        ok, problems = check_required_configs_results(results_dir, require_configs)
        if not ok:
            return {
                "project": project,
                "bug_id": bug_id,
                "status": "skipped",
                "reason": f"required configs incomplete: {', '.join(problems)}",
                "counts": {},
            }

    results = evaluate_bug_from_results(project, bug_id, dataset=dataset, sr_model_id=sr_model_id)

    # Empty universe ⟺ every slot empty (a non-empty universe always yields
    # baseline rows). For BugsInPy that means "not evaluable yet": skip before
    # any per-bug write, but purge prior long-CSV rows so nothing stale
    # survives (the same effect an empty append would have had).
    if dataset == "bugsinpy" and not any(results.values()):
        paths = slot_paths(dataset=dataset)
        for slot in SLOTS:
            long_csv, _ = paths[slot]
            append_to_long_csv(long_csv, project, bug_id, [])
        top20 = results_dir / "rankings" / "top20" / f"{sr_model_id}.txt"
        detail = "missing" if not top20.exists() else "empty or unresolvable"
        if (results_dir / RESULTS_EVALUATION_SUBDIR).is_dir():
            logger.warning(
                "stale per-bug evaluation/ CSVs left in place for skipped %s-%s",
                project,
                bug_id,
            )
        return {
            "project": project,
            "bug_id": bug_id,
            "status": "skipped",
            "skip_code": "no_universe",
            "reason": (
                f"no candidate universe: SR top-20 {detail} (rankings/top20/{sr_model_id}.txt)"
            ),
            "counts": {},
        }

    eval_dir = results_dir / RESULTS_EVALUATION_SUBDIR
    marker = eval_dir / "baselines.csv"
    if not (skip_existing and marker.exists()):
        write_per_bug_csvs(results_dir, results, subdir=RESULTS_EVALUATION_SUBDIR)

    # Always update the cross-bug long CSVs — idempotent on (Project, BugId).
    paths = slot_paths(dataset=dataset)
    for slot in SLOTS:
        long_csv, _ = paths[slot]
        append_to_long_csv(long_csv, project, bug_id, results.get(slot, []))

    counts = {slot: len(results.get(slot, [])) for slot in SLOTS}
    return {
        "project": project,
        "bug_id": bug_id,
        "status": "ok",
        "counts": counts,
    }


def write_summaries(dataset: str = "defects4j") -> None:
    """Regenerate every cross-bug summary CSV from the long CSVs."""
    paths = slot_paths(dataset=dataset)
    for slot in SLOTS:
        long_csv, summary_csv = paths[slot]
        write_summary_csv(long_csv, summary_csv)
        logger.info("wrote %s", summary_csv)


def _results_bugs(project: str, dataset: str) -> list[int]:
    """Bug ids present in ``results/<Benchmark>/<Project>/`` (numeric dirs).

    Enumerating from the results tree keeps the evaluation container-free:
    the Defects4J bug lister shells into the benchmark container, which a
    results-only checkout does not have.
    """
    project_dir = get_results_root(dataset) / project
    if not project_dir.is_dir():
        return []
    return sorted(int(d.name) for d in project_dir.iterdir() if d.is_dir() and d.name.isdigit())


def _results_projects(dataset: str) -> list[str]:
    """Projects present in ``results/<Benchmark>/`` (``_meta`` excluded)."""
    root = get_results_root(dataset)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _filter_range(bids: list[int], start: int | None, end: int | None) -> list[int]:
    result = bids
    if start is not None:
        result = [b for b in result if b >= start]
    if end is not None:
        result = [b for b in result if b <= end]
    return result


def _normalize_benchmark_args(args: argparse.Namespace) -> None:
    """Resolve ``--benchmark``/``--dataset`` to one canonical key on ``args.dataset``.

    ``--benchmark`` is authoritative (matches run_sbir.py/run_boostn.py/
    run_agent4sr.py/run_lr.py); ``-d/--dataset`` is a deprecated alias honored
    only when ``--benchmark`` was left at its default.
    """
    benchmark_key = args.benchmark
    if args.benchmark == "defects4j" and args.dataset != "defects4j":
        benchmark_key = args.dataset
    if benchmark_key not in supported_benchmarks():
        raise SystemExit(
            f"Benchmark {benchmark_key!r} not supported. Supported: {supported_benchmarks()}"
        )
    args.dataset = get_benchmark_adapter(benchmark_key).benchmark_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FL rankings against ground truth (per-bug + cross-bug)",
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
        "--sr-model-id",
        type=str,
        default=DEFAULT_SR_MODEL_ID,
        help=(
            "Agent4SR model id whose FlexFL/SR/rankings/top20/<id>.txt defines "
            "the candidate universe."
        ),
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=None,
        help="Project name (e.g. Chart, Lang).",
    )
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Evaluate every project present in the benchmark's results tree.",
    )
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Bug IDs to process. Omit to run all versions for the project.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Filter bug IDs >= start.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Filter bug IDs <= end.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate per-bug evaluation/*.csv even when they already exist.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the cross-bug summary aggregation step.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for run logs and JSON summaries.",
    )
    parser.add_argument(
        "--require-configs",
        nargs="+",
        default=None,
        metavar="CONFIG",
        help=(
            "Skip bugs missing any of these Agent4LR config names "
            "(or with an invalid top5). Useful to scope evaluation to a "
            "completed experiment sweep without hardcoding it."
        ),
    )

    args = parser.parse_args()
    _normalize_benchmark_args(args)
    if bool(args.project) == bool(args.all_projects):
        parser.error("exactly one of -p/--project or --all-projects is required")
    if args.all_projects and args.versions is not None:
        parser.error("-v/--versions cannot be combined with --all-projects")

    log_dir = args.log_dir or get_logs_dir("evaluation", args.dataset)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope = args.project.lower() if args.project else "all"
    log_path = log_dir / f"evaluation_{scope}_{run_stamp}.log"
    report_path = log_dir / f"evaluation_{scope}_{run_stamp}.json"

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

    projects = [args.project] if args.project else _results_projects(args.dataset)
    work: list[tuple[str, list[int]]] = []
    for project in projects:
        bids = args.versions if args.versions is not None else _results_bugs(project, args.dataset)
        work.append((project, _filter_range(bids, args.start, args.end)))
    total = sum(len(bids) for _project, bids in work)

    skip_existing = not args.force
    run_results: list[dict[str, Any]] = []
    failures: list[str] = []

    count = 0
    for project, versions in work:
        for bug_id in versions:
            count += 1
            started = time.time()
            bug_result: dict[str, Any] = {
                "project": project,
                "bug_id": bug_id,
                "status": "ok",
                "error": None,
                "duration_sec": 0.0,
            }
            logger.info("Evaluating %s-%d (%d/%d)", project, bug_id, count, total)
            try:
                details = process_bug(
                    project,
                    bug_id,
                    dataset=args.dataset,
                    skip_existing=skip_existing,
                    require_configs=args.require_configs,
                    sr_model_id=args.sr_model_id,
                )
                bug_result.update(details)
            except Exception as exc:
                msg = f"{project}-{bug_id}: {exc}"
                logger.error("Failed: %s", msg)
                logger.debug(traceback.format_exc())
                failures.append(msg)
                bug_result["status"] = "failed"
                bug_result["error"] = str(exc)
            finally:
                bug_result["duration_sec"] = round(time.time() - started, 2)
                run_results.append(bug_result)

    if not args.no_summary:
        write_summaries(dataset=args.dataset)

    if args.dataset == "bugsinpy":
        # Always regenerate: the report reflects full-corpus on-disk state,
        # independent of how this run was sliced.
        excl_path = write_exclusions_report(sr_model_id=args.sr_model_id)
        logger.info("Exclusions report (full corpus): %s", excl_path)

    skipped_gated = [
        r
        for r in run_results
        if r.get("status") == "skipped" and "required configs" in (r.get("reason") or "")
    ]
    if args.require_configs and skipped_gated:
        logger.info(
            "Skipped %d/%d bugs for failing --require-configs check",
            len(skipped_gated),
            total,
        )

    run_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "project": args.project or "(all)",
        "versions": dict(work),
        "force": args.force,
        "require_configs": args.require_configs,
        "sr_model_id": args.sr_model_id,
        "results": run_results,
    }
    report_path.write_text(json.dumps(run_report, indent=2) + "\n")
    logger.info("Run report: %s", report_path)

    if failures:
        logger.warning("Completed with %d/%d failures", len(failures), total)
        for err in failures:
            logger.warning("  - %s", err)
        sys.exit(1)
    logger.info("All %d bugs evaluated successfully.", total)


if __name__ == "__main__":
    main()
