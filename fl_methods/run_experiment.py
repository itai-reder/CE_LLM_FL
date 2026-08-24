"""Multi-config FL experiment orchestrator.

Drives the full CEFL pipeline (extraction → SBIR → BoostN → Agent4SR →
Agent4LR sweep → evaluation) across a configurable set of Defects4J
projects, bugs, and Agent4LR named configs by subprocessing the existing
per-phase ``run_*.py`` entry points.

Examples::

    # Default experiment: 6 Agent4LR configs across first-3 bugs of all D4J projects
    python fl_methods/run_experiment.py

    # Dry-run (print subprocess commands without executing them)
    python fl_methods/run_experiment.py --dry-run

    # Smoke test on one bug, one LR config
    python fl_methods/run_experiment.py -p Lang -n 1 --lr-configs M1R1-M1R1-M1R1

    # Skip already-extracted bugs but force re-evaluation
    python fl_methods/run_experiment.py --skip-stage extraction --force-stage evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any

from src.benchmarks.registry import get_benchmark_adapter
from src.common.config import get_logs_dir

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGES: tuple[str, ...] = (
    "extraction",
    "sbir",
    "boostn",
    "agent4sr",
    "agent4lr",
    "evaluation",
)

# Sentinel for "every bug of every project" — `_resolve_bugs` slices
# `list_cases(project)[:n]`, and the largest D4J project has ≈250 bugs, so any value
# well above that selects the full benchmark. Overridable at run time with `-n`.
FULL_BENCHMARK = 100_000


@dataclass(frozen=True)
class Experiment:
    """Declarative description of an end-to-end pipeline sweep."""

    name: str
    projects: tuple[str, ...] | None  # None ⇒ all projects from the adapter
    n_bugs_per_project: int
    lr_configs: tuple[str, ...]
    sr_model: str = "llama3.1:8b"
    sr_base_url: str = "http://localhost:11434"
    sr_verify_ssl: bool = True
    stages: tuple[str, ...] = field(default_factory=lambda: STAGES)


EXPERIMENTS: dict[str, Experiment] = {
    "default": Experiment(
        name="default",
        projects=None,
        n_bugs_per_project=3,
        lr_configs=(
            "M1R1-M1R1-M1R1",
            "M2R1-M2R1-M2R1",
            "M3R1-M3R1-M3R1",
            "M2R2-M2R2-M2R2",
            "M2R3-M2R3-M2R3",
            # "M2R4-M2R4-M2R4",  # avg 217s/bug @ high reasoning — excluded from default sweep
        ),
        sr_model="llama3.1:8b",
        sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
        sr_verify_ssl=False,
    ),
    "tool_caller_swap": Experiment(
        name="tool_caller_swap",
        projects=None,
        n_bugs_per_project=3,
        lr_configs=(
            "M2R1-M1R1-M2R1",
            "M2R1-M2R1-M2R1",
            "M2R1-M2R2-M2R1",
            "M2R1-M2R3-M2R1",
            "M2R1-M3R1-M2R1",
        ),
        sr_model="llama3.1:8b",
        sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
        sr_verify_ssl=False,
    ),
    "planner_swap": Experiment(
        name="planner_swap",
        projects=None,
        n_bugs_per_project=3,
        lr_configs=(
            "M1R1-M2R1-M2R1",
            "M2R1-M2R1-M2R1",
            "M2R2-M2R1-M2R1",
            "M2R3-M2R1-M2R1",
            "M3R1-M2R1-M2R1",
        ),
        sr_model="llama3.1:8b",
        sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
        sr_verify_ssl=False,
    ),
    "finisher_swap": Experiment(
        name="finisher_swap",
        projects=None,
        n_bugs_per_project=3,
        lr_configs=(
            "M2R1-M2R1-M1R1",
            "M2R1-M2R1-M2R1",
            "M2R1-M2R1-M2R2",
            "M2R1-M2R1-M2R3",
            "M2R1-M2R1-M3R1",
        ),
        sr_model="llama3.1:8b",
        sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
        sr_verify_ssl=False,
    ),
    "picky": Experiment(
        name="picky",
        projects=None,
        n_bugs_per_project=3,
        lr_configs=(
            "M2R2-M1R1-M2R2",
            "M2R2-M1R1-M3R1",
            "M2R2-M2R2-M2R2",
            "M2R2-M2R2-M3R1",
            "M2R2-M1R1-M2R3",
            "M2R2-M2R2-M2R3",
        ),
        sr_model="llama3.1:8b",
        sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
        sr_verify_ssl=False,
    ),
    "cheaper": Experiment(
        name="cheaper",
        projects=None,
        n_bugs_per_project=3,
        lr_configs=(
            "M1R1-M1R1-M1R1",
            "M1R1-M1R1-M2R1",
            "M1R1-M1R1-M3R1",
            "M1R1-M1R1-M2R2",
            "M1R1-M1R1-M2R3",
        ),
        sr_model="llama3.1:8b",
        sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
        sr_verify_ssl=False,
    ),
}


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for x in items:
        seen.setdefault(x, None)
    return tuple(seen)


EXPERIMENTS["swaps"] = Experiment(
    name="swaps",
    projects=None,
    n_bugs_per_project=3,
    lr_configs=_dedup(
        chain.from_iterable(
            EXPERIMENTS[k].lr_configs
            for k in ("default", "tool_caller_swap", "planner_swap", "finisher_swap")
        )
    ),
    sr_model="llama3.1:8b",
    sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
    sr_verify_ssl=False,
)


def _swap_expanded_configs() -> tuple[str, ...]:
    """The (cut) swap-expanded grid as canonical ``Planner-ToolCaller-Finisher`` strings.

    For every fixed tier ``i`` and swapped variant ``M<m>R<r>`` (``i,m,r`` in 1..3),
    swap each role in turn against the ``M<i>R1`` anchors:

    - ``P<i>(M<m>R<r>)`` → ``M<m>R<r>-M<i>R1-M<i>R1`` (planner swapped)
    - ``T<i>(M<m>R<r>)`` → ``M<i>R1-M<m>R<r>-M<i>R1`` (tool-caller swapped)
    - ``F<i>(M<m>R<r>)`` → ``M<i>R1-M<i>R1-M<m>R<r>`` (finisher swapped)

    The swept variant is pruned to ``m == 2`` OR ``r == 1`` (see the guard below): only
    the mid model ``M2`` gets its full reasoning sweep ``R1..R3``; ``M1`` and ``M3`` are
    pinned at ``R1``. That leaves 5 swept variants per (role, i) --
    ``{M1R1, M2R1, M2R2, M2R3, M3R1}`` -- so the 45 label instances collapse to 39 unique
    configs (the homogeneous anchors ``M<i>R1-M<i>R1-M<i>R1`` are shared across the three
    roles).
    """
    configs: list[str] = []
    for i in (1, 2, 3):
        anchor = f"M{i}R1"
        for m in (1, 2, 3):
            for r in (1, 2, 3):
                # NOTE: M1 and M3 are only executed with R1 to avoid exponential growth in config space
                if m != 2 and r != 1:
                    continue
                swap = f"M{m}R{r}"
                configs.append(f"{swap}-{anchor}-{anchor}")  # planner
                configs.append(f"{anchor}-{swap}-{anchor}")  # tool_caller
                configs.append(f"{anchor}-{anchor}-{swap}")  # finisher
    return _dedup(configs)


EXPERIMENTS["swap-expanded"] = Experiment(
    name="swap-expanded",
    projects=None,
    n_bugs_per_project=FULL_BENCHMARK,  # full D4J benchmark; override with -n
    lr_configs=_swap_expanded_configs(),
    sr_model="llama3.1:8b",
    sr_base_url="https://cis-ollama.auth.ad.bgu.ac.il",
    sr_verify_ssl=False,
)


def _build_stage_command(
    stage: str,
    project: str,
    bug_id: int,
    experiment: Experiment,
    *,
    force: bool,
    lr_config: str | None = None,
) -> list[str]:
    """Materialise the argv vector for a single stage subprocess."""
    base = [sys.executable]
    bug = str(bug_id)
    force_flag: list[str] = ["--force"] if force else []

    if stage == "extraction":
        return [
            *base,
            "fl_methods/run_extraction.py",
            "-p",
            project,
            "-v",
            bug,
            *force_flag,
        ]
    if stage == "sbir":
        return [
            *base,
            "fl_methods/run_sbir.py",
            "-p",
            project,
            "-v",
            bug,
            *force_flag,
        ]
    if stage == "boostn":
        return [
            *base,
            "fl_methods/run_boostn.py",
            "-p",
            project,
            "-v",
            bug,
            *force_flag,
        ]
    if stage == "agent4sr":
        cmd = [
            *base,
            "fl_methods/run_agent4sr.py",
            "run",
            "-p",
            project,
            "-v",
            bug,
            "--model",
            experiment.sr_model,
            "--base-url",
            experiment.sr_base_url,
        ]
        if not experiment.sr_verify_ssl:
            cmd.append("--no-verify")
        cmd.extend(force_flag)
        return cmd
    if stage == "agent4lr":
        if lr_config is None:
            raise ValueError("agent4lr stage requires lr_config")
        return [
            *base,
            "fl_methods/run_lr.py",
            "-p",
            project,
            "-v",
            bug,
            "--config",
            lr_config,
            *force_flag,
        ]
    if stage == "evaluation":
        return [
            *base,
            "fl_methods/run_evaluation.py",
            "-p",
            project,
            "-v",
            bug,
            *force_flag,
        ]
    raise ValueError(f"Unknown stage: {stage}")


def _stage_label(stage: str, lr_config: str | None) -> str:
    return f"{stage}:{lr_config}" if (stage == "agent4lr" and lr_config) else stage


def _run_stage(
    stage: str,
    project: str,
    bug_id: int,
    experiment: Experiment,
    *,
    force: bool,
    dry_run: bool,
    lr_config: str | None = None,
    log_path: Path | None = None,
) -> tuple[str, int, float]:
    """Run one stage subprocess; return (status, return_code, duration_sec).

    status is "ok" on rc=0, "dry-run" when dry_run is set, "failed" otherwise.

    If ``log_path`` is provided, subprocess stdout+stderr are appended to that
    file (used when multiple bugs run in parallel to avoid interleaved output).
    Otherwise stdout/stderr are inherited from the parent — today's behaviour
    at ``--workers 1``.
    """
    cmd = _build_stage_command(stage, project, bug_id, experiment, force=force, lr_config=lr_config)
    label = _stage_label(stage, lr_config)
    logger.info("  >>> [%s-%d] %s: %s", project, bug_id, label, " ".join(cmd))
    if dry_run:
        return "dry-run", 0, 0.0

    started = time.time()
    if log_path is not None:
        with log_path.open("ab") as sink:
            header = f"\n===== {project}-{bug_id} {label} =====\n$ {' '.join(cmd)}\n"
            sink.write(header.encode())
            sink.flush()
            result = subprocess.run(
                cmd, check=False, cwd=str(REPO_ROOT), stdout=sink, stderr=subprocess.STDOUT
            )
    else:
        result = subprocess.run(cmd, check=False, cwd=str(REPO_ROOT))
    duration = round(time.time() - started, 2)
    if result.returncode == 0:
        return "ok", 0, duration
    logger.warning(
        "  !!! %s for %s-%d exited rc=%d (duration=%.1fs)",
        label,
        project,
        bug_id,
        result.returncode,
        duration,
    )
    return "failed", result.returncode, duration


def _resolve_projects(experiment: Experiment, override: tuple[str, ...] | None) -> list[str]:
    adapter = get_benchmark_adapter("defects4j")
    if override:
        return list(override)
    if experiment.projects:
        return list(experiment.projects)
    return list(adapter.list_projects())


def _resolve_bugs(project: str, n: int) -> list[int]:
    adapter = get_benchmark_adapter("defects4j")
    return list(adapter.list_cases(project))[:n]


def _run_bug(
    project: str,
    bug_id: int,
    experiment: Experiment,
    *,
    lr_configs: tuple[str, ...],
    skip_stages: set[str],
    force_stages: set[str],
    dry_run: bool,
    log_path: Path | None,
) -> dict[str, Any]:
    """Run every stage for one (project, bug). LR configs run serially in-bug
    so the LR slot-checkpoint cache from a prior config is warm by the time
    the next one starts."""
    bug_started = time.time()
    stages_status: dict[str, str] = {}
    stages_rc: dict[str, int] = {}
    for stage in experiment.stages:
        if stage in skip_stages:
            stages_status[stage] = "skipped"
            continue
        if stage == "agent4lr":
            for cfg in lr_configs:
                label = _stage_label(stage, cfg)
                status, rc, _dur = _run_stage(
                    stage,
                    project,
                    bug_id,
                    experiment,
                    force=stage in force_stages,
                    dry_run=dry_run,
                    lr_config=cfg,
                    log_path=log_path,
                )
                stages_status[label] = status
                if rc != 0:
                    stages_rc[label] = rc
            continue
        status, rc, _dur = _run_stage(
            stage,
            project,
            bug_id,
            experiment,
            force=stage in force_stages,
            dry_run=dry_run,
            log_path=log_path,
        )
        stages_status[stage] = status
        if rc != 0:
            stages_rc[stage] = rc

    return {
        "project": project,
        "bug_id": bug_id,
        "stages": stages_status,
        "return_codes": stages_rc,
        "duration_sec": round(time.time() - bug_started, 2),
    }


def _run_experiment(
    experiment: Experiment,
    *,
    project_override: tuple[str, ...] | None,
    n_override: int | None,
    lr_override: tuple[str, ...] | None,
    skip_stages: set[str],
    force_stages: set[str],
    dry_run: bool,
    workers: int,
    bug_log_dir: Path | None,
) -> list[dict[str, Any]]:
    projects = _resolve_projects(experiment, project_override)
    n_bugs = n_override if n_override is not None else experiment.n_bugs_per_project
    lr_configs = lr_override if lr_override else experiment.lr_configs

    bug_pairs: list[tuple[str, int]] = [
        (project, bug_id) for project in projects for bug_id in _resolve_bugs(project, n_bugs)
    ]
    total_bugs = len(bug_pairs)

    parallel = workers > 1 and not dry_run
    if parallel and bug_log_dir is not None:
        bug_log_dir.mkdir(parents=True, exist_ok=True)

    def submit(project: str, bug_id: int) -> dict[str, Any]:
        log_path = (bug_log_dir / f"{project}-{bug_id}.log") if parallel and bug_log_dir else None
        logger.info("Started  %s-%d", project, bug_id)
        return _run_bug(
            project,
            bug_id,
            experiment,
            lr_configs=lr_configs,
            skip_stages=skip_stages,
            force_stages=force_stages,
            dry_run=dry_run,
            log_path=log_path,
        )

    results: list[dict[str, Any]] = []

    if not parallel:
        # Serial path: identical behaviour to pre-refactor, including inherited
        # stdout/stderr from subprocesses.
        for idx, (project, bug_id) in enumerate(bug_pairs, start=1):
            logger.info("Processing %s-%d (%d/%d)", project, bug_id, idx, total_bugs)
            results.append(submit(project, bug_id))
        return results

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(submit, project, bug_id): (project, bug_id) for project, bug_id in bug_pairs
        }
        for fut in as_completed(futures):
            project, bug_id = futures[fut]
            try:
                result = fut.result()
            except Exception:
                logger.exception("Worker for %s-%d raised", project, bug_id)
                result = {
                    "project": project,
                    "bug_id": bug_id,
                    "stages": {},
                    "return_codes": {"__worker_exception__": 1},
                    "duration_sec": 0.0,
                }
            results.append(result)
            completed += 1
            logger.info("Completed %s-%d (%d/%d)", project, bug_id, completed, total_bugs)

    results.sort(key=lambda r: (r["project"], r["bug_id"]))
    return results


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total_bugs": len(results), "ok": 0, "failed": 0, "with_failures": 0}
    for r in results:
        stages = r["stages"]
        failed_in_bug = sum(1 for s in stages.values() if s == "failed")
        if failed_in_bug == 0:
            summary["ok"] += 1
        else:
            summary["with_failures"] += 1
            summary["failed"] += failed_in_bug
    return summary


def _setup_logging(log_dir: Path, exp_name: str, verbose: bool) -> tuple[Path, Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"experiment_{exp_name}_{stamp}.log"
    report_path = log_dir / f"experiment_{exp_name}_{stamp}.json"
    bug_log_dir = log_dir / f"experiment_{exp_name}_{stamp}_bugs"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )
    return log_path, report_path, bug_log_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-config FL experiment orchestrator (subprocess-based).",
    )
    parser.add_argument(
        "--experiment",
        default="default",
        choices=sorted(EXPERIMENTS.keys()),
        help="Named experiment to run (default: default).",
    )
    parser.add_argument(
        "-p",
        "--projects",
        nargs="*",
        default=None,
        help="Override the experiment's project list.",
    )
    parser.add_argument(
        "-n",
        "--n-bugs-per-project",
        type=int,
        default=None,
        help="Override the number of bugs taken from each project.",
    )
    parser.add_argument(
        "--lr-configs",
        nargs="*",
        default=None,
        help="Override the Agent4LR config name list.",
    )
    parser.add_argument("--sr-model", default=None, help="Override the experiment's SR model.")
    parser.add_argument(
        "--sr-base-url",
        default=None,
        help="Override the experiment's SR (Ollama) base URL.",
    )
    parser.add_argument(
        "--no-sr-verify",
        action="store_true",
        help="Disable TLS verification for the SR Ollama base URL.",
    )
    parser.add_argument(
        "--skip-stage",
        nargs="*",
        default=[],
        choices=STAGES,
        help="Stages to skip entirely.",
    )
    parser.add_argument(
        "--force-stage",
        nargs="*",
        default=[],
        choices=STAGES,
        help="Stages to invoke with --force.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stage commands without executing them.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of bugs to process in parallel (default: 1, serial). "
            "Each worker handles one (project, bug) end-to-end; LR configs "
            "inside a bug still run sequentially so the slot-checkpoint cache "
            "is reused. Ignored with --dry-run."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=get_logs_dir("experiment"),
        help="Directory for run log + JSON report.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_experiment = EXPERIMENTS[args.experiment]

    overrides: dict[str, Any] = {}
    if args.sr_model is not None:
        overrides["sr_model"] = args.sr_model
    if args.sr_base_url is not None:
        overrides["sr_base_url"] = args.sr_base_url
    if args.no_sr_verify:
        overrides["sr_verify_ssl"] = False
    experiment = (
        Experiment(
            name=base_experiment.name,
            projects=base_experiment.projects,
            n_bugs_per_project=base_experiment.n_bugs_per_project,
            lr_configs=base_experiment.lr_configs,
            sr_model=overrides.get("sr_model", base_experiment.sr_model),
            sr_base_url=overrides.get("sr_base_url", base_experiment.sr_base_url),
            sr_verify_ssl=overrides.get("sr_verify_ssl", base_experiment.sr_verify_ssl),
            stages=base_experiment.stages,
        )
        if overrides
        else base_experiment
    )

    log_path, report_path, bug_log_dir = _setup_logging(args.log_dir, experiment.name, args.verbose)
    logger.info("Experiment: %s", experiment.name)
    logger.info("Log file: %s", log_path)
    logger.info("Report file: %s", report_path)
    if args.workers > 1 and not args.dry_run:
        logger.info("Parallel mode: %d workers, per-bug logs in %s", args.workers, bug_log_dir)
    if args.dry_run:
        logger.info("Dry-run mode: subprocesses will NOT be executed.")

    project_override = tuple(args.projects) if args.projects else None
    lr_override = tuple(args.lr_configs) if args.lr_configs else None

    results = _run_experiment(
        experiment,
        project_override=project_override,
        n_override=args.n_bugs_per_project,
        lr_override=lr_override,
        skip_stages=set(args.skip_stage),
        force_stages=set(args.force_stage),
        dry_run=args.dry_run,
        workers=max(1, args.workers),
        bug_log_dir=bug_log_dir,
    )

    summary = _summarize_results(results)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment.name,
        "arguments": {
            "experiment": args.experiment,
            "projects": args.projects,
            "n_bugs_per_project": args.n_bugs_per_project,
            "lr_configs": args.lr_configs,
            "sr_model": experiment.sr_model,
            "sr_base_url": experiment.sr_base_url,
            "sr_verify_ssl": experiment.sr_verify_ssl,
            "skip_stage": args.skip_stage,
            "force_stage": args.force_stage,
            "dry_run": args.dry_run,
            "workers": args.workers,
        },
        "stages_run": list(experiment.stages),
        "lr_configs": list(lr_override or experiment.lr_configs),
        "summary": summary,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    logger.info(
        "Completed: %d bugs (ok=%d, with_failures=%d, total stage failures=%d)",
        summary["total_bugs"],
        summary["ok"],
        summary["with_failures"],
        summary["failed"],
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
