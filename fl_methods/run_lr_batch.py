#!/usr/bin/env python3
"""Drive Agent4LR through the OpenAI Batch API (50% cost, 24h turnaround).

Each **tick** (see :mod:`src.agent4lr.batch_runner`):

1. process finished batches (apply responses, promote checkpoints),
2. finalize every (bug, config) whose chain is fully checkpointed —
   a zero-LLM replay through the regular ``run_lr`` path,
3. send one batch round advancing every remaining agent by one API call
   (one JSONL per model; requests deduped across configs sharing
   checkpoint prefixes and filtered against in-flight batches).

A full 3-slot chain with a k-iteration tool loop needs at most ``k + 2``
rounds. By default one tick runs per invocation; ``--ticks N`` runs up to
N ticks (polling the Batch API between them), and ``--auto`` keeps ticking
until every (bug, config) is finalized. ``--experiment`` pulls the config
grid (and default project list) straight from ``run_experiment`` and
implies ``--auto`` unless ``--ticks`` is given.

Usage::

    # Run the whole swap-expanded grid for four projects to completion
    python fl_methods/run_lr_batch.py --benchmark bugsinpy \\
        -p pandas keras youtube-dl scrapy --experiment swap-expanded

    # One round for two projects across four configs
    python fl_methods/run_lr_batch.py -p Lang Chart \\
        --configs M2R1-M2R1-M1R1 M2R1-M2R1-M2R2 M2R1-M2R1-M2R3 M2R1-M2R1-M3R1

    # Advance at most 3 rounds, then stop (polling between rounds)
    python fl_methods/run_lr_batch.py -p Lang --configs M1R1-M1R1-M1R1 --ticks 3

    # Inspect without touching the API or disk
    python fl_methods/run_lr_batch.py -p Lang -v 1 3 --configs M1R1-M1R1-M1R1 --dry-run

    # Consume finished batches + finalize, but don't send a new round
    python fl_methods/run_lr_batch.py -p Lang --configs M1R1-M1R1-M1R1 --no-send

    # Show the local batch registry (refreshes remote statuses when a key is set)
    python fl_methods/run_lr_batch.py --status

Batch bookkeeping lives under ``data/batches/agent4lr/`` (registry
JSONs are committed; archived ``*.jsonl`` request/response files are
git-ignored). Ollama-provider configs are rejected — run those through
``run_lr.py``.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import run_lr
from run_experiment import EXPERIMENTS

from src.agent4lr.batch_registry import BatchRegistry
from src.agent4lr.batch_runner import TickReport, run_tick, validate_openai_only
from src.agent4lr.configs import load_lr_configs
from src.agent4lr.providers.openai_batch import (
    ACTIVE_BATCH_STATUSES,
    DEFAULT_COMPLETION_WINDOW,
    OpenAIBatchBackend,
)
from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import get_lr_batch_registry_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_lr_batch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Agent4LR batch tick against the OpenAI Batch API.",
    )
    parser.add_argument(
        "--benchmark",
        default="defects4j",
        choices=supported_benchmarks(),
        help="Benchmark adapter key (e.g. defects4j, bugsinpy).",
    )
    parser.add_argument(
        "-p",
        "--projects",
        nargs="+",
        default=None,
        help="Project names (space-separated). Required unless --status.",
    )
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Version IDs, applied to every listed project. "
            "Omit to run all versions of each project."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Named configs from fl_methods/configs/lr_configs.json "
            "(all slots must use the openai provider). Required unless "
            "--experiment or --status supplies them."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=None,
        choices=sorted(EXPERIMENTS.keys()),
        help=(
            "Named experiment from run_experiment.py: supplies its config grid "
            "(--configs override wins) and its default project list / bug count "
            "(-p and -v override). Implies --auto unless --ticks is given."
        ),
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help=(
            "Process up to this many ticks then stop, polling the Batch API "
            "between rounds (default 1). Stops early on convergence."
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Keep ticking (polling between rounds) until every (bug, config) is "
            "finalized. Overrides --ticks."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between Batch API status polls while waiting (default 120).",
    )
    parser.add_argument(
        "--sr-model-id",
        default=run_lr.DEFAULT_SR_MODEL_ID,
        help=(
            "model_id of the SR run whose top-20 file feeds Agent4LR "
            f"(default {run_lr.DEFAULT_SR_MODEL_ID!r})."
        ),
    )
    parser.add_argument(
        "--input",
        choices=("All", "bug_report", "trigger_test"),
        default="All",
        help="Which optional inputs to include alongside the candidate list.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--completion-window",
        default=DEFAULT_COMPLETION_WINDOW,
        help=f"Batch completion window (default {DEFAULT_COMPLETION_WINDOW!r}).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1000,
        help="Maximum requests per created batch (default 1000; API cap 50000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what the tick would do; no API calls, no writes.",
    )
    parser.add_argument(
        "--no-send",
        action="store_true",
        help="Process finished batches and finalize, but send no new batch.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the local batch registry (refreshing remote statuses) and exit.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _finalize_via_run_lr(openai_api_key: str | None) -> Any:
    """Bind run_lr.run_lr_for_bug as the tick's zero-LLM finalizer.

    A fully-checkpointed chain replays from cache without provider
    calls, so the Ollama transport args are inert placeholders.
    """

    def finalize(**kwargs: Any) -> dict[str, Any]:
        return run_lr.run_lr_for_bug(
            ollama_base_url="",
            ollama_verify=True,
            openai_api_key=openai_api_key,
            **kwargs,
        )

    return finalize


def _wait_for_batches(
    registry: BatchRegistry,
    backend: OpenAIBatchBackend,
    poll_interval: int,
) -> None:
    """Block until every unprocessed batch in the registry is terminal.

    Mirrors the poll loop used to drive the tick sequence by hand: refresh
    each unprocessed record's remote status and return once none remain in
    an active (non-terminal) state. Transient poll errors keep the loop
    running rather than aborting the sequence.
    """
    while True:
        pending = list(registry.unprocessed())
        if not pending:
            return
        statuses: dict[str, str] = {}
        for record in pending:
            try:
                remote = backend.retrieve_batch(record.batch_id)
                statuses[record.batch_id] = str(remote.get("status") or "?")
            except Exception as exc:  # transient network errors: keep polling
                statuses[record.batch_id] = f"pollerror:{exc}"
        active = [
            s for s in statuses.values() if s in ACTIVE_BATCH_STATUSES or s.startswith("pollerror")
        ]
        if not active:
            return
        log.info(
            "Waiting on %d batch(es): %s",
            len(pending),
            ", ".join(f"…{bid[-8:]}={s}" for bid, s in sorted(statuses.items())),
        )
        time.sleep(poll_interval)


def _tick_converged(report: TickReport, registry: BatchRegistry) -> bool:
    """True when a tick left nothing to do: no new round, nothing in flight."""
    return (
        not report.requests
        and not report.batches_created
        and not report.in_flight
        and not list(registry.unprocessed())
    )


def _print_status(registry: BatchRegistry, backend: OpenAIBatchBackend | None) -> None:
    records = registry.all_records()
    if not records:
        log.info("No batches in registry (%s)", registry.root)
        return
    for record in records:
        status = record.last_status
        if backend is not None and not record.processed and status in ACTIVE_BATCH_STATUSES:
            status = str(backend.retrieve_batch(record.batch_id).get("status") or status)
            if status != record.last_status:
                registry.mark(record.batch_id, last_status=status)
        log.info(
            "%s  status=%-12s processed=%-5s model=%-12s n_requests=%d",
            record.batch_id,
            status,
            record.processed,
            record.model,
            record.n_requests,
        )


def _print_report(report: TickReport) -> None:
    log.info("=" * 70)
    log.info("TICK SUMMARY")
    log.info("=" * 70)
    if report.processed_batches:
        log.info(
            "Processed batches: %s (outcomes: %s)",
            ", ".join(report.processed_batches),
            report.outcomes,
        )
    for project, bug_id, config_name in report.finalized:
        log.info("Finalized %s-%s [%s]", project, bug_id, config_name)
    for project, bug_id, config_name, reason in report.skipped:
        log.info("Skipped %s-%s [%s]: %s", project, bug_id, config_name, reason)
    if report.in_flight:
        log.info("In flight (not resent): %d requests", len(report.in_flight))
    for item in report.requests:
        log.info(
            "Request %s (model=%s, role=%s, configs=%s)",
            item.key.encode(),
            item.spec.model,
            item.spec.role,
            ",".join(item.config_names),
        )
    log.info(
        "Totals: %d finalized, %d skipped, %d in-flight, %d requests, %d batches created",
        len(report.finalized),
        len(report.skipped),
        len(report.in_flight),
        len(report.requests),
        len(report.batches_created),
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    registry = BatchRegistry(get_lr_batch_registry_dir())

    if args.status:
        backend: OpenAIBatchBackend | None
        try:
            backend = OpenAIBatchBackend(api_key=args.openai_api_key)
        except (ImportError, ValueError):
            backend = None
            log.info("No OpenAI key/SDK available — showing local registry state only.")
        _print_status(registry, backend)
        return 0

    dataset = get_benchmark_adapter(args.benchmark).benchmark_key
    adapter = get_benchmark_adapter(dataset)

    experiment = EXPERIMENTS[args.experiment] if args.experiment else None

    # Config grid: explicit --configs wins, else the experiment's grid.
    config_names = args.configs or (list(experiment.lr_configs) if experiment else None)
    # Project list: explicit -p wins, else the experiment's (None ⇒ all projects).
    projects = args.projects or (
        list(experiment.projects) if experiment and experiment.projects else None
    )
    if experiment is not None and projects is None:
        projects = list(adapter.list_projects())

    if not projects or not config_names:
        log.error(
            "-p/--projects and --configs are required "
            "(supply --experiment to source them, or --status)."
        )
        return 2

    all_configs = load_lr_configs()
    missing = [name for name in config_names if name not in all_configs]
    if missing:
        log.error("Unknown config(s) %s. Known: %s", missing, sorted(all_configs))
        return 2
    configs = {name: all_configs[name] for name in config_names}
    validate_openai_only(configs)

    bugs: list[tuple[str, str]] = []
    for project in projects:
        if args.versions is not None:
            bugs.extend((project, str(v)) for v in args.versions)
        else:
            cases = adapter.list_cases(project)
            # Mirror run_experiment: an experiment may cap bugs/project.
            if experiment is not None:
                cases = list(cases)[: experiment.n_bugs_per_project]
            bugs.extend((project, str(v)) for v in cases)

    # --auto (or --experiment without an explicit --ticks) runs to convergence;
    # otherwise process --ticks rounds (default 1). dry-run / no-send are always
    # a single pass — there is nothing to poll for.
    auto = args.auto or (experiment is not None and args.ticks is None)
    max_ticks = args.ticks if args.ticks is not None else 1
    if args.dry_run or args.no_send:
        auto, max_ticks = False, 1

    live_backend: OpenAIBatchBackend | None = None
    if not args.dry_run:
        live_backend = OpenAIBatchBackend(api_key=args.openai_api_key)

    tick_no = 0
    while True:
        tick_no += 1
        if auto or max_ticks > 1:
            log.info("--- tick %d%s ---", tick_no, "" if auto else f"/{max_ticks}")
        report = run_tick(
            dataset=dataset,
            bugs=bugs,
            configs=configs,
            sr_model_id=args.sr_model_id,
            input_keys=run_lr._input_keys(args.input),
            registry=registry,
            backend=live_backend,
            finalize_fn=None if args.dry_run else _finalize_via_run_lr(args.openai_api_key),
            max_requests=args.max_requests,
            completion_window=args.completion_window,
            dry_run=args.dry_run,
            no_send=args.no_send,
        )
        _print_report(report)

        if args.dry_run or args.no_send:
            break
        if _tick_converged(report, registry):
            log.info("Converged after %d tick(s): all (bug, config) results finalized.", tick_no)
            break
        if not auto and tick_no >= max_ticks:
            log.info("Reached --ticks limit (%d); stopping with work still in flight.", max_ticks)
            break
        assert live_backend is not None  # not dry_run ⇒ backend exists
        _wait_for_batches(registry, live_backend, args.poll_interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
