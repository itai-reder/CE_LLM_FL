#!/usr/bin/env python3
"""Run the CEFL Localization-Refinement (LR) pipeline for one or more bugs.

Reads SR outputs (``FlexFL/SR/rankings/top20/<sr_model_id>.txt``) plus
the cleaned trigger test and bug report, runs Agent4LR
(planner → tool loop → structured-rank finisher) for one **multi-agent
configuration**, and writes ``FlexFL/LR/Agent4LR/<lr_model_id>/`` plus
``rankings/top5/<lr_model_id>.*`` files.

Configurations are either named (loaded from
``fl_methods/configs/lr_configs.json``) or assembled inline from
per-role CLI flags. The two modes are mutually exclusive.

Usage::

    # Named config
    python run_lr.py -p Lang -v 1 --config ollama_llama3.1_8b_x3

    # Ad-hoc, per-role
    python run_lr.py -p Lang -v 1 \\
        --planner-provider openai --planner-model gpt-4.1-mini \\
        --tool-caller-provider ollama --tool-caller-model llama3.1:8b \\
            --tool-caller-iterations 10 \\
            --ollama-base-url https://cis-ollama.auth.ad.bgu.ac.il --no-verify \\
        --finisher-provider openai --finisher-model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback

from src.agent4lr.agent import run_agent4lr_for_bug
from src.agent4lr.configs import (
    ADHOC_CONFIG_NAME,
    PerRoleOverrides,
    load_lr_configs,
    resolve_config,
)
from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.bip_gate import bip_run_skip_reason
from src.common.config import get_lr_candidate_file
from src.common.rankings import generate_top5
from src.common.tracker import (
    TrackerStep,
    get_or_assign_lr_model_id,
    load_tracker,
    save_tracker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_lr")

DEFAULT_SR_MODEL_ID = "llama3.1_8b"


def _add_per_role_flags(group: argparse._ArgumentGroup, role: str) -> None:
    """Add a uniform set of per-role override flags for one slot.

    Flag names are ``--<role>-provider``, ``--<role>-model``,
    ``--<role>-temperature``, ``--<role>-top-p``,
    ``--<role>-reasoning-effort`` (OpenAI o-series / gpt-5 only) and —
    for ``tool_caller`` — ``--tool-caller-iterations``.
    """
    role_flag = role.replace("_", "-")
    dest_prefix = role
    group.add_argument(
        f"--{role_flag}-provider",
        choices=("ollama", "openai"),
        dest=f"{dest_prefix}_provider",
        default=None,
    )
    group.add_argument(f"--{role_flag}-model", dest=f"{dest_prefix}_model", default=None)
    group.add_argument(
        f"--{role_flag}-temperature",
        dest=f"{dest_prefix}_temperature",
        type=float,
        default=None,
    )
    group.add_argument(
        f"--{role_flag}-top-p",
        dest=f"{dest_prefix}_top_p",
        type=float,
        default=None,
    )
    group.add_argument(
        f"--{role_flag}-reasoning-effort",
        dest=f"{dest_prefix}_reasoning_effort",
        default=None,
        help="OpenAI o-series / gpt-5 only.",
    )
    if role == "tool_caller":
        group.add_argument(
            "--tool-caller-iterations",
            dest="tool_caller_iterations",
            type=int,
            default=None,
            help="Max ReAct iterations for the tool_caller slot.",
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the LR CLI.

    Required: one of ``--config <name>`` or per-role flags (mutually
    exclusive). Plus the usual ``-p`` / ``-v`` selectors and the
    SR-link ``--sr-model-id`` (defaults to
    :data:`DEFAULT_SR_MODEL_ID`).

    Ollama provider also needs ``--ollama-base-url`` (and optionally
    ``--no-verify`` for self-signed TLS); OpenAI relies on the standard
    ``OPENAI_API_KEY`` env var unless ``--openai-api-key`` is given.
    """
    parser = argparse.ArgumentParser(
        description="Run the CEFL Localization-Refinement (LR) pipeline.",
    )

    # Bug selectors and SR link.
    # --benchmark is authoritative (matches run_sbir.py/run_boostn.py/run_agent4sr.py);
    # -d/--dataset is a deprecated alias honored only when --benchmark is left at default.
    parser.add_argument(
        "--benchmark",
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
    parser.add_argument("-p", "--project", required=True, help="Project name")
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Version IDs to process (space-separated). Omit to run all versions.",
    )
    parser.add_argument(
        "--sr-model-id",
        default=DEFAULT_SR_MODEL_ID,
        help=(
            f"model_id of the SR run whose top-20 file feeds Agent4LR. "
            f"Used to locate FlexFL/SR/rankings/top20/<sr_model_id>.txt. "
            f"Defaults to {DEFAULT_SR_MODEL_ID!r}."
        ),
    )

    # Config selection: --config XOR per-role flags
    config_grp = parser.add_argument_group("Configuration (mutually exclusive)")
    config_grp.add_argument(
        "--config",
        dest="config_name",
        default=None,
        help=(
            "Name of a configuration in fl_methods/configs/lr_configs.json. "
            "Mutually exclusive with per-role flags."
        ),
    )

    planner_grp = parser.add_argument_group("Per-role: planner")
    _add_per_role_flags(planner_grp, "planner")
    loop_grp = parser.add_argument_group("Per-role: tool_caller")
    _add_per_role_flags(loop_grp, "tool_caller")
    finisher_grp = parser.add_argument_group("Per-role: finisher")
    _add_per_role_flags(finisher_grp, "finisher")

    # Provider auth/transport (shared across slots)
    auth_grp = parser.add_argument_group("Provider transport")
    auth_grp.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL. Used by any slot whose provider is 'ollama'.",
    )
    auth_grp.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable TLS cert verification (Ollama only).",
    )
    auth_grp.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var).",
    )

    # Inputs and misc
    parser.add_argument(
        "--input",
        choices=("All", "bug_report", "trigger_test"),
        default="All",
        help="Which optional inputs to include alongside the candidate list.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist on disk.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> PerRoleOverrides:
    """Materialise per-role CLI flags into a :class:`PerRoleOverrides`."""
    return PerRoleOverrides(
        planner_provider=args.planner_provider,
        planner_model=args.planner_model,
        planner_temperature=args.planner_temperature,
        planner_top_p=args.planner_top_p,
        planner_reasoning_effort=args.planner_reasoning_effort,
        tool_caller_provider=args.tool_caller_provider,
        tool_caller_model=args.tool_caller_model,
        tool_caller_iterations=args.tool_caller_iterations,
        tool_caller_temperature=args.tool_caller_temperature,
        tool_caller_top_p=args.tool_caller_top_p,
        tool_caller_reasoning_effort=args.tool_caller_reasoning_effort,
        finisher_provider=args.finisher_provider,
        finisher_model=args.finisher_model,
        finisher_temperature=args.finisher_temperature,
        finisher_top_p=args.finisher_top_p,
        finisher_reasoning_effort=args.finisher_reasoning_effort,
    )


def _input_keys(arg: str) -> tuple[str, ...]:
    """Map the ``--input`` flag to the LRConfig.input_keys tuple."""
    if arg == "All":
        return ("bug_report", "trigger_test")
    return (arg,)


def run_lr_for_bug(
    *,
    project: str,
    bug_id: str,
    config_name: str,
    config: tuple,
    sr_model_id: str,
    input_keys: tuple[str, ...],
    dataset: str,
    ollama_base_url: str,
    ollama_verify: bool,
    openai_api_key: str | None,
) -> dict:
    """Run LR for one bug end-to-end (tracker + agent + rankings).

    BugsInPy bugs are gated on extraction readiness (audit status + SR corpus) **and**
    the presence of the SR top-20 candidate list before any LLM work; un-ready bugs
    return a ``SKIPPED`` summary. The tracker is dataset-aware so ``tracker.json``
    lands under the bug's processed dir.
    """
    summary: dict = {"project": project, "bug_id": bug_id, "config_name": config_name}
    candidate_source = f"FlexFL/SR/rankings/top20/{sr_model_id}.txt"

    # --- BugsInPy readiness gate (audit + SR corpus + SR top-20 present) ---
    if dataset == "bugsinpy":
        skip_reason = bip_run_skip_reason(project, bug_id)
        if skip_reason is None:
            top20 = get_lr_candidate_file(project, bug_id, sr_model_id=sr_model_id, dataset=dataset)
            if not top20.exists():
                skip_reason = f"SR top-20 missing ({candidate_source})"
        if skip_reason is not None:
            log.warning("Skipping %s-%s: %s", project, bug_id, skip_reason)
            summary["status"] = "SKIPPED"
            summary["skip_reason"] = skip_reason
            return summary

    tracker = load_tracker(project, bug_id, dataset=dataset)
    lr_model_id = get_or_assign_lr_model_id(
        tracker,
        config_name=config_name,
        agent_chain=[a.identity() for a in config],
        sr_model_id=sr_model_id,
        candidate_source=candidate_source,
        input_keys=list(input_keys),
    )
    save_tracker(tracker, project, bug_id, dataset=dataset)
    summary["lr_model_id"] = lr_model_id

    with TrackerStep(
        project, bug_id, section="lr", step="run", model_id=lr_model_id, dataset=dataset
    ) as ts:
        try:
            t0 = time.time()
            result = run_agent4lr_for_bug(
                project=project,
                bug_id=bug_id,
                config_name=config_name,
                config=config,
                sr_model_id=sr_model_id,
                input_keys=input_keys,
                dataset=dataset,
                lr_model_id=lr_model_id,
                ollama_base_url=ollama_base_url,
                ollama_verify=ollama_verify,
                openai_api_key=openai_api_key,
            )
            summary["status"] = "SKIPPED" if result is None else "OK"
            summary["time_s"] = round(time.time() - t0, 1)
            if result is not None:
                summary["top5_count"] = len(result.top5)
        except Exception as exc:
            log.exception("Agent4LR run failed for %s-%s", project, bug_id)
            ts.record_error(repr(exc))
            summary["status"] = "FAILED"
            summary["error"] = str(exc)
            return summary

    try:
        generate_top5(project, bug_id, lr_model_id=lr_model_id, dataset=dataset)
    except Exception as exc:
        log.exception("generate_top5 failed for %s-%s", project, bug_id)
        summary.setdefault("error", str(exc))
        summary["status"] = "FAILED"

    return summary


def main() -> int:
    """Drive the LR pipeline for each requested bug.

    Steps:

    1. Parse args; validate the ``--config`` XOR per-role-flags mutual
       exclusion via :func:`src.agent4lr.configs.resolve_config`.
    2. For each requested bug, allocate ``lr_model_id`` via the
       tracker, then run Agent4LR inside a
       ``TrackerStep(section="lr")``.
    3. Generate ``rankings/top5/<lr_model_id>.{txt,csv}``.
    4. Sync the processed dir back to ``$CEFL_ROOT``.

    Returns 0 on success, 2 if any bug failed.
    """
    args = build_parser().parse_args()

    # --benchmark is authoritative; -d/--dataset is a legacy alias honored only when
    # --benchmark was left at its default (so old `-d <bench>` invocations keep working).
    # Normalize to the adapter's canonical benchmark_key and store it back on args.dataset.
    benchmark_key = args.benchmark
    if args.benchmark == "defects4j" and args.dataset != "defects4j":
        benchmark_key = args.dataset
    if benchmark_key not in supported_benchmarks():
        raise NotImplementedError(
            f"Benchmark {benchmark_key!r} not supported. Supported: {supported_benchmarks()}"
        )
    args.dataset = get_benchmark_adapter(benchmark_key).benchmark_key

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    overrides = _overrides_from_args(args)
    config_name, config = resolve_config(
        config_name=args.config_name,
        overrides=overrides,
        configs=load_lr_configs(),
    )
    log.info("Resolved config: %s (chain=%s)", config_name, [a.role for a in config])

    if args.versions is not None:
        bugs = [(args.project, str(v)) for v in args.versions]
    else:
        adapter = get_benchmark_adapter(args.dataset)
        bugs = [(args.project, str(v)) for v in adapter.list_cases(args.project)]

    input_keys = _input_keys(args.input)
    t_global = time.time()
    summaries: list[dict] = []
    for project, bug_id in bugs:
        log.info("=" * 70)
        log.info("Processing %s-%s with config %s", project, bug_id, config_name)
        log.info("=" * 70)
        try:
            summary = run_lr_for_bug(
                project=project,
                bug_id=bug_id,
                config_name=config_name,
                config=config,
                sr_model_id=args.sr_model_id,
                input_keys=input_keys,
                dataset=args.dataset,
                ollama_base_url=args.ollama_base_url,
                ollama_verify=not args.no_verify,
                openai_api_key=args.openai_api_key,
            )
        except Exception as exc:  # pragma: no cover — surface unexpected failures
            log.error(
                "Unhandled error processing %s-%s: %s\n%s",
                project,
                bug_id,
                exc,
                traceback.format_exc(),
            )
            summary = {"project": project, "bug_id": bug_id, "status": "FAILED", "error": str(exc)}
        summaries.append(summary)

    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY (config=%s, total %.1fs)", config_name, time.time() - t_global)
    log.info("=" * 70)
    had_error = False
    for s in summaries:
        status = s.get("status", "N/A")
        log.info(
            "%s-%s  %-8s  lr_model_id=%s  top5=%s",
            s.get("project"),
            s.get("bug_id"),
            status,
            s.get("lr_model_id", "?"),
            s.get("top5_count", "?"),
        )
        if status == "FAILED":
            had_error = True

    # Suppress unused-import warning for ADHOC_CONFIG_NAME (documentation aid).
    _ = ADHOC_CONFIG_NAME
    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
