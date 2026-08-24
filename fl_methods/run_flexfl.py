#!/usr/bin/env python3
"""Run the full FlexFL pipeline (SR + LR) for one or more bugs.

Composes :mod:`run_sr` followed by :mod:`run_lr`. SR produces the
top-20 candidate list and the surrounding extraction artifacts; LR
consumes that top-20 plus the cleaned trigger test + bug report to
emit the final top-5.

Usage::

    python run_flexfl.py -p Lang -v 1 \
        --sr-model llama3.1:8b --sr-base-url https://... --sr-no-verify \
        --lr-config M1R1-M1R1-M1R1

The SR phase's model_id is derived from ``_model_slug(--sr-model)`` and
forwarded to LR as ``sr_model_id`` so the two halves stay linked. Pass
``--sr-model-id`` to override the derived slug (needed only when the
tracker has multiple SR runs and the suffix differs).
"""

from __future__ import annotations

import argparse
import logging
import time

import run_lr
import run_sr

from src.agent4lr.configs import (
    PerRoleOverrides,
    load_lr_configs,
    resolve_config,
)
from src.agent4sr.agent import SRConfig
from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.config import _model_slug

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_flexfl")


def _add_lr_per_role_flags(group: argparse._ArgumentGroup, role: str) -> None:
    """Mirror :func:`run_lr._add_per_role_flags` under an ``--lr-`` prefix."""
    role_flag = role.replace("_", "-")
    group.add_argument(
        f"--lr-{role_flag}-provider",
        choices=("ollama", "openai"),
        dest=f"lr_{role}_provider",
        default=None,
    )
    group.add_argument(f"--lr-{role_flag}-model", dest=f"lr_{role}_model", default=None)
    group.add_argument(
        f"--lr-{role_flag}-temperature",
        dest=f"lr_{role}_temperature",
        type=float,
        default=None,
    )
    group.add_argument(f"--lr-{role_flag}-top-p", dest=f"lr_{role}_top_p", type=float, default=None)
    group.add_argument(
        f"--lr-{role_flag}-reasoning-effort",
        dest=f"lr_{role}_reasoning_effort",
        default=None,
    )
    if role == "tool_caller":
        group.add_argument(
            "--lr-tool-caller-iterations",
            dest="lr_tool_caller_iterations",
            type=int,
            default=None,
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the composed driver.

    Flags are namespaced: SR options carry ``--sr-`` prefix, LR options
    ``--lr-``. Shared options (``-p``, ``-v``, ``--force``) live
    without a prefix.
    """
    parser = argparse.ArgumentParser(
        description="Run the full FlexFL pipeline (SR + LR) end-to-end.",
    )
    parser.add_argument("-d", "--dataset", default="defects4j")
    parser.add_argument("-p", "--project", required=True)
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Version IDs to process (omit to run all versions).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if outputs exist.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--sr-model-id",
        default=None,
        help="Override sr_model_id forwarded to LR. Defaults to _model_slug(--sr-model).",
    )

    # SR phase flags (mirrors run_sr.py)
    sr = parser.add_argument_group("SR phase (Agent4SR)")
    sr.add_argument("--sr-model", default="llama3.1:8b")
    sr.add_argument("--sr-base-url", default="http://localhost:11434")
    sr.add_argument("--sr-no-verify", action="store_true")
    sr.add_argument("--sr-iterations", type=int, default=10)
    sr.add_argument("--sr-temperature", type=float, default=0.0)
    sr.add_argument("--use-ce", action="store_true", help="Use CE Monte Carlo for RAFL")
    sr.add_argument("--ce-max-iter", type=int, default=1000)
    sr.add_argument("--ce-pop-size", type=int, default=100)
    sr.add_argument(
        "--keep-checkouts",
        action="store_true",
        help="Do not remove the checkout after the SR pipeline finishes.",
    )

    # LR phase: --lr-config XOR per-role flags
    lr_grp = parser.add_argument_group("LR phase (Agent4LR)")
    lr_grp.add_argument(
        "--lr-config",
        dest="lr_config_name",
        default=None,
        help="Named LR configuration from fl_methods/configs/lr_configs.json.",
    )
    lr_grp.add_argument(
        "--lr-input",
        choices=("All", "bug_report", "trigger_test"),
        default="All",
    )
    lr_grp.add_argument(
        "--lr-ollama-base-url",
        default="http://localhost:11434",
        help="Ollama URL for LR slots whose provider is 'ollama'.",
    )
    lr_grp.add_argument("--lr-no-verify", action="store_true")
    lr_grp.add_argument(
        "--lr-openai-api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var).",
    )
    _add_lr_per_role_flags(lr_grp, "planner")
    _add_lr_per_role_flags(lr_grp, "tool_caller")
    _add_lr_per_role_flags(lr_grp, "finisher")

    return parser


def _lr_overrides(args: argparse.Namespace) -> PerRoleOverrides:
    return PerRoleOverrides(
        planner_provider=args.lr_planner_provider,
        planner_model=args.lr_planner_model,
        planner_temperature=args.lr_planner_temperature,
        planner_top_p=args.lr_planner_top_p,
        planner_reasoning_effort=args.lr_planner_reasoning_effort,
        tool_caller_provider=args.lr_tool_caller_provider,
        tool_caller_model=args.lr_tool_caller_model,
        tool_caller_iterations=args.lr_tool_caller_iterations,
        tool_caller_temperature=args.lr_tool_caller_temperature,
        tool_caller_top_p=args.lr_tool_caller_top_p,
        tool_caller_reasoning_effort=args.lr_tool_caller_reasoning_effort,
        finisher_provider=args.lr_finisher_provider,
        finisher_model=args.lr_finisher_model,
        finisher_temperature=args.lr_finisher_temperature,
        finisher_top_p=args.lr_finisher_top_p,
        finisher_reasoning_effort=args.lr_finisher_reasoning_effort,
    )


def main() -> int:
    """Drive SR then LR for each requested bug.

    Returns 0 on success, 2 if any bug's SR or LR phase failed.
    """
    args = build_parser().parse_args()

    if args.dataset not in supported_benchmarks():
        raise NotImplementedError(
            f"Dataset {args.dataset!r} not supported. Supported: {supported_benchmarks()}"
        )
    if args.dataset != "defects4j":
        raise NotImplementedError(
            f"Dataset {args.dataset!r} is not supported yet; only 'defects4j'."
        )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    overrides = _lr_overrides(args)
    lr_config_name, lr_config = resolve_config(
        config_name=args.lr_config_name,
        overrides=overrides,
        configs=load_lr_configs(),
    )

    sr_model_id = args.sr_model_id or _model_slug(args.sr_model)
    lr_input_keys = run_lr._input_keys(args.lr_input)

    if args.versions is not None:
        bugs = [(args.project, str(v)) for v in args.versions]
    else:
        adapter = get_benchmark_adapter(args.dataset)
        bugs = [(args.project, str(v)) for v in adapter.list_cases(args.project)]

    t_global = time.time()
    summaries: list[dict] = []
    for project, bug_id in bugs:
        log.info("=" * 70)
        log.info("Bug %s-%s: SR phase", project, bug_id)
        log.info("=" * 70)
        sr_cfg = SRConfig.from_cli(
            model=args.sr_model,
            iterations=args.sr_iterations,
            temperature=args.sr_temperature,
            base_url=args.sr_base_url,
            verify=not args.sr_no_verify,
            bugs=[bug_id],
        )
        sr_summary = run_sr.run_single_bug(
            project,
            bug_id,
            dataset=args.dataset,
            sr_cfg=sr_cfg,
            use_ce=args.use_ce,
            ce_max_iter=args.ce_max_iter,
            ce_pop_size=args.ce_pop_size,
            skip_existing=not args.force,
            keep_checkouts=args.keep_checkouts,
        )
        sr_failed = any(
            stage.get("status") == "FAILED" for stage in sr_summary.get("stages", {}).values()
        )
        log.info("=" * 70)
        log.info(
            "Bug %s-%s: LR phase (config=%s, sr_model_id=%s)",
            project,
            bug_id,
            lr_config_name,
            sr_model_id,
        )
        log.info("=" * 70)
        if sr_failed:
            log.warning("SR failed for %s-%s; skipping LR.", project, bug_id)
            summaries.append(
                {
                    "project": project,
                    "bug_id": bug_id,
                    "sr": sr_summary,
                    "lr": {"status": "SKIPPED", "reason": "SR failed"},
                }
            )
            continue
        try:
            lr_summary = run_lr.run_lr_for_bug(
                project=project,
                bug_id=bug_id,
                config_name=lr_config_name,
                config=lr_config,
                sr_model_id=sr_model_id,
                input_keys=lr_input_keys,
                dataset=args.dataset,
                ollama_base_url=args.lr_ollama_base_url,
                ollama_verify=not args.lr_no_verify,
                openai_api_key=args.lr_openai_api_key,
            )
        except Exception as exc:  # pragma: no cover
            log.exception("Unhandled LR error for %s-%s", project, bug_id)
            lr_summary = {"status": "FAILED", "error": str(exc)}
        summaries.append({"project": project, "bug_id": bug_id, "sr": sr_summary, "lr": lr_summary})

    log.info("")
    log.info("=" * 70)
    log.info("FlexFL FINAL SUMMARY (total %.1fs)", time.time() - t_global)
    log.info("=" * 70)
    had_error = False
    for s in summaries:
        sr_stages = s.get("sr", {}).get("stages", {})
        sr_status = (
            "FAILED" if any(st.get("status") == "FAILED" for st in sr_stages.values()) else "OK"
        )
        lr_status = s.get("lr", {}).get("status", "?")
        if sr_status == "FAILED" or lr_status == "FAILED":
            had_error = True
        log.info(
            "%s-%s  SR=%s  LR=%s  lr_model_id=%s",
            s["project"],
            s["bug_id"],
            sr_status,
            lr_status,
            s.get("lr", {}).get("lr_model_id", "?"),
        )

    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
