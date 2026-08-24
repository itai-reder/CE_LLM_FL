"""CLI entry point for Agent4SR: corpus generation, SR runs, and combine.

Usage::

    python -m fl_methods.run_agent4sr corpus -p Chart -v 1
    python -m fl_methods.run_agent4sr run -p Chart -v 1 2 3
    python -m fl_methods.run_agent4sr combine -p Chart -v 1 2 3
    python -m fl_methods.run_agent4sr smoke --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from typing import Any, cast

import requests
from update_tracker import update_single_bug
from urllib3.exceptions import InsecureRequestWarning

from src.agent4sr.agent import SRConfig, run_agent4sr_for_bug
from src.agent4sr.combine import write_candidates
from src.agent4sr.corpus import save_corpus
from src.agent4sr.io import load_bug_inputs
from src.agent4sr.tools import tool_schemas
from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.common.bip_gate import (
    bip_corpus_exists as _bip_corpus_exists,
)
from src.common.bip_gate import (
    bip_run_skip_reason as _bip_run_skip_reason,
)
from src.common.bip_gate import (
    classify_bip_bug as _bip_classify,
)
from src.common.config import (
    get_processed_dir,
    get_rankings_dir,
    get_sr_model_dir,
)
from src.common.rankings import generate_all_rankings, generate_top15, generate_top20
from src.common.tracker import (
    TrackerStep,
    get_or_assign_sr_model_id,
    load_tracker,
    save_tracker,
)
from src.extraction.bugsinpy import BugsInPyRepo
from src.extraction.d4j import D4JRepo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_project(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Validate that --project was provided; exit with error if not."""
    if not args.project:
        parser.error("the following arguments are required: -p/--project")
    return cast(str, args.project)


def _resolve_versions(args: argparse.Namespace) -> list[str]:
    """Resolve version IDs from CLI args."""
    if args.versions is not None:
        return [str(v) for v in args.versions]
    dataset = getattr(args, "dataset", "defects4j")
    adapter = get_benchmark_adapter(dataset)
    return [str(v) for v in adapter.list_cases(args.project)]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_corpus(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Dispatch corpus generation based on ``args.dataset`` (canonical benchmark key)."""
    _require_project(parser, args)
    dataset = getattr(args, "dataset", "defects4j")
    if dataset == "defects4j":
        return cmd_corpus_d4j(args, parser)
    if dataset == "bugsinpy":
        return cmd_corpus_bugsinpy(args, parser)
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def cmd_corpus_d4j(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Generate corpus files for the specified Defects4J bugs."""
    bugs = _resolve_versions(args)
    had_error = False
    for bug_id in bugs:
        try:
            bug_num = int(bug_id)
        except ValueError:
            logger.error("Invalid bug ID %r: expected an integer", bug_id)
            had_error = True
            continue

        logger.info("Generating corpus for %s-%s", args.project, bug_id)
        repo = D4JRepo(args.project, bug_num)

        try:
            # Match extraction behavior: ensure checkout exists before corpus parsing.
            repo.checkout(skip_existing=True)
            repo.export_property("dir.src.classes")
            save_corpus(
                args.project,
                bug_id,
                skip_existing=not args.force,
                dataset="defects4j",
            )
        except Exception:
            had_error = True
            logger.exception("Failed generating corpus for %s-%s", args.project, bug_id)
        finally:
            if not args.keep_checkouts:
                try:
                    repo.remove_repo()
                except Exception:
                    had_error = True
                    logger.exception(
                        "Failed to clean up checkout for %s-%s",
                        args.project,
                        bug_id,
                    )

    return 2 if had_error else 0


def cmd_corpus_bugsinpy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Generate corpus files for valid BugsInPy bugs.

    Hard valid-bug gate: only bugs whose prior-step extraction
    outputs validate (audit status in ``COMPLETE_STATUSES``) get a corpus. Every
    excluded bug is logged with its status/reason — never an empty corpus for a bug
    that did not extract. The source is checked out on demand (repos are not synced)
    and removed afterwards unless ``--keep-checkouts``.
    """
    bugs = _resolve_versions(args)
    had_error = False
    n_generated = 0
    n_skipped = 0
    for bug_id in bugs:
        try:
            bug_num = int(bug_id)
        except ValueError:
            logger.error("Invalid bug ID %r: expected an integer", bug_id)
            had_error = True
            continue

        # --- Valid-bug gate (reads on-disk outputs only; no checkout needed) ---
        result, skip_reason = _bip_classify(args.project, bug_num)
        if skip_reason is not None:
            logger.warning(
                "Skipping %s-%s: extraction not corpus-ready (%s)",
                args.project,
                bug_id,
                skip_reason,
            )
            n_skipped += 1
            continue

        # Resume cheaply: if the corpus already exists and we are not forcing, skip
        # before the (expensive) checkout.
        if not args.force and _bip_corpus_exists(args.project, bug_num):
            logger.info(
                "Skipping %s-%s: corpus already exists (use --force to overwrite)",
                args.project,
                bug_id,
            )
            n_skipped += 1
            continue

        status = getattr(result, "status", "unknown")
        logger.info("Generating corpus for %s-%s (audit: %s)", args.project, bug_id, status)
        repo = BugsInPyRepo(args.project, bug_num)
        try:
            # Repos are not synced; check out + compile on demand so get_src_dir
            # resolves the import root (e.g. ansible's build/lib). Idempotent.
            repo.checkout(skip_existing=True)
            repo.compile(skip_existing=True)
            save_corpus(
                args.project,
                bug_id,
                skip_existing=not args.force,
                dataset="bugsinpy",
            )
            n_generated += 1
        except Exception:
            had_error = True
            logger.exception("Failed generating corpus for %s-%s", args.project, bug_id)
        finally:
            if not args.keep_checkouts:
                try:
                    repo.remove_repo()
                except Exception:
                    had_error = True
                    logger.exception("Failed to clean up checkout for %s-%s", args.project, bug_id)

    logger.info(
        "BugsInPy corpus: %d generated, %d skipped (of %d requested)",
        n_generated,
        n_skipped,
        len(bugs),
    )
    return 2 if had_error else 0


def cmd_smoke(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Smoke-test Ollama tool-calling connectivity."""
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Call the exit tool now."}],
        "tools": tool_schemas()[-1:],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        if args.no_verify:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InsecureRequestWarning)
                resp = requests.post(
                    f"{args.base_url}/api/chat",
                    json=payload,
                    timeout=120,
                    verify=False,
                )
        else:
            resp = requests.post(
                f"{args.base_url}/api/chat",
                json=payload,
                timeout=120,
                verify=True,
            )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Smoke test failed: %s", exc)
        return 2
    data = resp.json()
    tool_calls = (data.get("message", {}) or {}).get("tool_calls")
    if not tool_calls:
        logger.error("SMOKE_FAIL: no tool_calls returned")
        return 2
    logger.info("SMOKE_OK")
    return 0


def cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Dispatch the Agent4SR run pipeline based on ``args.dataset``."""
    _require_project(parser, args)
    dataset = getattr(args, "dataset", "defects4j")
    if dataset == "defects4j":
        return _cmd_run_d4j(args, parser)
    if dataset == "bugsinpy":
        return _cmd_run_bugsinpy(args, parser)
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def _cmd_run_d4j(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the Agent4SR pipeline for the specified Defects4J bugs."""
    bugs = _resolve_versions(args)
    cfg = SRConfig.from_cli(
        model=args.model,
        iterations=args.iterations,
        temperature=args.temperature,
        base_url=args.base_url,
        verify=not args.no_verify,
        bugs=bugs,
    )
    skip_existing = not getattr(args, "force", False)

    for bug_id in cfg.bugs:
        processed_dir = get_processed_dir(args.project, bug_id, dataset="defects4j")
        inferred_tracker = cast(
            dict[str, Any],
            update_single_bug(
                processed_dir, args.project, int(bug_id), dry_run=True, no_backup=True
            ),
        )

        # Resolve model_id via tracker
        tracker = load_tracker(args.project, bug_id)
        model_id = get_or_assign_sr_model_id(
            tracker,
            model=cfg.model,
            temperature=cfg.temperature,
            iterations=cfg.iterations,
            base_url=cfg.base_url,
            input_keys=["bug_report", "ochiai", "boostn", "sbir"],
        )
        save_tracker(tracker, args.project, bug_id)

        if skip_existing and "run" in inferred_tracker["sr"].get(model_id, {}).get("completed", []):
            logger.info("Agent4SR run for model %s already complete, skipping.", model_id)
            continue

        with TrackerStep(args.project, bug_id, section="sr", step="run", model_id=model_id) as ts:
            try:
                run_agent4sr_for_bug(
                    project=args.project,
                    bug_id=bug_id,
                    cfg=cfg,
                    dataset="defects4j",
                    model_id=model_id,
                )
            except Exception as exc:
                ts.record_error(repr(exc))
                raise
    return 0


def _cmd_run_bugsinpy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the Agent4SR pipeline for valid BugsInPy bugs.

    Mirrors the Defects4J path but: (1) no ``update_single_bug`` (the D4J static analyzer
    expects GZoltar) — skip-existing is inferred from the SR result on disk; (2) a hard
    valid-bug gate (audit status + corpus present) plus a blank-trigger skip. The corpus,
    bug report, and trigger are already on disk from extraction, so no checkout is needed.
    The tracker is still used (it is dataset-aware) so ``tracker.json`` lands under
    ``data/BIP/``.
    """
    bugs = _resolve_versions(args)
    cfg = SRConfig.from_cli(
        model=args.model,
        iterations=args.iterations,
        temperature=args.temperature,
        base_url=args.base_url,
        verify=not args.no_verify,
        bugs=bugs,
    )
    skip_existing = not getattr(args, "force", False)
    n_run = 0
    n_skipped = 0

    for bug_id in cfg.bugs:
        # --- Valid-bug gate (audit status + SR corpus present) ---
        skip_reason = _bip_run_skip_reason(args.project, bug_id)
        if skip_reason is not None:
            logger.warning("Skipping %s-%s: %s", args.project, bug_id, skip_reason)
            n_skipped += 1
            continue

        # --- Blank-trigger skip: never run the LLM on an empty trigger ---
        inputs = load_bug_inputs(args.project, bug_id, dataset="bugsinpy")
        if not inputs.trigger_test.strip():
            logger.warning(
                "Skipping %s-%s: blank trigger test (no trigger_test_clean.txt / trigger_tests)",
                args.project,
                bug_id,
            )
            n_skipped += 1
            continue

        # Resolve model_id via the (dataset-aware) tracker so tracker.json lands under BIP.
        tracker = load_tracker(args.project, bug_id, dataset="bugsinpy")
        model_id = get_or_assign_sr_model_id(
            tracker,
            model=cfg.model,
            temperature=cfg.temperature,
            iterations=cfg.iterations,
            base_url=cfg.base_url,
            input_keys=["bug_report", "ochiai", "boostn", "sbir"],
        )
        save_tracker(tracker, args.project, bug_id, dataset="bugsinpy")

        # Skip-existing from disk (no update_single_bug for BIP).
        sr_result = (
            get_sr_model_dir(args.project, bug_id, cfg.model, dataset="bugsinpy", model_id=model_id)
            / "sr_result.json"
        )
        if skip_existing and sr_result.exists():
            logger.info("Agent4SR run for model %s already complete, skipping.", model_id)
            n_skipped += 1
            continue

        with TrackerStep(
            args.project, bug_id, section="sr", step="run", model_id=model_id, dataset="bugsinpy"
        ) as ts:
            try:
                run_agent4sr_for_bug(
                    project=args.project,
                    bug_id=bug_id,
                    cfg=cfg,
                    dataset="bugsinpy",
                    model_id=model_id,
                )
            except Exception as exc:
                ts.record_error(repr(exc))
                raise
        n_run += 1

    logger.info(
        "BugsInPy Agent4SR run: %d run, %d skipped (of %d requested)",
        n_run,
        n_skipped,
        len(cfg.bugs),
    )
    return 0


def cmd_combine(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Dispatch the combine/ranking stage based on ``args.dataset``."""
    _require_project(parser, args)
    dataset = getattr(args, "dataset", "defects4j")
    if dataset == "defects4j":
        return _cmd_combine_d4j(args, parser)
    if dataset == "bugsinpy":
        return _cmd_combine_bugsinpy(args, parser)
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def _cmd_combine_d4j(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Generate standardised rankings and combine FL + SR results for Defects4J."""
    bugs = _resolve_versions(args)
    skip_existing = not getattr(args, "force", False)

    for bug_id in bugs:
        processed_dir = get_processed_dir(args.project, bug_id, dataset="defects4j")
        inferred_tracker = cast(
            dict[str, Any],
            update_single_bug(
                processed_dir, args.project, int(bug_id), dry_run=True, no_backup=True
            ),
        )

        if skip_existing and "top15" in inferred_tracker["fl"]["completed"]:
            logger.info("Combine outputs already exist, skipping.")
            continue

        # Resolve model_id from tracker (if available)
        tracker = load_tracker(args.project, bug_id)
        model_id = get_or_assign_sr_model_id(
            tracker,
            model=args.model,
            temperature=0.0,
            iterations=10,
            base_url="",
            input_keys=["bug_report", "ochiai", "boostn", "sbir"],
        )
        save_tracker(tracker, args.project, bug_id)

        with TrackerStep(args.project, bug_id, section="fl", step="top15") as ts:
            try:
                # Generate method-level ranking CSVs (ochiai, sbir, boostn)
                rankings = generate_all_rankings(args.project, bug_id, dataset="defects4j")
                # Generate combined top15 (SBIR + Ochiai + BoostN top 5 each)
                generate_top15(args.project, bug_id, rankings, dataset="defects4j")
                # Generate FlexFL top20 (SBIR[:5]+Ochiai[:5]+BoostN[:5]+Agent4SR[:5], no dedup)
                generate_top20(
                    args.project,
                    bug_id,
                    args.model,
                    dataset="defects4j",
                    model_id=model_id,
                )
                # Legacy candidates.txt for backward compatibility
                write_candidates(
                    project=args.project,
                    bug_id=bug_id,
                    model=args.model,
                    dataset="defects4j",
                    model_id=model_id,
                )
            except Exception as exc:
                ts.record_error(repr(exc))
                raise
    return 0


def _cmd_combine_bugsinpy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Generate standardised rankings + top-20 for valid BugsInPy bugs.

    Like ``_cmd_run_bugsinpy``: no ``update_single_bug`` (skip-existing
    inferred from ``rankings/top15.txt``), valid-bug gated. The ranking writers already
    accept ``dataset`` and degrade gracefully when the SR ``top5`` is absent. The tracker
    is dataset-aware so ``tracker.json`` lands under ``data/BIP/``.
    """
    bugs = _resolve_versions(args)
    skip_existing = not getattr(args, "force", False)
    n_done = 0
    n_skipped = 0

    for bug_id in bugs:
        # --- Valid-bug gate (audit status + SR corpus present) ---
        skip_reason = _bip_run_skip_reason(args.project, bug_id)
        if skip_reason is not None:
            logger.warning("Skipping %s-%s: %s", args.project, bug_id, skip_reason)
            n_skipped += 1
            continue

        # Skip-existing from disk (no update_single_bug for BIP).
        if (
            skip_existing
            and (get_rankings_dir(args.project, bug_id, dataset="bugsinpy") / "top15.txt").exists()
        ):
            logger.info("Combine outputs already exist for %s-%s, skipping.", args.project, bug_id)
            n_skipped += 1
            continue

        # Resolve model_id via the (dataset-aware) tracker so tracker.json lands under BIP.
        tracker = load_tracker(args.project, bug_id, dataset="bugsinpy")
        model_id = get_or_assign_sr_model_id(
            tracker,
            model=args.model,
            temperature=0.0,
            iterations=10,
            base_url="",
            input_keys=["bug_report", "ochiai", "boostn", "sbir"],
        )
        save_tracker(tracker, args.project, bug_id, dataset="bugsinpy")

        with TrackerStep(
            args.project, bug_id, section="fl", step="top15", dataset="bugsinpy"
        ) as ts:
            try:
                # Method-level ranking CSVs (ochiai, sbir, boostn)
                rankings = generate_all_rankings(args.project, bug_id, dataset="bugsinpy")
                # Combined top15 (SBIR + Ochiai + BoostN top 5 each)
                generate_top15(args.project, bug_id, rankings, dataset="bugsinpy")
                # FlexFL top20 (SBIR[:5]+Ochiai[:5]+BoostN[:5]+Agent4SR[:5], no
                # dedup); shorter than 20 when any source is short or missing
                generate_top20(
                    args.project,
                    bug_id,
                    args.model,
                    dataset="bugsinpy",
                    model_id=model_id,
                )
                # Legacy candidates.txt for backward compatibility
                write_candidates(
                    project=args.project,
                    bug_id=bug_id,
                    model=args.model,
                    dataset="bugsinpy",
                    model_id=model_id,
                )
            except Exception as exc:
                ts.record_error(repr(exc))
                raise
        n_done += 1

    logger.info(
        "BugsInPy Agent4SR combine: %d done, %d skipped (of %d requested)",
        n_done,
        n_skipped,
        len(bugs),
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="run_agent4sr",
        description="Agent4SR: LLM-based fault localization pipeline",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # Shared flag added to subcommands that need it (not parent, to avoid
    # nargs="*" greedily consuming the subcommand name).
    _versions_kwargs: dict[str, object] = {
        "type": int,
        "nargs": "*",
        "default": None,
        "help": "Version IDs to process (space-separated). Omit to run all versions.",
    }

    def _add_benchmark_flags(sub: argparse.ArgumentParser) -> None:
        """Add --benchmark (authoritative) + deprecated -d/--dataset alias, matching run_sbir.py."""
        sub.add_argument(
            "--benchmark",
            default="defects4j",
            choices=supported_benchmarks(),
            help="Benchmark adapter key (e.g. defects4j, bugsinpy). Authoritative.",
        )
        sub.add_argument(
            "-d",
            "--dataset",
            default="defects4j",
            help="Deprecated legacy alias for --benchmark; honored only when --benchmark is unset.",
        )

    # --- corpus ---
    corpus = sp.add_parser("corpus", help="Generate method-level corpus files")
    corpus.add_argument("--verbose", action="store_true", help="Enable debug logging")
    _add_benchmark_flags(corpus)
    corpus.add_argument("-p", "--project", required=True, help="D4J project (e.g. Chart)")
    corpus.add_argument("-v", "--versions", **_versions_kwargs)  # type: ignore[arg-type]
    corpus.add_argument("--force", action="store_true", help="Overwrite existing corpus")
    corpus.add_argument(
        "--keep-checkouts",
        action="store_true",
        help="Do not remove checked-out repositories after corpus generation",
    )
    corpus.set_defaults(func=cmd_corpus)

    # --- smoke ---
    smoke = sp.add_parser("smoke", help="Smoke-test Ollama tool-calling")
    smoke.add_argument("--verbose", action="store_true", help="Enable debug logging")
    _add_benchmark_flags(smoke)
    smoke.add_argument("--model", default="llama3.1:8b")
    smoke.add_argument("--base-url", default="http://localhost:11434")
    smoke.add_argument("--no-verify", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    # --- run ---
    run = sp.add_parser("run", help="Run Agent4SR for specified bugs")
    run.add_argument("--verbose", action="store_true", help="Enable debug logging")
    _add_benchmark_flags(run)
    run.add_argument("-p", "--project", required=True, help="D4J project (e.g. Chart)")
    run.add_argument("-v", "--versions", **_versions_kwargs)  # type: ignore[arg-type]
    run.add_argument("--iterations", type=int, default=10, help="Max tool-call iterations")
    run.add_argument("--model", default="llama3.1:8b")
    run.add_argument("--base-url", default="http://localhost:11434")
    run.add_argument("--no-verify", action="store_true")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--force", action="store_true", help="Overwrite existing runs")
    run.set_defaults(func=cmd_run)

    # --- combine ---
    combine = sp.add_parser("combine", help="Combine FL + SR results")
    combine.add_argument("--verbose", action="store_true", help="Enable debug logging")
    _add_benchmark_flags(combine)
    combine.add_argument("-p", "--project", required=True, help="D4J project (e.g. Chart)")
    combine.add_argument("-v", "--versions", **_versions_kwargs)  # type: ignore[arg-type]
    combine.add_argument("--model", default="llama3.1:8b")
    combine.add_argument("--force", action="store_true", help="Overwrite existing combine outputs")
    combine.set_defaults(func=cmd_combine)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    subcommands = {"corpus", "run", "combine", "smoke"}
    if raw_argv and raw_argv[0] not in subcommands and raw_argv[0] not in {"-h", "--help"}:
        raise SystemExit("First argument must be a subcommand: corpus, run, combine, or smoke")

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    # --benchmark is authoritative; -d/--dataset is a legacy alias honored only when
    # --benchmark was left at its default (so old `-d <bench>` invocations keep working).
    # Normalize to the adapter's canonical benchmark_key and store it back on args.dataset,
    # which the subcommand handlers read via getattr(args, "dataset", "defects4j").
    benchmark_key = args.benchmark
    if args.benchmark == "defects4j" and getattr(args, "dataset", "defects4j") != "defects4j":
        benchmark_key = args.dataset
    if benchmark_key not in supported_benchmarks():
        raise NotImplementedError(
            f"Benchmark {benchmark_key!r} not supported. Supported: {supported_benchmarks()}"
        )
    args.dataset = get_benchmark_adapter(benchmark_key).benchmark_key

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    return int(args.func(args, parser))


if __name__ == "__main__":
    raise SystemExit(main())
