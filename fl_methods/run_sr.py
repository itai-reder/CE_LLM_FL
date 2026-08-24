#!/usr/bin/env python3
"""Run the CEFL Space-Reduction (SR) pipeline for one or more bugs.

Renamed from ``run_all.py``. The Localization-Refinement (LR) phase lives
in ``run_lr.py``; ``run_flexfl.py`` chains SR + LR end-to-end.

Performs a single checkout + cleanup per bug, running in order:
    1. Extraction (GZoltar + faults + bug report)
    2. BoostN
    3. SBIR (SBFL -> Blues -> RAFL)
    4. Agent4SR corpus generation
    5. Agent4SR run (LLM tool-calling)
    6. Combine (standardised rankings + top15/top20)

Usage:
    python run_sr.py -p Lang -v 4 --model llama3.1:8b \
        --no-verify --base-url https://cis-ollama.auth.ad.bgu.ac.il
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, cast

from update_tracker import compute_coverage, update_single_bug

from src.agent4sr.agent import SRConfig, run_agent4sr_for_bug
from src.agent4sr.combine import write_candidates
from src.agent4sr.corpus import save_corpus
from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.boostn.boostn import BoostN
from src.common.config import get_processed_dir
from src.common.rankings import generate_all_rankings, generate_top15, generate_top20
from src.common.tracker import (
    TrackerStep,
    get_or_assign_sr_model_id,
    load_tracker,
    save_tracker,
)
from src.extraction.d4j import D4JRepo
from src.extraction.pipeline import ensure_d4j_outputs
from src.sbir.blues import Blues
from src.sbir.rafl import RAFL
from src.sbir.sbfl import SBFL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_sr")


def _run_extraction_d4j(repo: D4JRepo, *, skip_existing: bool, project: str, bug_id: str) -> None:
    """Run the extraction steps for a Defects4J checkout (caller checks out).

    Delegates to :func:`ensure_d4j_outputs` for the canonical step set;
    ``checked_out=True`` because :func:`run_single_bug_d4j` runs the
    ``ensure_checkout()`` step before invoking this helper.  Each tracked
    sub-step is wrapped in a :class:`TrackerStep` inside the shared
    pipeline; ``checkout`` and ``compile`` are deliberately NOT tracked.
    """
    issues = ensure_d4j_outputs(
        repo,
        project=project,
        bug_id=bug_id,
        skip_existing=skip_existing,
        checked_out=True,
    )
    for issue in issues:
        level = logging.ERROR if issue["severity"] == "error" else logging.WARNING
        log.log(
            level,
            "[%s-%s] Validation %s in %s: %s",
            project,
            bug_id,
            issue["severity"],
            issue["file"],
            issue["message"],
        )


def run_single_bug(
    project: str,
    bug_id: str,
    *,
    dataset: str = "defects4j",
    sr_cfg: SRConfig,
    use_ce: bool,
    ce_max_iter: int,
    ce_pop_size: int,
    skip_existing: bool,
    keep_checkouts: bool,
) -> dict:
    """Dispatch the full SR pipeline for one bug based on ``dataset``."""
    if dataset == "defects4j":
        return run_single_bug_d4j(
            project,
            bug_id,
            sr_cfg=sr_cfg,
            use_ce=use_ce,
            ce_max_iter=ce_max_iter,
            ce_pop_size=ce_pop_size,
            skip_existing=skip_existing,
            keep_checkouts=keep_checkouts,
        )
    raise NotImplementedError(f"Dataset {dataset!r} not supported yet")


def run_single_bug_d4j(
    project: str,
    bug_id: str,
    *,
    sr_cfg: SRConfig,
    use_ce: bool,
    ce_max_iter: int,
    ce_pop_size: int,
    skip_existing: bool,
    keep_checkouts: bool,
) -> dict:
    """Run the full SR pipeline for one Defects4J bug. Returns a summary dict."""
    processed_dir = get_processed_dir(project, bug_id, dataset="defects4j")
    summary: dict = {
        "project": project,
        "bug_id": bug_id,
        "processed_dir": str(processed_dir),
        "stages": {},
    }

    log.info("=" * 70)
    log.info("Processing %s-%s", project, bug_id)
    log.info("  Output dir: %s", processed_dir)
    log.info("=" * 70)

    repo = D4JRepo(project, int(bug_id))
    _checked_out = False

    def ensure_checkout() -> None:
        nonlocal _checked_out
        if not _checked_out:
            log.info("Checking out %s-%s...", project, bug_id)
            repo.checkout(skip_existing=skip_existing)
            _checked_out = True

    try:
        # Infer current pipeline state from disk
        inferred_tracker = cast(
            dict[str, Any],
            update_single_bug(processed_dir, project, int(bug_id), dry_run=True, no_backup=True),
        )

        # -- 1. Extraction ----------------------------------------------------
        log.info(">>> Stage 1/6: Extraction <<<")
        t0 = time.time()
        ext_reqs = {
            "properties",
            "signatures",
            "relevant_tests",
            "failing_tests",
            "gzoltar",
            "faults",
            "bug_report",
            "trigger_test_processed",
        }
        if skip_existing and ext_reqs.issubset(set(inferred_tracker["extraction"]["completed"])):
            log.info("Extraction outputs already exist, skipping.")
            summary["stages"]["extraction"] = {"status": "SKIPPED"}
        else:
            ensure_checkout()
            try:
                _run_extraction_d4j(
                    repo, skip_existing=skip_existing, project=project, bug_id=bug_id
                )
                summary["stages"]["extraction"] = {
                    "status": "OK",
                    "time_s": round(time.time() - t0, 1),
                }
            except Exception as exc:
                log.exception("Extraction failed")
                summary["stages"]["extraction"] = {"status": "FAILED", "error": str(exc)}
                raise

        # -- 2. BoostN --------------------------------------------------------
        log.info(">>> Stage 2/6: BoostN <<<")
        t0 = time.time()
        if skip_existing and "boostn" in inferred_tracker["fl"]["completed"]:
            log.info("BoostN outputs already exist, skipping.")
            summary["stages"]["boostn"] = {"status": "SKIPPED"}
        else:
            ensure_checkout()
            with TrackerStep(project, bug_id, section="fl", step="boostn") as ts:
                try:
                    boostn = BoostN()
                    boostn_results = boostn.process_project(project, bug_id, dataset="defects4j")
                    summary["stages"]["boostn"] = {
                        "status": "OK",
                        "n_methods": len(boostn_results) if boostn_results else 0,
                        "time_s": round(time.time() - t0, 1),
                    }
                except Exception as exc:
                    log.exception("BoostN failed")
                    ts.record_error(repr(exc))
                    summary["stages"]["boostn"] = {"status": "FAILED", "error": str(exc)}

        # -- 3. SBIR ----------------------------------------------------------
        log.info(">>> Stage 3/6: SBIR <<<")
        t0 = time.time()

        if skip_existing and "ochiai" in inferred_tracker["fl"]["completed"]:
            log.info("Ochiai outputs already exist, skipping.")
        else:
            ensure_checkout()
            with TrackerStep(project, bug_id, section="fl", step="ochiai") as ts_ochiai:
                try:
                    SBFL().process_project(project, bug_id, dataset="defects4j")
                except Exception as exc:
                    ts_ochiai.record_error(repr(exc))
                    raise

        if skip_existing and "sbir" in inferred_tracker["fl"]["completed"]:
            log.info("SBIR outputs already exist, skipping.")
            summary["stages"]["sbir"] = {"status": "SKIPPED"}
        else:
            ensure_checkout()
            with TrackerStep(project, bug_id, section="fl", step="sbir") as ts_sbir:
                try:
                    Blues().process_project(project, bug_id, dataset="defects4j")
                    rafl_results = RAFL().process_project(
                        project,
                        bug_id,
                        use_ce=use_ce,
                        ce_max_iter=ce_max_iter,
                        ce_pop_size=ce_pop_size,
                        dataset="defects4j",
                    )
                    summary["stages"]["sbir"] = {
                        "status": "OK",
                        "n_statements": len(rafl_results) if rafl_results else 0,
                        "time_s": round(time.time() - t0, 1),
                        "method": "CE" if use_ce else "Borda",
                    }
                except Exception as exc:
                    log.exception("SBIR failed")
                    ts_sbir.record_error(repr(exc))
                    summary["stages"]["sbir"] = {"status": "FAILED", "error": str(exc)}

        # -- 4. Agent4SR corpus ----------------------------------------------
        log.info(">>> Stage 4/6: Agent4SR corpus <<<")
        t0 = time.time()
        sr_dir = processed_dir / "FlexFL" / "SR"
        if (
            skip_existing
            and (sr_dir / "corpus_methods.txt").exists()
            and (sr_dir / "corpus_codes.txt").exists()
        ):
            log.info("Corpus already exists, skipping.")
            summary["stages"]["corpus"] = {"status": "SKIPPED"}
        else:
            ensure_checkout()
            try:
                save_corpus(project, bug_id, skip_existing=skip_existing, dataset="defects4j")
                summary["stages"]["corpus"] = {
                    "status": "OK",
                    "time_s": round(time.time() - t0, 1),
                }
            except Exception as exc:
                log.exception("Corpus generation failed")
                summary["stages"]["corpus"] = {"status": "FAILED", "error": str(exc)}

        # -- 5. Agent4SR run --------------------------------------------------
        log.info(">>> Stage 5/6: Agent4SR run <<<")
        t0 = time.time()

        # Resolve model_id via tracker before the run
        tracker = load_tracker(project, bug_id)
        model_id = get_or_assign_sr_model_id(
            tracker,
            model=sr_cfg.model,
            temperature=sr_cfg.temperature,
            iterations=sr_cfg.iterations,
            base_url=sr_cfg.base_url,
            input_keys=["bug_report", "ochiai", "boostn", "sbir"],
        )
        save_tracker(tracker, project, bug_id)

        if skip_existing and "run" in inferred_tracker["sr"].get(model_id, {}).get("completed", []):
            log.info("Agent4SR run for model %s already complete, skipping.", model_id)
            summary["stages"]["agent4sr"] = {"status": "SKIPPED"}
        else:
            with TrackerStep(project, bug_id, section="sr", step="run", model_id=model_id) as ts:
                try:
                    run_agent4sr_for_bug(
                        project=project,
                        bug_id=bug_id,
                        cfg=sr_cfg,
                        dataset="defects4j",
                        model_id=model_id,
                    )
                    summary["stages"]["agent4sr"] = {
                        "status": "OK",
                        "time_s": round(time.time() - t0, 1),
                    }
                except Exception as exc:
                    log.exception("Agent4SR run failed")
                    ts.record_error(repr(exc))
                    summary["stages"]["agent4sr"] = {"status": "FAILED", "error": str(exc)}

        # -- 6. Combine -------------------------------------------------------
        log.info(">>> Stage 6/6: Combine <<<")
        t0 = time.time()
        if skip_existing and "top15" in inferred_tracker["fl"]["completed"]:
            log.info("Combine outputs already exist, skipping.")
            summary["stages"]["combine"] = {"status": "SKIPPED"}
        else:
            with TrackerStep(project, bug_id, section="fl", step="top15") as ts:
                try:
                    rankings = generate_all_rankings(project, bug_id, dataset="defects4j")
                    generate_top15(project, bug_id, rankings, dataset="defects4j")
                    generate_top20(
                        project,
                        bug_id,
                        sr_cfg.model,
                        dataset="defects4j",
                        model_id=model_id,
                    )
                    write_candidates(
                        project=project,
                        bug_id=bug_id,
                        model=sr_cfg.model,
                        dataset="defects4j",
                        model_id=model_id,
                    )
                    summary["stages"]["combine"] = {
                        "status": "OK",
                        "time_s": round(time.time() - t0, 1),
                    }
                except Exception as exc:
                    log.exception("Combine failed")
                    ts.record_error(repr(exc))
                    summary["stages"]["combine"] = {"status": "FAILED", "error": str(exc)}

        # -- Coverage update ---------------------------------------------------
        try:
            cov_tracker = load_tracker(project, bug_id)
            compute_coverage(processed_dir, cov_tracker)
            save_tracker(cov_tracker, project, bug_id)
        except Exception:
            log.exception("Coverage update failed for %s-%s", project, bug_id)

    except Exception as exc:
        log.exception("Pipeline aborted for %s-%s", project, bug_id)
        summary.setdefault("error", str(exc))
    finally:
        if not keep_checkouts:
            try:
                repo.remove_repo()
            except Exception:
                log.exception("Failed to clean up checkout for %s-%s", project, bug_id)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full CEFL SR pipeline (extraction + BoostN + SBIR + Agent4SR).",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="defects4j",
        help="Benchmark dataset (currently only 'defects4j' is supported)",
    )
    parser.add_argument("-p", "--project", required=True, help="Defects4J project name")
    parser.add_argument(
        "-v",
        "--versions",
        type=int,
        nargs="*",
        default=None,
        help="Version IDs to process (space-separated). Omit to run all versions.",
    )
    parser.add_argument("--use_ce", action="store_true", help="Use CE Monte Carlo for RAFL")
    parser.add_argument("--ce_max_iter", type=int, default=1000)
    parser.add_argument("--ce_pop_size", type=int, default=100)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if outputs already exist",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
        help="Do not remove the checkout after the pipeline finishes",
    )
    # Agent4SR run flags
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    # TODO: revise this script to recognise the --dataset flag when integrating
    # additional benchmarks. For now only Defects4J is supported.
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

    if args.versions is not None:
        bugs = [(args.project, str(v)) for v in args.versions]
    else:
        adapter = get_benchmark_adapter(args.dataset)
        bugs = [(args.project, str(v)) for v in adapter.list_cases(args.project)]

    t_global = time.time()
    all_summaries: list[dict] = []

    for project, bug_id in bugs:
        sr_cfg = SRConfig.from_cli(
            model=args.model,
            iterations=args.iterations,
            temperature=args.temperature,
            base_url=args.base_url,
            verify=not args.no_verify,
            bugs=[bug_id],
        )
        summary = run_single_bug(
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
        all_summaries.append(summary)

    # -- Final Summary --------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    for s in all_summaries:
        log.info("")
        log.info("%s-%s:", s["project"], s["bug_id"])
        for stage_name, stage in s["stages"].items():
            log.info("  %-10s %s", stage_name + ":", stage.get("status", "N/A"))

    log.info("")
    log.info("Total wall time: %.1f seconds.", time.time() - t_global)

    had_error = any(
        st.get("status") == "FAILED" for s in all_summaries for st in s["stages"].values()
    )
    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
