"""Shared D4J extraction pipeline used by ``run_extraction`` and ``run_sr``.

Both entry points used to inline the same set of D4J extraction calls in
slightly different orders, leading to artifact-set drift (e.g. one wrote
``trigger_test_clean.txt`` and ``method_signatures.csv`` while the
other wrote ``faults_first.csv`` + ran validation, but neither produced
the union).  :func:`ensure_d4j_outputs` runs the canonical union of steps
in the right order, wraps each tracked sub-step in ``TrackerStep``, and
optionally returns the validation issue list.

Step groups (in canonical order):

1. ``repo_setup`` — checkout (if not already done) + compile + properties
2. ``signatures`` — corpus_method_extraction
3. ``tests`` — relevant_tests + failing_tests + trigger_test_clean
4. ``gzoltar`` — full GZoltar coverage pipeline
5. ``faults`` — save_fault_lines + save_first_fault_lines
6. ``bug_report`` — save_bug_report

Callers pass ``steps=None`` for "run everything", or a subset like
``steps=("repo_setup", "signatures", "tests", "gzoltar")`` for the
``--gzoltar-only`` partial run.  Validation runs at the end and is
parameterised by which step groups ran.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from src.common.tracker import TrackerStep
from src.extraction.bug_report import save_bug_report
from src.extraction.d4j import write_parsed_failing_tests
from src.extraction.faults import save_fault_lines, save_first_fault_lines
from src.extraction.fauxpy import fauxpy_supported, run_fauxpy_pipeline
from src.extraction.gzoltar import (
    run_corpus_method_extraction,
    run_gzoltar_pipeline,
)
from src.extraction.trigger_test import save_trigger_test_clean
from src.extraction.validation import validate_extraction_outputs

if TYPE_CHECKING:
    from src.extraction.bugsinpy import BugsInPyRepo
    from src.extraction.d4j import D4JRepo

logger = logging.getLogger(__name__)

ALL_STEPS: tuple[str, ...] = (
    "repo_setup",
    "signatures",
    "tests",
    "gzoltar",
    "faults",
    "bug_report",
)


def ensure_d4j_outputs(
    repo: D4JRepo,
    *,
    project: str,
    bug_id: int | str,
    skip_existing: bool = True,
    steps: Iterable[str] | None = None,
    checked_out: bool = False,
    validate: bool = True,
) -> list[dict[str, str]]:
    """Run the canonical union of D4J extraction steps for one bug.

    ``steps`` filters which step groups to run; ``None`` means all.
    ``checked_out=True`` tells the helper the caller has already invoked
    ``repo.checkout(...)``, so ``repo_setup`` only runs compile +
    properties.  Returns the list of validation issues (empty when
    ``validate=False`` or no issues found).
    """
    selected = set(steps) if steps is not None else set(ALL_STEPS)
    unknown = selected - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"Unknown step group(s): {sorted(unknown)}")

    if "repo_setup" in selected:
        if not checked_out:
            logger.info("Checking out %s-%s", project, bug_id)
            repo.checkout(skip_existing=skip_existing)
        logger.info("Compiling %s-%s", project, bug_id)
        repo.compile(skip_existing=skip_existing)
        with TrackerStep(project, bug_id, section="extraction", step="properties"):
            logger.info("Exporting properties for %s-%s", project, bug_id)
            repo.export_all_properties()

    if "signatures" in selected:
        with TrackerStep(project, bug_id, section="extraction", step="signatures"):
            logger.info("Extracting method signatures for %s-%s", project, bug_id)
            run_corpus_method_extraction(repo, skip_existing=skip_existing, dataset="defects4j")

    if "tests" in selected:
        with TrackerStep(project, bug_id, section="extraction", step="relevant_tests"):
            repo.get_relevant_test_methods(skip_existing=skip_existing)
        write_parsed_failing_tests(repo.output_dir)
        with TrackerStep(project, bug_id, section="extraction", step="trigger_test_processed"):
            logger.info("Building cleaned trigger_test_clean.txt for %s-%s", project, bug_id)
            save_trigger_test_clean(repo, skip_existing=skip_existing)

    if "gzoltar" in selected:
        with TrackerStep(project, bug_id, section="extraction", step="gzoltar"):
            logger.info("Running GZoltar pipeline for %s-%s", project, bug_id)
            run_gzoltar_pipeline(repo, skip_existing=skip_existing)

    if "faults" in selected:
        with TrackerStep(project, bug_id, section="extraction", step="faults"):
            logger.info("Extracting fault ground truth for %s-%s", project, bug_id)
            save_fault_lines(repo, skip_existing=skip_existing)
        with TrackerStep(project, bug_id, section="extraction", step="faults_first"):
            logger.info("Filtering faults to first triggering test for %s-%s", project, bug_id)
            save_first_fault_lines(repo, skip_existing=skip_existing)

    if "bug_report" in selected:
        with TrackerStep(project, bug_id, section="extraction", step="bug_report"):
            logger.info("Fetching bug report for %s-%s", project, bug_id)
            save_bug_report(repo, skip_existing=skip_existing)

    if not validate:
        return []
    return validate_extraction_outputs(
        repo.output_dir,
        expect_gzoltar="gzoltar" in selected,
        expect_faults="faults" in selected,
        expect_bug_report="bug_report" in selected,
        dataset="defects4j",
    )


# ---------------------------------------------------------------------------
# BugsInPy pipeline
# ---------------------------------------------------------------------------


def _write_bugsinpy_tests(repo: BugsInPyRepo, *, skip_existing: bool) -> None:
    """Write the BugsInPy ``failing_tests.txt`` trigger-name list (read by ``compute_universe``).

    The extensionless ``trigger_tests`` is **deprecated for BugsInPy** (it was a degenerate clone
    of D4J's raw ``defects4j`` dump): trigger names come from ``failing_tests.txt``.

    ``trigger_test_clean.txt`` (the Agent4LR/Agent4SR consumer) is **not** written here — it is
    built from the live-captured ``trigger_trace.txt`` in the ``gzoltar``/FauxPy step
    (:func:`src.extraction.fauxpy.run_fauxpy_pipeline`), since the trace is produced by the test
    run, not derivable container-free.
    """
    out = repo.output_dir
    out.mkdir(parents=True, exist_ok=True)
    triggers = repo.get_trigger_tests()

    failing = out / "failing_tests.txt"
    if not (skip_existing and failing.exists()):
        failing.write_text("\n".join(triggers) + ("\n" if triggers else ""), encoding="utf-8")


def ensure_bugsinpy_outputs(
    repo: BugsInPyRepo,
    *,
    project: str,
    bug_id: int | str,
    skip_existing: bool = True,
    steps: Iterable[str] | None = None,
    checked_out: bool = False,
    validate: bool = True,
) -> list[dict[str, str]]:
    """Run the canonical union of BugsInPy extraction steps for one bug.

    Mirrors :func:`ensure_d4j_outputs` (same ``ALL_STEPS`` order) but dispatches to
    the Python toolchain: the ``gzoltar`` step is the FauxPy coverage pipeline, and
    no GZoltar/JVM property exports or tracker steps run.
    """
    selected = set(steps) if steps is not None else set(ALL_STEPS)
    unknown = selected - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"Unknown step group(s): {sorted(unknown)}")

    # FauxPy cannot run on a too-old Python env (only cookiecutter/4 @ 3.5.6 in the corpus); when
    # unsupported, the coverage step and the coverage-dependent faults_first are skipped-and-warned
    # rather than hard-failed, and validation is relaxed accordingly.
    fauxpy_ok = fauxpy_supported(repo)

    if "repo_setup" in selected:
        if not checked_out:
            logger.info("Checking out %s-%s", project, bug_id)
            repo.checkout(skip_existing=skip_existing)
        logger.info("Compiling %s-%s", project, bug_id)
        repo.compile(skip_existing=skip_existing)

    if "signatures" in selected:
        logger.info("Extracting method signatures for %s-%s", project, bug_id)
        run_corpus_method_extraction(repo, skip_existing=skip_existing, dataset="bugsinpy")

    if "tests" in selected:
        logger.info("Writing trigger/failing test artifacts for %s-%s", project, bug_id)
        _write_bugsinpy_tests(repo, skip_existing=skip_existing)

    if "gzoltar" in selected:
        logger.info("Running FauxPy coverage pipeline for %s-%s", project, bug_id)
        run_fauxpy_pipeline(repo, skip_existing=skip_existing)

    if "faults" in selected:
        logger.info("Extracting fault ground truth for %s-%s", project, bug_id)
        save_fault_lines(repo, skip_existing=skip_existing, dataset="bugsinpy")
        if fauxpy_ok:
            save_first_fault_lines(repo, skip_existing=skip_existing, dataset="bugsinpy")
        else:
            logger.warning(
                "%s-%s: skipping faults_first (no coverage — FauxPy unsupported on this env).",
                project,
                bug_id,
            )

    if "bug_report" in selected:
        logger.info("Fetching bug report for %s-%s", project, bug_id)
        save_bug_report(repo, skip_existing=skip_existing, dataset="bugsinpy")

    if not validate:
        return []
    issues = validate_extraction_outputs(
        repo.output_dir,
        expect_gzoltar=("gzoltar" in selected) and fauxpy_ok,
        expect_faults="faults" in selected,
        expect_bug_report="bug_report" in selected,
        dataset="bugsinpy",
    )
    if not fauxpy_ok:
        issues.append(
            {
                "severity": "warning",
                "file": "FauxPy/coverage",
                "message": (
                    f"FauxPy skipped: Python {repo._bug_info_value('python_version')!r} is below "
                    "the FauxPy floor (pytest-FauxPy needs coverage>=6.2, unavailable for <3.6). "
                    "Coverage, all_tests/relevant_tests, faults_first, and a non-blank "
                    "trigger_test_clean.txt are absent for this bug."
                ),
            }
        )
    return issues
