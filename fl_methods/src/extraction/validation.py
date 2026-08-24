"""Validation helpers for extraction pipeline outputs.

These checks are intentionally lightweight: they verify existence and basic
format sanity so batch runs can flag missing or corrupted artifacts early.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.common.config import (
    FAUXPY_COVERAGE_SUBDIR,
    FAUXPY_REPORTS_SUBDIR,
    OCHIAI_RANKING_FILE,
    SFL_SUBDIR,
    SPECTRA_FILE,
    TESTS_FILE,
)
from src.core.layout import normalize_benchmark_name
from src.extraction.d4j import EXPORT_PROPERTIES

logger = logging.getLogger(__name__)

MATRIX_FILE = "matrix.txt"

# D4J-convention artifacts that must NOT be written for the Python toolchain:
# the GZoltar/JVM property+test files (the archived attempt's anti-pattern) plus the
# raw `trigger_tests` dump, deprecated for BugsInPy in favour of failing_tests.txt
# (trigger names) + trigger_test_clean.txt (cleaned failing-test block).
_BIP_FORBIDDEN_ARTIFACTS = (
    "classes.modified",
    "dir.src.classes",
    "junit_tests.txt",
    "trigger_tests",
)


def validate_extraction_outputs(
    output_dir: Path,
    *,
    expect_gzoltar: bool,
    expect_faults: bool,
    expect_bug_report: bool,
    dataset: str = "defects4j",
) -> list[dict[str, str]]:
    """Return validation issues for one processed bug directory.

    Each issue is a dict with keys: ``severity`` (``error`` or ``warning``),
    ``file`` (relative path), and ``message``. Dispatches on ``dataset`` to the
    Defects4J or BugsInPy profile.
    """
    canonical = normalize_benchmark_name(dataset)
    issues: list[dict[str, str]] = []

    if canonical == "D4J":
        _validate_exported_properties(output_dir, issues)
        _validate_test_enumeration(output_dir, issues)
        _validate_failing_tests(output_dir, issues)
        if expect_gzoltar:
            _validate_gzoltar_outputs(output_dir, issues)
        if expect_faults:
            _validate_faults(output_dir, issues)
        if expect_bug_report:
            _validate_bug_report(output_dir, issues)
        return issues

    if canonical == "BIP":
        _validate_bugsinpy_outputs(
            output_dir,
            issues,
            expect_gzoltar=expect_gzoltar,
            expect_faults=expect_faults,
            expect_bug_report=expect_bug_report,
        )
        return issues

    raise NotImplementedError(f"Dataset {dataset!r} is not supported yet.")


# ---------------------------------------------------------------------------
# BugsInPy validation profile
# ---------------------------------------------------------------------------


def _validate_bugsinpy_outputs(
    output_dir: Path,
    issues: list[dict[str, str]],
    *,
    expect_gzoltar: bool,
    expect_faults: bool,
    expect_bug_report: bool,
) -> None:
    """Validate the Python-shaped outputs for one BugsInPy bug."""
    _check_non_empty_file(output_dir / "method_signatures.csv", "method_signatures.csv", issues)
    _check_non_empty_file(output_dir / "failing_tests.txt", "failing_tests.txt", issues)
    # all_tests.txt / relevant_tests.txt are produced by the gzoltar/FauxPy step (they reflect the
    # suite FauxPy ran), so only require them when that step ran.
    if expect_gzoltar:
        for file_name in ("all_tests.txt", "relevant_tests.txt"):
            _check_non_empty_file(output_dir / file_name, file_name, issues)
    # trigger_test_clean.txt is built in the gzoltar/FauxPy step from the live-captured trace, and
    # may legitimately be BLANK for BIP (non-mirrorable failures: collection/import errors,
    # captured-passes, not-collected). Only require its presence when that step ran; structurally
    # check it only when non-blank.
    _validate_bip_trigger_clean(output_dir, issues, expect_gzoltar=expect_gzoltar)

    # Negative checks: no D4J-convention artifacts written for the Python toolchain.
    for forbidden in _BIP_FORBIDDEN_ARTIFACTS:
        if (output_dir / forbidden).exists():
            _issue(
                issues,
                severity="error",
                file=forbidden,
                message="D4J-convention artifact must not be written for BugsInPy.",
            )

    if expect_gzoltar:
        _validate_bugsinpy_coverage(output_dir, issues)
    if expect_faults:
        # faults.txt comes from the patch (always present); faults_first is coverage-derived, so
        # only require it when the gzoltar/FauxPy step ran.
        _validate_faults(output_dir, issues, expect_first=expect_gzoltar)
    if expect_bug_report:
        _validate_bug_report(output_dir, issues)


def _validate_bugsinpy_coverage(output_dir: Path, issues: list[dict[str, str]]) -> None:
    """Validate FauxPy coverage outputs (FauxPy/coverage + FauxPy/reports)."""
    coverage_dir = output_dir / FAUXPY_COVERAGE_SUBDIR
    _check_non_empty_file(coverage_dir / SPECTRA_FILE, SPECTRA_FILE, issues)
    _check_non_empty_file(coverage_dir / MATRIX_FILE, MATRIX_FILE, issues)

    tests_path = coverage_dir / TESTS_FILE
    if not tests_path.exists():
        _issue(issues, severity="error", file=TESTS_FILE, message="Missing FauxPy tests output.")
    else:
        lines = tests_path.read_text().splitlines()
        if not lines or lines[0].strip() != "name,outcome,runtime,stacktrace":
            _issue(
                issues, severity="error", file=TESTS_FILE, message="tests.csv header is malformed."
            )

    ranking_path = output_dir / FAUXPY_REPORTS_SUBDIR / OCHIAI_RANKING_FILE
    if not ranking_path.exists():
        _issue(
            issues, severity="error", file=OCHIAI_RANKING_FILE, message="Missing Ochiai ranking."
        )
    else:
        lines = [ln.strip() for ln in ranking_path.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            _issue(
                issues,
                severity="error",
                file=OCHIAI_RANKING_FILE,
                message="Ochiai ranking has too few rows.",
            )
        elif ";" not in lines[1]:
            _issue(
                issues,
                severity="error",
                file=OCHIAI_RANKING_FILE,
                message="Expected semicolon-delimited Ochiai rows.",
            )


def _issue(
    issues: list[dict[str, str]],
    *,
    severity: str,
    file: str,
    message: str,
) -> None:
    issues.append({"severity": severity, "file": file, "message": message})


def _validate_exported_properties(output_dir: Path, issues: list[dict[str, str]]) -> None:
    for prop in EXPORT_PROPERTIES:
        path = output_dir / prop
        if not path.exists():
            _issue(
                issues,
                severity="error",
                file=prop,
                message="Missing Defects4J exported property file.",
            )
            continue
        content = path.read_text().strip()
        if not content:
            _issue(
                issues,
                severity="error",
                file=prop,
                message="Exported property file is empty.",
            )


def _validate_test_enumeration(output_dir: Path, issues: list[dict[str, str]]) -> None:
    required = {
        "all_tests.txt": "#",
        "relevant_tests.txt": "#",
        "junit_tests.txt": "JUNIT,",
    }
    for file_name, marker in required.items():
        path = output_dir / file_name
        if not path.exists():
            _issue(
                issues,
                severity="error",
                file=file_name,
                message="Missing test enumeration artifact.",
            )
            continue

        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            _issue(
                issues,
                severity="error",
                file=file_name,
                message="Test enumeration file is empty.",
            )
            continue
        if file_name == "junit_tests.txt":
            valid = all(line.startswith(marker) for line in lines)
        else:
            valid = all(marker in line for line in lines)
        if not valid:
            _issue(
                issues,
                severity="error",
                file=file_name,
                message="Test enumeration format looks invalid.",
            )


def _validate_failing_tests(output_dir: Path, issues: list[dict[str, str]]) -> None:
    raw = output_dir / "failing_tests"
    parsed = output_dir / "failing_tests.txt"
    if not raw.exists():
        _issue(
            issues,
            severity="error",
            file="failing_tests",
            message="Missing raw Defects4J failing tests artifact.",
        )
        return
    if not parsed.exists():
        _issue(
            issues,
            severity="error",
            file="failing_tests.txt",
            message="Missing parsed failing tests artifact.",
        )
        return
    lines = [line.strip() for line in parsed.read_text().splitlines() if line.strip()]
    if not lines:
        _issue(
            issues,
            severity="error",
            file="failing_tests.txt",
            message="Parsed failing tests artifact is empty.",
        )


def _validate_gzoltar_outputs(output_dir: Path, issues: list[dict[str, str]]) -> None:
    report_dir = output_dir / SFL_SUBDIR

    spectra_path = report_dir / SPECTRA_FILE
    _check_non_empty_file(spectra_path, SPECTRA_FILE, issues)

    matrix_path = report_dir / MATRIX_FILE
    _check_non_empty_file(matrix_path, MATRIX_FILE, issues)

    tests_path = report_dir / TESTS_FILE
    if tests_path.exists():
        lines = tests_path.read_text().splitlines()
        if not lines:
            _issue(
                issues,
                severity="error",
                file=TESTS_FILE,
                message="tests.csv is empty.",
            )
        else:
            expected_header = "name,outcome,runtime,stacktrace"
            if lines[0].strip() != expected_header:
                _issue(
                    issues,
                    severity="error",
                    file=TESTS_FILE,
                    message="tests.csv header is malformed.",
                )
    else:
        _issue(
            issues,
            severity="error",
            file=TESTS_FILE,
            message="Missing GZoltar tests output.",
        )

    ranking_path = report_dir / OCHIAI_RANKING_FILE
    if ranking_path.exists():
        lines = [line.strip() for line in ranking_path.read_text().splitlines() if line.strip()]
        if len(lines) < 2:
            _issue(
                issues,
                severity="error",
                file=OCHIAI_RANKING_FILE,
                message="Ochiai ranking has too few rows.",
            )
        elif ";" not in lines[1]:
            _issue(
                issues,
                severity="error",
                file=OCHIAI_RANKING_FILE,
                message="Expected semicolon-delimited Ochiai rows.",
            )
    else:
        _issue(
            issues,
            severity="error",
            file=OCHIAI_RANKING_FILE,
            message="Missing Ochiai ranking output.",
        )


def _validate_faults(
    output_dir: Path, issues: list[dict[str, str]], *, expect_first: bool = True
) -> None:
    file_name = "faults.txt"
    path = output_dir / file_name
    if not path.exists():
        _issue(
            issues,
            severity="error",
            file=file_name,
            message="Missing fault ground truth file.",
        )
        return

    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2 or not parts[1].isdigit():
            _issue(
                issues,
                severity="error",
                file=file_name,
                message=f"Malformed fault entry at line {idx}.",
            )
            return

    # faults_first is coverage-derived; skip it when no coverage was produced (e.g. a FauxPy-
    # unsupported BugsInPy env). D4J and supported BugsInPy bugs keep ``expect_first=True``.
    if expect_first:
        _validate_faults_first(output_dir, issues)


def _validate_faults_first(output_dir: Path, issues: list[dict[str, str]]) -> None:
    """Check faults_first.{csv,txt} exist; both may be header-only or empty."""
    csv_name = "faults_first.csv"
    txt_name = "faults_first.txt"
    csv_path = output_dir / csv_name
    txt_path = output_dir / txt_name
    if not csv_path.exists():
        _issue(
            issues,
            severity="warning",
            file=csv_name,
            message="Missing first-test-filtered faults CSV.",
        )
    if not txt_path.exists():
        _issue(
            issues,
            severity="warning",
            file=txt_name,
            message="Missing first-test-filtered faults TXT.",
        )


def _validate_bug_report(output_dir: Path, issues: list[dict[str, str]]) -> None:
    file_name = "bug_report.json"
    path = output_dir / file_name
    if not path.exists():
        _issue(
            issues,
            severity="warning",
            file=file_name,
            message="Bug report is missing (valid for bugs without report URLs).",
        )
        return

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        _issue(
            issues,
            severity="error",
            file=file_name,
            message="bug_report.json is not valid JSON.",
        )
        return

    if not isinstance(data, dict):
        _issue(
            issues,
            severity="error",
            file=file_name,
            message="bug_report.json must contain a JSON object.",
        )
        return

    if "error" in data:
        _issue(
            issues,
            severity="warning",
            file=file_name,
            message=f"Bug report parser returned error: {data['error']}",
        )
        return

    for key in ("title", "description"):
        value = data.get(key, "")
        if not isinstance(value, str) or not value.strip():
            _issue(
                issues,
                severity="error",
                file=file_name,
                message=f"Missing or empty '{key}' in bug report.",
            )


def _validate_bip_trigger_clean(
    output_dir: Path, issues: list[dict[str, str]], *, expect_gzoltar: bool
) -> None:
    """Validate the BugsInPy ``trigger_test_clean.txt`` (blank is valid; check shape if not).

    The file is produced by the gzoltar/FauxPy step; when that step did not run
    (``expect_gzoltar`` False) its absence is not an error.
    """
    from src.extraction.trigger_test import TRANSITION_LINE

    path = output_dir / "trigger_test_clean.txt"
    if not path.exists():
        if expect_gzoltar:
            _issue(
                issues,
                severity="error",
                file="trigger_test_clean.txt",
                message="Missing required file (gzoltar/FauxPy step ran but produced no trace).",
            )
        return
    body = path.read_text(encoding="utf-8")
    if not body.strip():
        return  # blank = a non-mirrorable failure; expected for ~17 of the 501 bugs
    if TRANSITION_LINE not in body or "Traceback (most recent call last):" not in body:
        _issue(
            issues,
            severity="error",
            file="trigger_test_clean.txt",
            message="Non-blank trigger_test_clean.txt missing the transition line or traceback block.",
        )


def _check_non_empty_file(
    path: Path,
    file_name: str,
    issues: list[dict[str, str]],
) -> None:
    if not path.exists():
        _issue(
            issues,
            severity="error",
            file=file_name,
            message="Missing required file.",
        )
        return
    if not path.read_text().strip():
        _issue(
            issues,
            severity="error",
            file=file_name,
            message="Required file is empty.",
        )
        return
    logger.debug("Validated non-empty file: %s", path)
