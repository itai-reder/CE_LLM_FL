"""Skip-and-warn for FauxPy-incompatible BugsInPy envs (Python < 3.6).

pytest-FauxPy requires ``coverage>=6.2`` (Python 3.6+); the lone 3.5 bug in the corpus
(cookiecutter/4) cannot run FauxPy. Rather than hard-fail, the pipeline skips the coverage step,
the coverage-dependent ``faults_first``, and relaxes validation (coverage / ``all_tests`` /
``relevant_tests`` / ``faults_first`` / non-blank ``trigger_test_clean`` no longer required).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extraction.fauxpy import fauxpy_supported
from src.extraction.validation import validate_extraction_outputs


class _StubRepo:
    """Minimal surface for fauxpy_supported (only needs ``_bug_info_value``)."""

    def __init__(self, python_version: str | None) -> None:
        self._python_version = python_version

    def _bug_info_value(self, key: str) -> str | None:
        return self._python_version if key == "python_version" else None


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("3.5.6", False),  # the only sub-3.6 bug (cookiecutter/4)
        ("3.6.9", True),
        ("3.7.0", True),
        ("3.8.3", True),
        ("", True),  # unparseable -> attempt rather than silently skip
        ("garbage", True),
        (None, True),
    ],
)
def test_fauxpy_supported_floor(version: str | None, supported: bool) -> None:
    assert fauxpy_supported(_StubRepo(version)) is supported  # type: ignore[arg-type]


def _make_bip_outputs_without_coverage(out: Path) -> None:
    """A BIP processed dir with only the non-coverage artifacts (the FauxPy-skipped shape)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "method_signatures.csv").write_text(
        "corpus_id;path;startLine;endLine\ncookiecutter.generate$f(a);x;1;2\n", encoding="utf-8"
    )
    (out / "failing_tests.txt").write_text(
        "tests.test_hooks$TestExternalHooks::test_run_failing_hook\n", encoding="utf-8"
    )
    (out / "faults.txt").write_text("cookiecutter.generate 85\n", encoding="utf-8")
    (out / "faults.csv").write_text(
        "path,line,signature\ncookiecutter/generate.py,85,cookiecutter.generate$f(a)\n",
        encoding="utf-8",
    )
    (out / "trigger_test_clean.txt").write_text("", encoding="utf-8")  # blank is valid


def test_validation_relaxed_when_fauxpy_skipped(tmp_path: Path) -> None:
    """With ``expect_gzoltar=False`` the missing coverage/all_tests/faults_first are not errors."""
    out = tmp_path / "processed"
    _make_bip_outputs_without_coverage(out)
    issues = validate_extraction_outputs(
        out,
        expect_gzoltar=False,
        expect_faults=True,
        expect_bug_report=False,
        dataset="bugsinpy",
    )
    assert [i for i in issues if i["severity"] == "error"] == []


def test_validation_still_requires_coverage_when_expected(tmp_path: Path) -> None:
    """The contrast: ``expect_gzoltar=True`` on the same dir flags the missing coverage outputs."""
    out = tmp_path / "processed"
    _make_bip_outputs_without_coverage(out)
    issues = validate_extraction_outputs(
        out,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=False,
        dataset="bugsinpy",
    )
    errors = {i["file"] for i in issues if i["severity"] == "error"}
    assert "all_tests.txt" in errors  # FauxPy-produced, now required
    assert any("spectra" in f or "faults_first" in f or "matrix" in f for f in errors)
