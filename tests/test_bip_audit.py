"""Tests for the BugsInPy extraction completeness audit (src.extraction.bip_audit).

Covers the three pieces the driver and the audit CLI both rely on:
- the fault.csv <-> spectra parsing helpers,
- the pure status decision table, and
- ``classify_bug`` end-to-end over crafted output directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extraction import bip_audit
from src.extraction.bip_audit import (
    BugAuditResult,
    classify_bug,
    decide_status,
    read_fault_signatures,
    read_spectra_methods,
    summarize,
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def test_read_fault_signatures_skips_empty(tmp_path: Path) -> None:
    # The signature contains a comma, so it is CSV-quoted (as csv.writer emits it).
    (tmp_path / "faults.csv").write_text(
        "path,line,signature\n"
        'black.py,621,"black$reformat_many(sources,fast)"\n'
        'black.py,640,"black$reformat_many(sources,fast)"\n'  # duplicate collapses
        "util.py,10,\n",  # empty signature -> skipped
        encoding="utf-8",
    )
    assert read_fault_signatures(tmp_path) == {"black$reformat_many(sources,fast)"}


def test_read_fault_signatures_missing_file(tmp_path: Path) -> None:
    assert read_fault_signatures(tmp_path) == set()


def test_read_spectra_methods_strips_header_and_line(tmp_path: Path) -> None:
    cov = tmp_path / "FauxPy" / "coverage"
    cov.mkdir(parents=True)
    (cov / "spectra.csv").write_text(
        "name\nblack$Ok.__init__(self,value):120\nblack$Ok.ok(self):123\n",
        encoding="utf-8",
    )
    assert read_spectra_methods(tmp_path) == {
        "black$Ok.__init__(self,value)",
        "black$Ok.ok(self)",
    }


def test_read_spectra_methods_missing_file(tmp_path: Path) -> None:
    assert read_spectra_methods(tmp_path) == set()


# ---------------------------------------------------------------------------
# Pure decision table
# ---------------------------------------------------------------------------


def test_decide_status_error_wins() -> None:
    assert (
        decide_status(
            has_errors=True,
            has_warnings=True,
            fauxpy_supported=True,
            fault_sigs={"a"},
            unreached={"a"},
        )
        == "real_failure"
    )


def test_decide_status_fl_unreachable() -> None:
    assert (
        decide_status(
            has_errors=False,
            has_warnings=False,
            fauxpy_supported=True,
            fault_sigs={"a", "b"},
            unreached={"a", "b"},
        )
        == "fl_unreachable"
    )


def test_decide_status_partial_coverage_is_not_unreachable() -> None:
    # One of the two fault methods IS covered -> not fl_unreachable.
    assert (
        decide_status(
            has_errors=False,
            has_warnings=False,
            fauxpy_supported=True,
            fault_sigs={"a", "b"},
            unreached={"a"},
        )
        == "ok"
    )


def test_decide_status_unsupported_is_reduced() -> None:
    assert (
        decide_status(
            has_errors=False,
            has_warnings=False,
            fauxpy_supported=False,
            fault_sigs=set(),
            unreached=set(),
        )
        == "ok_reduced"
    )


def test_decide_status_warnings_are_reduced() -> None:
    assert (
        decide_status(
            has_errors=False,
            has_warnings=True,
            fauxpy_supported=True,
            fault_sigs={"a"},
            unreached=set(),
        )
        == "ok_reduced"
    )


def test_decide_status_clean_is_ok() -> None:
    assert (
        decide_status(
            has_errors=False,
            has_warnings=False,
            fauxpy_supported=True,
            fault_sigs={"a"},
            unreached=set(),
        )
        == "ok"
    )


# ---------------------------------------------------------------------------
# classify_bug end-to-end over crafted output dirs
# ---------------------------------------------------------------------------

FAULT_SIG = "black$reformat_many(sources,fast)"


def _make_complete_bug(out: Path, *, covered: bool = True, with_bug_report: bool = True) -> None:
    """Write a fully valid BugsInPy output set (no validation errors)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "method_signatures.csv").write_text(
        f"corpus_id;path;startLine;endLine\n{FAULT_SIG};black.py;600;700\n", encoding="utf-8"
    )
    (out / "failing_tests.txt").write_text("tests.test_black$T::test_x\n", encoding="utf-8")
    (out / "all_tests.txt").write_text("tests.test_black$T::test_x\n", encoding="utf-8")
    (out / "relevant_tests.txt").write_text("tests.test_black$T::test_x\n", encoding="utf-8")
    (out / "trigger_test_clean.txt").write_text("", encoding="utf-8")  # blank is valid
    (out / "faults.txt").write_text("black 621\n", encoding="utf-8")
    # FAULT_SIG contains a comma -> CSV-quote it (as csv.writer does).
    (out / "faults.csv").write_text(
        f'path,line,signature\nblack.py,621,"{FAULT_SIG}"\n', encoding="utf-8"
    )
    (out / "faults_first.csv").write_text("path,line,signature\n", encoding="utf-8")
    (out / "faults_first.txt").write_text("", encoding="utf-8")

    cov = out / "FauxPy" / "coverage"
    cov.mkdir(parents=True, exist_ok=True)
    spectra_method = FAULT_SIG if covered else "black$other(x)"
    (cov / "spectra.csv").write_text(f"name\n{spectra_method}:621\n", encoding="utf-8")
    (cov / "matrix.txt").write_text("1 +\n", encoding="utf-8")
    (cov / "tests.csv").write_text(
        "name,outcome,runtime,stacktrace\ntests.test_black$T::test_x,FAIL,0,\n", encoding="utf-8"
    )
    reports = out / "FauxPy" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ochiai.ranking.csv").write_text(
        f"name;suspiciousness_value\n{spectra_method}:621;0.5\n", encoding="utf-8"
    )
    if with_bug_report:
        (out / "bug_report.json").write_text(
            json.dumps({"title": "t", "description": "d", "url": "u", "raw": "r"}),
            encoding="utf-8",
        )


class _FakeRepo:
    """Minimal stand-in for BugsInPyRepo (classify_bug only needs project/bug_id/output_dir)."""

    def __init__(self, output_dir: Path) -> None:
        self.project = "black"
        self.bug_id = 1
        self.output_dir = output_dir


@pytest.fixture
def supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bip_audit, "fauxpy_supported", lambda repo: True)


def test_classify_missing(tmp_path: Path, supported: None) -> None:
    repo = _FakeRepo(tmp_path / "nope")
    result = classify_bug(repo)  # type: ignore[arg-type]
    assert result.status == "missing"
    assert result.needs_review


def test_classify_ok(tmp_path: Path, supported: None) -> None:
    _make_complete_bug(tmp_path, covered=True, with_bug_report=True)
    result = classify_bug(_FakeRepo(tmp_path))  # type: ignore[arg-type]
    assert result.status == "ok"
    assert not result.needs_review
    assert result.fault_sigs == (FAULT_SIG,)
    assert result.unreached_sigs == ()


def test_classify_ok_reduced_on_warning(tmp_path: Path, supported: None) -> None:
    _make_complete_bug(tmp_path, covered=True, with_bug_report=False)  # missing report -> warning
    result = classify_bug(_FakeRepo(tmp_path))  # type: ignore[arg-type]
    assert result.status == "ok_reduced"
    assert not result.needs_review
    assert result.n_warnings >= 1


def test_classify_fl_unreachable(tmp_path: Path, supported: None) -> None:
    _make_complete_bug(tmp_path, covered=False, with_bug_report=True)
    result = classify_bug(_FakeRepo(tmp_path))  # type: ignore[arg-type]
    assert result.status == "fl_unreachable"
    assert result.needs_review
    assert result.unreached_sigs == (FAULT_SIG,)


def test_classify_real_failure(tmp_path: Path, supported: None) -> None:
    _make_complete_bug(tmp_path, covered=True)
    (tmp_path / "method_signatures.csv").write_text("", encoding="utf-8")  # required-but-empty
    result = classify_bug(_FakeRepo(tmp_path))  # type: ignore[arg-type]
    assert result.status == "real_failure"
    assert result.needs_review
    assert result.n_errors >= 1


def test_classify_unsupported_is_reduced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # FauxPy-unsupported (e.g. cookiecutter/4 @ Py3.5): no coverage, reduced set, no errors.
    monkeypatch.setattr(bip_audit, "fauxpy_supported", lambda repo: False)
    out = tmp_path
    out.mkdir(parents=True, exist_ok=True)
    (out / "method_signatures.csv").write_text(
        "corpus_id;path;s;e\na$b();f.py;1;2\n", encoding="utf-8"
    )
    (out / "failing_tests.txt").write_text("m$T::t\n", encoding="utf-8")
    (out / "faults.txt").write_text("mod 5\n", encoding="utf-8")
    (out / "faults.csv").write_text("path,line,signature\nf.py,5,\n", encoding="utf-8")
    (out / "bug_report.json").write_text(
        json.dumps({"title": "t", "description": "d"}), encoding="utf-8"
    )
    result = classify_bug(_FakeRepo(out))  # type: ignore[arg-type]
    assert result.status == "ok_reduced"
    assert not result.needs_review


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_counts_and_review() -> None:
    results = [
        BugAuditResult("p", 1, "ok", True),
        BugAuditResult("p", 2, "ok_reduced", True),
        BugAuditResult("p", 3, "fl_unreachable", True),
        BugAuditResult("p", 4, "real_failure", True),
        BugAuditResult("p", 5, "missing", True),
    ]
    summary = summarize(results)
    assert summary["total"] == 5
    assert summary["needs_review"] == 3  # fl_unreachable + real_failure + missing
    assert summary["ok"] == 1
    assert summary["ok_reduced"] == 1
