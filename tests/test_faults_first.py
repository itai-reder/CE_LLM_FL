"""Tests for src.extraction.faults.save_first_fault_lines."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.common.config import get_processed_dir
from src.extraction.faults import (
    FAULTS_FIRST_CSV,
    FAULTS_FIRST_TXT,
    save_first_fault_lines,
)


def _stub_repo(output_dir: Path) -> Any:
    repo = MagicMock()
    repo.output_dir = output_dir
    return repo


def _write_processed_dir(
    tmp_path: Path,
    *,
    faults_csv_body: str,
    faults_txt_body: str,
    corpus_csv_body: str,
    trigger_tests_body: str,
    tests_csv_body: str,
    matrix_body: str,
    spectra_body: str,
) -> Path:
    (tmp_path / "faults.csv").write_text(faults_csv_body)
    (tmp_path / "faults.txt").write_text(faults_txt_body)
    (tmp_path / "method_signatures.csv").write_text(corpus_csv_body)
    (tmp_path / "trigger_tests").write_text(trigger_tests_body)
    sfl = tmp_path / "sfl" / "sfl" / "txt"
    sfl.mkdir(parents=True)
    (sfl / "tests.csv").write_text(tests_csv_body)
    (sfl / "matrix.txt").write_text(matrix_body)
    (sfl / "spectra.csv").write_text(spectra_body)
    return tmp_path


# ---------------------------------------------------------------------------
# Single trigger test — faults_first == faults
# ---------------------------------------------------------------------------


class TestSingleTrigger:
    def test_single_trigger_keeps_all_covered_faults(self, tmp_path: Path) -> None:
        proc = _write_processed_dir(
            tmp_path,
            faults_csv_body=textwrap.dedent(
                """\
                path,line,signature
                a/b/Cls.java,12,a.b$Cls.foo(int)
                a/b/Cls.java,30,a.b$Cls.bar(String)
                """
            ),
            faults_txt_body=textwrap.dedent(
                """\
                a.b.Cls 12
                a.b.Cls 30
                """
            ),
            corpus_csv_body=textwrap.dedent(
                """\
                corpus_id;path;startLine;endLine
                a.b$Cls.foo(int);a/b/Cls.java;10;20
                a.b$Cls.bar(String);a/b/Cls.java;25;40
                """
            ),
            trigger_tests_body=("--- a.b.ClsTest::testIt\nsome stack trace\n"),
            tests_csv_body=("name,outcome,runtime,stacktrace\na.b.ClsTest#testIt,FAIL,1,boom\n"),
            # Matrix: 1 row, 2 statement bits + outcome marker
            matrix_body="1 1 -\n",
            spectra_body=textwrap.dedent(
                """\
                name
                a.b$Cls#foo(int):12
                a.b$Cls#bar(java.lang.String):30
                """
            ),
        )

        result = save_first_fault_lines(_stub_repo(proc))
        assert result == proc / FAULTS_FIRST_CSV

        rows = list(csv.DictReader((proc / FAULTS_FIRST_CSV).open(encoding="utf-8")))
        sigs = [r["signature"] for r in rows]
        assert sigs == ["a.b$Cls.foo(int)", "a.b$Cls.bar(String)"]

        txt_lines = (proc / FAULTS_FIRST_TXT).read_text().splitlines()
        assert txt_lines == ["a.b.Cls 12", "a.b.Cls 30"]


# ---------------------------------------------------------------------------
# Two triggers — first one covers only foo
# ---------------------------------------------------------------------------


class TestTwoTriggersFirstOnly:
    def test_filters_to_first_test_coverage(self, tmp_path: Path) -> None:
        proc = _write_processed_dir(
            tmp_path,
            faults_csv_body=textwrap.dedent(
                """\
                path,line,signature
                a/b/Cls.java,12,a.b$Cls.foo(int)
                a/b/Cls.java,30,a.b$Cls.bar(String)
                """
            ),
            faults_txt_body=textwrap.dedent(
                """\
                a.b.Cls 12
                a.b.Cls 30
                """
            ),
            corpus_csv_body=textwrap.dedent(
                """\
                corpus_id;path;startLine;endLine
                a.b$Cls.foo(int);a/b/Cls.java;10;20
                a.b$Cls.bar(String);a/b/Cls.java;25;40
                """
            ),
            trigger_tests_body=(
                "--- a.b.ClsTest::testFoo\nfoo stack trace\n"
                "--- a.b.ClsTest::testBar\nbar stack trace\n"
            ),
            tests_csv_body=textwrap.dedent(
                """\
                name,outcome,runtime,stacktrace
                a.b.ClsTest#testFoo,FAIL,1,boom1
                a.b.ClsTest#testBar,FAIL,1,boom2
                """
            ),
            # Row 0 (testFoo) covers col 0 only (foo line 12)
            # Row 1 (testBar) covers col 1 only (bar line 30)
            matrix_body="1 0 -\n0 1 -\n",
            spectra_body=textwrap.dedent(
                """\
                name
                a.b$Cls#foo(int):12
                a.b$Cls#bar(java.lang.String):30
                """
            ),
        )

        save_first_fault_lines(_stub_repo(proc))

        rows = list(csv.DictReader((proc / FAULTS_FIRST_CSV).open(encoding="utf-8")))
        sigs = [r["signature"] for r in rows]
        assert sigs == ["a.b$Cls.foo(int)"]

        txt_lines = (proc / FAULTS_FIRST_TXT).read_text().splitlines()
        assert txt_lines == ["a.b.Cls 12"]


# ---------------------------------------------------------------------------
# Empty result — no fault covered by first test
# ---------------------------------------------------------------------------


class TestEmptyResult:
    def test_header_only_csv_and_empty_txt(self, tmp_path: Path) -> None:
        proc = _write_processed_dir(
            tmp_path,
            faults_csv_body=textwrap.dedent(
                """\
                path,line,signature
                a/b/Cls.java,12,a.b$Cls.foo(int)
                """
            ),
            faults_txt_body="a.b.Cls 12\n",
            corpus_csv_body=textwrap.dedent(
                """\
                corpus_id;path;startLine;endLine
                a.b$Cls.foo(int);a/b/Cls.java;10;20
                """
            ),
            trigger_tests_body=("--- a.b.ClsTest::testNothing\nsome stack trace\n"),
            tests_csv_body=(
                "name,outcome,runtime,stacktrace\na.b.ClsTest#testNothing,FAIL,1,boom\n"
            ),
            # Test covers nothing
            matrix_body="0 -\n",
            spectra_body="name\na.b$Cls#foo(int):12\n",
        )

        save_first_fault_lines(_stub_repo(proc))

        csv_path = proc / FAULTS_FIRST_CSV
        txt_path = proc / FAULTS_FIRST_TXT
        assert csv_path.exists()
        assert txt_path.exists()
        # Header-only CSV
        text = csv_path.read_text(encoding="utf-8").strip()
        assert text == "path,line,signature"
        # Empty TXT
        assert txt_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Idempotency — skip_existing
# ---------------------------------------------------------------------------


class TestSkipExisting:
    def test_does_not_overwrite_when_skip_existing(self, tmp_path: Path) -> None:
        proc = _write_processed_dir(
            tmp_path,
            faults_csv_body="path,line,signature\n",
            faults_txt_body="",
            corpus_csv_body="corpus_id;path;startLine;endLine\n",
            trigger_tests_body="--- a.b.ClsTest::testIt\nstuff\n",
            tests_csv_body="name,outcome,runtime,stacktrace\na.b.ClsTest#testIt,FAIL,1,\n",
            matrix_body="0 -\n",
            spectra_body="name\n",
        )
        (proc / FAULTS_FIRST_CSV).write_text("MARKER\n")
        (proc / FAULTS_FIRST_TXT).write_text("MARKER\n")

        save_first_fault_lines(_stub_repo(proc), skip_existing=True)

        assert (proc / FAULTS_FIRST_CSV).read_text() == "MARKER\n"
        assert (proc / FAULTS_FIRST_TXT).read_text() == "MARKER\n"


# ---------------------------------------------------------------------------
# Real Lang/1 fixture — single trigger test ⇒ faults_first == faults
# ---------------------------------------------------------------------------


REAL_PROCESSED = get_processed_dir("Lang", 1)


@pytest.mark.skipif(
    not (REAL_PROCESSED / "faults.csv").exists()
    or not (REAL_PROCESSED / "method_signatures.csv").exists(),
    reason="Lang/1 processed data not available",
)
class TestRealLang1:
    def test_single_trigger_equals_faults(self, tmp_path: Path) -> None:
        # Copy required files into tmp_path to avoid mutating the real dir.
        import shutil

        for name in (
            "faults.csv",
            "faults.txt",
            "method_signatures.csv",
            "trigger_tests",
        ):
            shutil.copy(REAL_PROCESSED / name, tmp_path / name)
        sfl_src = REAL_PROCESSED / "sfl" / "sfl" / "txt"
        sfl_dst = tmp_path / "sfl" / "sfl" / "txt"
        sfl_dst.mkdir(parents=True)
        for name in ("tests.csv", "matrix.txt", "spectra.csv"):
            shutil.copy(sfl_src / name, sfl_dst / name)

        save_first_fault_lines(_stub_repo(tmp_path))

        orig_rows = list(csv.DictReader((tmp_path / "faults.csv").open(encoding="utf-8")))
        first_rows = list(csv.DictReader((tmp_path / FAULTS_FIRST_CSV).open(encoding="utf-8")))
        # Lang/1 has only one trigger test, so the filter is a no-op.
        assert first_rows == orig_rows
