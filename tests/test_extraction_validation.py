"""Tests for src.extraction.validation output checks."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.config import OCHIAI_RANKING_FILE, SFL_SUBDIR, SPECTRA_FILE, TESTS_FILE
from src.extraction.d4j import EXPORT_PROPERTIES
from src.extraction.validation import validate_extraction_outputs


def _create_valid_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "processed" / "Chart" / "1"
    out.mkdir(parents=True)

    for prop in EXPORT_PROPERTIES:
        (out / prop).write_text("value\n")

    (out / "all_tests.txt").write_text("org.example.FooTest#testA\n")
    (out / "relevant_tests.txt").write_text("org.example.FooTest#testA\n")
    (out / "junit_tests.txt").write_text("JUNIT,org.example.FooTest#testA\n")
    (out / "failing_tests").write_text("--- org.example.FooTest::testA\nstack\n")
    (out / "failing_tests.txt").write_text("org.example.FooTest::testA\n")

    sfl_dir = out / SFL_SUBDIR
    sfl_dir.mkdir(parents=True)
    (sfl_dir / SPECTRA_FILE).write_text("name\norg.example$Foo#doIt():10\n")
    (sfl_dir / "matrix.txt").write_text("1 +\n")
    (sfl_dir / TESTS_FILE).write_text(
        "name,outcome,runtime,stacktrace\norg.example.FooTest#testA,PASS,1,\n"
    )
    (sfl_dir / OCHIAI_RANKING_FILE).write_text(
        "name;suspiciousness_value\norg.example$Foo#doIt():10;1.0\n"
    )

    (out / "faults.txt").write_text("org.example.Foo 10\n")
    (out / "faults_first.csv").write_text("path,line,signature\n")
    (out / "faults_first.txt").write_text("")
    (out / "bug_report.json").write_text(
        json.dumps(
            {
                "title": "NullPointerException in doIt",
                "description": "Calling doIt with null fails.",
                "url": "https://example.test/issue/1",
                "raw": "raw body",
            }
        )
    )

    return out


def test_validate_extraction_outputs_success(tmp_path: Path) -> None:
    output_dir = _create_valid_output_dir(tmp_path)

    issues = validate_extraction_outputs(
        output_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
    )

    assert issues == []


def test_validate_extraction_outputs_missing_required_file(tmp_path: Path) -> None:
    output_dir = _create_valid_output_dir(tmp_path)
    (output_dir / SFL_SUBDIR / "matrix.txt").unlink()

    issues = validate_extraction_outputs(
        output_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
    )

    assert any(issue["severity"] == "error" and issue["file"] == "matrix.txt" for issue in issues)


def test_validate_extraction_outputs_missing_bug_report_warns(tmp_path: Path) -> None:
    output_dir = _create_valid_output_dir(tmp_path)
    (output_dir / "bug_report.json").unlink()

    issues = validate_extraction_outputs(
        output_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
    )

    assert any(
        issue["severity"] == "warning" and issue["file"] == "bug_report.json" for issue in issues
    )


def test_validate_extraction_outputs_malformed_faults(tmp_path: Path) -> None:
    output_dir = _create_valid_output_dir(tmp_path)
    (output_dir / "faults.txt").write_text("org.example.Foo#10\n")

    issues = validate_extraction_outputs(
        output_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
    )

    assert any(issue["severity"] == "error" and issue["file"] == "faults.txt" for issue in issues)


def test_validate_extraction_outputs_missing_failing_tests_routes_to_failing_tag(
    tmp_path: Path,
) -> None:
    output_dir = _create_valid_output_dir(tmp_path)
    (output_dir / "failing_tests.txt").unlink()

    issues = validate_extraction_outputs(
        output_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
    )

    assert any(
        issue["severity"] == "error" and issue["file"] == "failing_tests.txt" for issue in issues
    )
