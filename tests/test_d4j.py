"""Tests for src.extraction.d4j — D4JRepo initialisation and path helpers.

Only tests pure logic that does not require a running Docker container.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.common.config import (
    get_benchmark_fixed_root,
    get_benchmark_processed_root,
    get_benchmark_repos_root,
)
from src.extraction.d4j import D4JRepo, write_parsed_failing_tests


class TestD4JRepoInit:
    """Tests for D4JRepo constructor and path properties."""

    def test_buggy_version_flag(self) -> None:
        repo = D4JRepo("Chart", 1)
        assert repo.version_flag == "1b"

    def test_fixed_version_flag(self) -> None:
        repo = D4JRepo("Chart", 1, buggy=False)
        assert repo.version_flag == "1f"

    def test_buggy_repo_dir(self) -> None:
        repo = D4JRepo("Math", 42)
        assert repo.repo_dir == get_benchmark_repos_root() / "Math" / "42"

    def test_fixed_repo_dir(self) -> None:
        repo = D4JRepo("Math", 42, buggy=False)
        assert repo.repo_dir == get_benchmark_fixed_root() / "Math" / "42"

    def test_output_dir(self) -> None:
        repo = D4JRepo("Lang", 5)
        assert repo.output_dir == get_benchmark_processed_root() / "Lang" / "5"

    def test_output_dir_same_for_buggy_and_fixed(self) -> None:
        buggy = D4JRepo("Cli", 10)
        fixed = D4JRepo("Cli", 10, buggy=False)
        assert buggy.output_dir == fixed.output_dir

    def test_classpath_conversion_logic(self) -> None:
        # Verify the FQCN → path conversion logic used by classpath_from_class_signature
        sig = "org.example.MyClass"
        expected_rel = "org/example/MyClass.java"
        result_rel = sig.replace(".", "/") + ".java"
        assert result_rel == expected_rel

    def test_is_checked_out_false_by_default(self) -> None:
        repo = D4JRepo("Chart", 999)
        assert repo.is_checked_out() is False

    def test_project_and_bug_id_stored(self) -> None:
        repo = D4JRepo("Closure", 174)
        assert repo.project == "Closure"
        assert repo.bug_id == 174
        assert repo.buggy is True


def test_checkout_retries_on_chart_dir_layout_permission_error(tmp_path, monkeypatch) -> None:
    repo = D4JRepo("Chart", 26)
    repo.repo_dir = tmp_path / "repos" / "Chart" / "26"
    repo.output_dir = tmp_path / "processed" / "Chart" / "26"

    calls = {"count": 0}

    def _run_defects4j(_args):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("dir-layout.csv: Permission denied")

    fix_mock = MagicMock()
    monkeypatch.setattr("src.extraction.d4j.run_defects4j", _run_defects4j)
    monkeypatch.setattr("src.extraction.d4j.ensure_project_dir_layout_writable", fix_mock)

    repo.checkout(skip_existing=False)

    assert calls["count"] == 2
    fix_mock.assert_called_once_with("Chart")


def test_get_relevant_test_methods_matches_simple_class_names(tmp_path, monkeypatch) -> None:
    repo = D4JRepo("Collections", 1)
    repo.output_dir = tmp_path / "processed" / "Collections" / "1"
    repo.output_dir.mkdir(parents=True)

    monkeypatch.setattr(
        repo,
        "get_relevant_test_classes",
        lambda: ["org.apache.commons.collections.map.TestFlat3Map"],
    )
    monkeypatch.setattr(
        repo,
        "get_all_test_methods",
        lambda **kwargs: [
            "TestFlat3Map.bulkTestMapIterator.testEmptyMapIterator#testEmptyMapIterator",
            "org.apache.commons.collections.map.OtherTest#testOther",
        ],
    )

    methods = repo.get_relevant_test_methods()

    assert methods == ["org.apache.commons.collections.map.TestFlat3Map#testEmptyMapIterator"]
    assert (repo.output_dir / "relevant_tests.txt").read_text().splitlines() == methods
    assert (repo.output_dir / "junit_tests.txt").read_text().splitlines() == [
        "JUNIT,org.apache.commons.collections.map.TestFlat3Map#testEmptyMapIterator"
    ]


def test_get_relevant_test_methods_recomputes_when_skip_existing_false(
    tmp_path, monkeypatch
) -> None:
    repo = D4JRepo("Collections", 1)
    repo.output_dir = tmp_path / "processed" / "Collections" / "1"
    repo.output_dir.mkdir(parents=True)

    (repo.output_dir / "relevant_tests.txt").write_text("stale.Class#test\n")
    (repo.output_dir / "junit_tests.txt").write_text("JUNIT,stale.Class#test\n")

    monkeypatch.setattr(
        repo,
        "get_relevant_test_classes",
        lambda: ["org.example.RealTest"],
    )
    monkeypatch.setattr(
        repo,
        "get_all_test_methods",
        lambda **kwargs: ["org.example.RealTest#testFresh"],
    )

    methods = repo.get_relevant_test_methods(skip_existing=False)

    assert methods == ["org.example.RealTest#testFresh"]
    assert (repo.output_dir / "relevant_tests.txt").read_text().splitlines() == methods
    assert (repo.output_dir / "junit_tests.txt").read_text().splitlines() == [
        "JUNIT,org.example.RealTest#testFresh"
    ]


# ---------------------------------------------------------------------------
# _write_parsed_failing_tests
# ---------------------------------------------------------------------------


def test_parsed_failing_tests_extracts_marker_lines(tmp_path) -> None:
    output_dir = tmp_path / "processed" / "Lang" / "1"
    output_dir.mkdir(parents=True)

    (output_dir / "failing_tests").write_text(
        "--- org.apache.commons.lang3.SystemUtilsTest::testGetUserHome\n"
        "junit.framework.AssertionFailedError\n"
        "\tat org.junit.Assert.fail(Assert.java:86)\n"
        "\tat org.junit.Assert.assertTrue(Assert.java:41)\n"
        "--- org.apache.commons.lang3.math.NumberUtilsTest::TestLang747\n"
        'java.lang.NumberFormatException: For input string: "80000000"\n'
        "\tat java.lang.Integer.parseInt(Integer.java:583)\n"
    )

    assert write_parsed_failing_tests(output_dir)

    parsed = (output_dir / "failing_tests.txt").read_text().splitlines()
    assert parsed == [
        "org.apache.commons.lang3.SystemUtilsTest::testGetUserHome",
        "org.apache.commons.lang3.math.NumberUtilsTest::TestLang747",
    ]


def test_parsed_failing_tests_no_op_when_raw_missing(tmp_path) -> None:
    output_dir = tmp_path / "processed" / "Lang" / "1"
    output_dir.mkdir(parents=True)

    assert not write_parsed_failing_tests(output_dir)

    assert not (output_dir / "failing_tests.txt").exists()


def test_parsed_failing_tests_writes_empty_when_no_markers(tmp_path) -> None:
    """Stack-trace-only file (no `---` markers) yields an empty failing_tests.txt."""
    output_dir = tmp_path / "processed" / "Lang" / "1"
    output_dir.mkdir(parents=True)

    (output_dir / "failing_tests").write_text("just a log line\nanother\n")
    assert write_parsed_failing_tests(output_dir)

    assert (output_dir / "failing_tests.txt").read_text() == ""


def test_parsed_failing_tests_idempotent(tmp_path) -> None:
    repo = D4JRepo("Lang", 1)
    repo.output_dir = tmp_path / "processed" / "Lang" / "1"
    repo.output_dir.mkdir(parents=True)

    (repo.output_dir / "failing_tests").write_text("--- pkg.T::m\nstack\n")
    repo._write_parsed_failing_tests()
    first = (repo.output_dir / "failing_tests.txt").read_text()
    repo._write_parsed_failing_tests()
    second = (repo.output_dir / "failing_tests.txt").read_text()
    assert first == second == "pkg.T::m\n"
