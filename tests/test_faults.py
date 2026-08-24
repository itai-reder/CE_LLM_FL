"""Tests for src.extraction.faults — diff_start_lines() and save_fault_lines()."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.faults import (
    FAULTS_CSV_HEADER,
    _match_src_path,
    _parse_patch_buggy_lines,
    diff_start_lines,
    save_fault_lines,
)

# ---------------------------------------------------------------------------
# BugsInPy: unified-diff parsing (_parse_patch_buggy_lines / _match_src_path)
# ---------------------------------------------------------------------------


class TestParsePatchBuggyLines:
    def test_records_removed_buggy_lines(self) -> None:
        patch = (
            "diff --git a/pkg/mod.py b/pkg/mod.py\n"
            "index abc..def 100644\n"
            "--- a/pkg/mod.py\n"
            "+++ b/pkg/mod.py\n"
            "@@ -10,4 +10,3 @@ class C:\n"
            "     context_a\n"
            "-    removed_1\n"
            "-    removed_2\n"
            "+    added_1\n"
            "     context_b\n"
        )
        # context_a is line 10 -> removals fall on buggy lines 11 and 12.
        assert _parse_patch_buggy_lines(patch) == {"pkg/mod.py": [11, 12]}

    def test_pure_insertion_records_hunk_anchor(self) -> None:
        patch = (
            "diff --git a/pkg/mod.py b/pkg/mod.py\n"
            "--- a/pkg/mod.py\n"
            "+++ b/pkg/mod.py\n"
            "@@ -5,2 +5,3 @@\n"
            "     ctx1\n"
            "+    new_line\n"
            "     ctx2\n"
        )
        assert _parse_patch_buggy_lines(patch) == {"pkg/mod.py": [5]}

    def test_ignores_non_python_files(self) -> None:
        patch = (
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n+++ b/README.md\n"
            "@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        assert _parse_patch_buggy_lines(patch) == {}


class TestMatchSrcPath:
    def test_exact_match(self) -> None:
        known = {"youtube_dl/extractor/common.py"}
        assert _match_src_path("youtube_dl/extractor/common.py", known) == (
            "youtube_dl/extractor/common.py"
        )

    def test_strips_build_prefix(self) -> None:
        # ansible: patch path is repo-relative (lib/...), src root is build/lib.
        known = {"ansible/galaxy/collection.py"}
        assert _match_src_path("lib/ansible/galaxy/collection.py", known) == (
            "ansible/galaxy/collection.py"
        )

    def test_fallback_to_patch_path(self) -> None:
        assert _match_src_path("a/b.py", set()) == "a/b.py"


@pytest.fixture()
def buggy_file(tmp_path: Path) -> Path:
    """Create a simple 'buggy' Java file."""
    f = tmp_path / "Buggy.java"
    f.write_text("line1\nline2\nbuggy_line3\nline4\nline5\n")
    return f


@pytest.fixture()
def fixed_file(tmp_path: Path) -> Path:
    """Create a simple 'fixed' Java file."""
    f = tmp_path / "Fixed.java"
    f.write_text("line1\nline2\nfixed_line3\nline4\nline5\n")
    return f


class TestDiffStartLines:
    """Tests for diff_start_lines()."""

    def test_single_change(self, buggy_file: Path, fixed_file: Path) -> None:
        result = diff_start_lines(buggy_file, fixed_file)
        assert result == [3]

    def test_identical_files(self, buggy_file: Path) -> None:
        result = diff_start_lines(buggy_file, buggy_file)
        assert result == []

    def test_both_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Both files do not exist"):
            diff_start_lines(
                tmp_path / "nonexistent1.java",
                tmp_path / "nonexistent2.java",
            )

    def test_buggy_missing_returns_empty(self, tmp_path: Path, fixed_file: Path) -> None:
        result = diff_start_lines(tmp_path / "missing.java", fixed_file)
        assert result == []

    def test_fixed_missing_returns_empty(self, buggy_file: Path, tmp_path: Path) -> None:
        result = diff_start_lines(buggy_file, tmp_path / "missing.java")
        assert result == []

    def test_multi_change(self, tmp_path: Path) -> None:
        f1 = tmp_path / "f1.java"
        f1.write_text("a\nb\nc\nd\ne\n")
        f2 = tmp_path / "f2.java"
        f2.write_text("a\nBB\nc\nDD\ne\n")
        result = diff_start_lines(f1, f2)
        assert result == [2, 4]

    def test_deletion_hunk(self, tmp_path: Path) -> None:
        f1 = tmp_path / "f1.java"
        f1.write_text("a\nb\nc\nd\n")
        f2 = tmp_path / "f2.java"
        f2.write_text("a\nd\n")
        result = diff_start_lines(f1, f2)
        # Lines 2-3 deleted from f1; diff header is "2,3d1" → first line = 2
        assert 2 in result

    def test_addition_hunk(self, tmp_path: Path) -> None:
        f1 = tmp_path / "f1.java"
        f1.write_text("a\nb\n")
        f2 = tmp_path / "f2.java"
        f2.write_text("a\nX\nY\nb\n")
        result = diff_start_lines(f1, f2)
        # Addition after line 1 in f1; diff header is "1a2,3" → first line = 1
        assert 1 in result


def test_save_fault_lines_always_cleans_fixed_checkout(tmp_path: Path) -> None:
    buggy_repo = MagicMock()
    buggy_repo.output_dir = tmp_path / "processed" / "Chart" / "1"
    buggy_repo.output_dir.mkdir(parents=True)
    buggy_repo.project = "Chart"
    buggy_repo.bug_id = 1
    buggy_repo.get_modified_classes.return_value = ["org.example.Foo"]
    buggy_repo.classpath_from_class_signature.return_value = tmp_path / "Foo.java"

    fixed_repo = MagicMock()
    fixed_repo.classpath_from_class_signature.return_value = tmp_path / "FooFixed.java"

    with (
        patch("src.extraction.faults.D4JRepo", return_value=fixed_repo),
        patch("src.extraction.faults.diff_start_lines", side_effect=RuntimeError("diff failed")),
        pytest.raises(RuntimeError, match="diff failed"),
    ):
        save_fault_lines(buggy_repo, skip_existing=False)

    fixed_repo.remove_repo.assert_called_once()


# ---------------------------------------------------------------------------
# faults.csv
# ---------------------------------------------------------------------------


def _write_corpus_signatures(processed_dir: Path, rows: list[tuple[str, str, int, int]]) -> None:
    """Helper: write method_signatures.csv with the given (corpus_id, path, start, end) rows."""
    csv_path = processed_dir / "method_signatures.csv"
    csv_path.write_text(
        "corpus_id;path;startLine;endLine\n"
        + "\n".join(f"{cid};{p};{s};{e}" for cid, p, s, e in rows)
        + "\n",
        encoding="utf-8",
    )


def _make_fault_repo(tmp_path: Path, fault_class: str = "org.example.Foo"):  # type: ignore[no-untyped-def]
    """Build a MagicMock buggy repo with one modified class and a 1-line diff."""
    buggy_repo = MagicMock()
    buggy_repo.output_dir = tmp_path / "processed" / "Chart" / "1"
    buggy_repo.output_dir.mkdir(parents=True)
    buggy_repo.project = "Chart"
    buggy_repo.bug_id = 1
    buggy_repo.get_modified_classes.return_value = [fault_class]
    buggy_repo.classpath_from_class_signature.return_value = tmp_path / "Foo.java"
    return buggy_repo


def _patch_fixed_repo_and_diff(diff_lines: list[int]):  # type: ignore[no-untyped-def]
    fixed_repo = MagicMock()
    fixed_repo.classpath_from_class_signature.return_value = Path("/tmp/FooFixed.java")
    return (
        patch("src.extraction.faults.D4JRepo", return_value=fixed_repo),
        patch("src.extraction.faults.diff_start_lines", return_value=diff_lines),
        fixed_repo,
    )


def test_faults_csv_written_with_header_and_rows(tmp_path: Path) -> None:
    repo = _make_fault_repo(tmp_path)
    _write_corpus_signatures(
        repo.output_dir,
        [
            ("org.example$Foo.bar()", "org/example/Foo.java", 10, 20),
            ("org.example$Foo.baz()", "org/example/Foo.java", 30, 40),
        ],
    )

    p1, p2, _ = _patch_fixed_repo_and_diff([15, 35])
    with p1, p2:
        save_fault_lines(repo, skip_existing=False)

    csv_path = repo.output_dir / "faults.csv"
    assert csv_path.exists()
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(FAULTS_CSV_HEADER)
    # Rows: (path, line, signature)
    assert rows[1] == ["org/example/Foo.java", "15", "org.example$Foo.bar()"]
    assert rows[2] == ["org/example/Foo.java", "35", "org.example$Foo.baz()"]


def test_faults_csv_empty_signature_when_line_outside_any_method(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _make_fault_repo(tmp_path)
    _write_corpus_signatures(
        repo.output_dir,
        [("org.example$Foo.bar()", "org/example/Foo.java", 10, 20)],
    )

    p1, p2, _ = _patch_fixed_repo_and_diff([5])  # Line 5 is outside any method
    caplog.set_level(logging.WARNING, logger="src.extraction.faults")
    with p1, p2:
        save_fault_lines(repo, skip_existing=False)

    with (repo.output_dir / "faults.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[1] == ["org/example/Foo.java", "5", ""]
    assert any("No method maps to org.example.Foo:5" in r.message for r in caplog.records)


def test_faults_csv_picks_innermost_method_for_nested_overlap(tmp_path: Path) -> None:
    repo = _make_fault_repo(tmp_path)
    # Outer.foo spans 10-100; inner lambda spans 50-60. Fault at 55 → inner wins.
    _write_corpus_signatures(
        repo.output_dir,
        [
            ("org.example$Foo.foo()", "org/example/Foo.java", 10, 100),
            ("org.example$Foo.lambda$0()", "org/example/Foo.java", 50, 60),
        ],
    )

    p1, p2, _ = _patch_fixed_repo_and_diff([55])
    with p1, p2:
        save_fault_lines(repo, skip_existing=False)

    with (repo.output_dir / "faults.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[1][2] == "org.example$Foo.lambda$0()"


def test_faults_csv_warns_and_continues_when_signatures_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _make_fault_repo(tmp_path)
    # NOTE: method_signatures.csv intentionally absent

    p1, p2, _ = _patch_fixed_repo_and_diff([42])
    caplog.set_level(logging.WARNING, logger="src.extraction.faults")
    with p1, p2:
        save_fault_lines(repo, skip_existing=False)

    csv_path = repo.output_dir / "faults.csv"
    assert csv_path.exists()
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    # Path falls back to the FQCN-derived path; signature is empty
    assert rows[1] == ["org/example/Foo.java", "42", ""]
    assert any("method_signatures.csv not found" in r.message for r in caplog.records)


def test_faults_skip_existing_re_runs_when_csv_missing(tmp_path: Path) -> None:
    """skip_existing must require BOTH faults.txt and faults.csv."""
    repo = _make_fault_repo(tmp_path)
    _write_corpus_signatures(
        repo.output_dir, [("org.example$Foo.bar()", "org/example/Foo.java", 1, 50)]
    )
    # faults.txt exists but faults.csv does not — re-run must produce CSV.
    (repo.output_dir / "faults.txt").write_text("org.example.Foo 10\n")

    p1, p2, _ = _patch_fixed_repo_and_diff([10])
    with p1, p2:
        save_fault_lines(repo, skip_existing=True)

    assert (repo.output_dir / "faults.csv").exists()


def test_faults_skip_existing_skips_when_both_files_present(tmp_path: Path) -> None:
    repo = _make_fault_repo(tmp_path)
    (repo.output_dir / "faults.txt").write_text("stale\n")
    (repo.output_dir / "faults.csv").write_text("path,line,signature\nstale,1,sig\n")

    save_fault_lines(repo, skip_existing=True)

    # Must NOT have called get_modified_classes / fixed checkout
    repo.get_modified_classes.assert_not_called()
    # Files unchanged
    assert (repo.output_dir / "faults.txt").read_text() == "stale\n"
