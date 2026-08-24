"""Tests for BugsInPy metadata parsing (src.extraction.bugsinpy).

These cover the pure host-metadata parsers — no container or checkout needed.
"""

from __future__ import annotations

from pathlib import Path

from src.extraction.bugsinpy import (
    _make_trigger_id,
    _modified_files,
    _parse_trigger_tests,
    _path_to_module,
)

# ---------------------------------------------------------------------------
# _path_to_module
# ---------------------------------------------------------------------------


class TestPathToModule:
    def test_plain_module(self) -> None:
        assert _path_to_module("test/test_InfoExtractor.py") == "test.test_InfoExtractor"

    def test_package_init_collapses(self) -> None:
        assert _path_to_module("pkg/sub/__init__.py") == "pkg.sub"

    def test_strips_leading_slash(self) -> None:
        assert _path_to_module("/a/b.py") == "a.b"


# ---------------------------------------------------------------------------
# _parse_trigger_tests
# ---------------------------------------------------------------------------


class TestParseTriggerTests:
    def test_unittest_class_method(self) -> None:
        text = (
            "python -m unittest -q test.test_InfoExtractor.TestInfoExtractor.test_parse_mpd_formats"
        )
        assert _parse_trigger_tests(text, "test/test_InfoExtractor.py") == [
            "test.test_InfoExtractor$TestInfoExtractor::test_parse_mpd_formats"
        ]

    def test_unittest_module_level_function(self) -> None:
        text = "python -m unittest tests.test_pysnooper.test_file_output"
        assert _parse_trigger_tests(text, "tests/test_pysnooper.py") == [
            "tests.test_pysnooper::test_file_output"
        ]

    def test_pytest_node_class(self) -> None:
        text = "python -m pytest tests/test_x.py::TestX::test_a"
        assert _parse_trigger_tests(text, "tests/test_x.py") == ["tests.test_x$TestX::test_a"]

    def test_pytest_node_function(self) -> None:
        text = "pytest tests/test_x.py::test_a"
        assert _parse_trigger_tests(text, "tests/test_x.py") == ["tests.test_x::test_a"]

    def test_no_match_returns_empty(self) -> None:
        assert _parse_trigger_tests("python -m pytest", "tests/test_x.py") == []


# ---------------------------------------------------------------------------
# _make_trigger_id
# ---------------------------------------------------------------------------


class TestMakeTriggerId:
    def test_class_method(self) -> None:
        assert _make_trigger_id("a.b", ["Cls", "m"]) == "a.b$Cls::m"

    def test_nested_class(self) -> None:
        assert _make_trigger_id("a.b", ["Outer", "Inner", "m"]) == "a.b$Outer.Inner::m"

    def test_module_level(self) -> None:
        assert _make_trigger_id("a.b", ["test_x"]) == "a.b::test_x"

    def test_empty_returns_none(self) -> None:
        assert _make_trigger_id("a.b", []) is None


# ---------------------------------------------------------------------------
# _modified_files
# ---------------------------------------------------------------------------


class TestModifiedFiles:
    def test_from_unified_diff(self) -> None:
        patch = (
            "diff --git a/youtube_dl/extractor/common.py b/youtube_dl/extractor/common.py\n"
            "index 3b79b8cb4..35d427eec 100644\n"
            "--- a/youtube_dl/extractor/common.py\n"
            "+++ b/youtube_dl/extractor/common.py\n"
        )
        assert _modified_files(patch, None) == ["youtube_dl/extractor/common.py"]

    def test_patchfile_info_fallback(self) -> None:
        assert _modified_files(None, "pkg/a.py;pkg/b.py") == ["pkg/a.py", "pkg/b.py"]

    def test_only_python_files(self) -> None:
        assert _modified_files(None, "pkg/a.py;README.md;pkg/b.py") == ["pkg/a.py", "pkg/b.py"]

    def test_diff_preferred_over_patchfile(self) -> None:
        patch = "diff --git a/x.py b/x.py\n"
        assert _modified_files(patch, "other.py") == ["x.py"]


# ---------------------------------------------------------------------------
# Bug-report sourcing helpers (src.extraction.bug_report)
# ---------------------------------------------------------------------------


class TestBugReportHelpers:
    def test_owner_repo(self) -> None:
        from src.extraction.bug_report import _owner_repo

        assert _owner_repo("https://github.com/ytdl-org/youtube-dl") == ("ytdl-org", "youtube-dl")
        assert _owner_repo("https://github.com/psf/black.git") == ("psf", "black")
        assert _owner_repo("https://github.com/cool-RR/PySnooper/") == ("cool-RR", "PySnooper")

    def test_owner_repo_invalid(self) -> None:
        from src.extraction.bug_report import _owner_repo

        assert _owner_repo("https://example.com/not/github") is None

    def test_minimal_report_schema(self) -> None:
        from src.extraction.bug_report import _minimal_bip_report

        r = _minimal_bip_report("https://github.com/o/r", "abc123", "Fix bug\n\nDetails here")
        # A fetched message yields a complete report with no error marker.
        assert sorted(r) == ["description", "raw", "title", "url"]
        assert r["url"] == "https://github.com/o/r/commit/abc123"
        assert r["title"] == "Fix bug"
        assert r["raw"] == "Fix bug\n\nDetails here"

    def test_minimal_report_empty_message_is_error_marked(self) -> None:
        from src.extraction.bug_report import _minimal_bip_report

        # An unavailable commit message (offline / rate-limited / bad .gh_token -> 401) must
        # tag the report with an `error` key so validation degrades it to a warning, not a hard
        # failure of the whole bug.
        r = _minimal_bip_report("https://github.com/o/r/", "deadbeef", None)
        assert r["title"] == ""
        assert r["description"] == ""
        assert r["url"] == "https://github.com/o/r/commit/deadbeef"
        assert "error" in r

    def test_failed_bug_report_is_warning_not_error(self, tmp_path: Path) -> None:
        """A fetch-failed (error-marked) bug report must not produce a validation *error*."""
        import json

        from src.extraction.bug_report import _minimal_bip_report
        from src.extraction.validation import validate_extraction_outputs

        report = _minimal_bip_report("https://github.com/o/r", "deadbeef", None)
        (tmp_path / "bug_report.json").write_text(json.dumps(report), encoding="utf-8")
        issues = validate_extraction_outputs(
            tmp_path,
            expect_gzoltar=False,
            expect_faults=False,
            expect_bug_report=True,
            dataset="bugsinpy",
        )
        report_issues = [i for i in issues if i["file"] == "bug_report.json"]
        assert report_issues, "expected a bug_report.json issue"
        assert all(i["severity"] == "warning" for i in report_issues)
