"""Tests for fl_methods.update_tracker (static-analysis tracker backfill)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from update_tracker import (
    _count_method_hits_ranking,
    _count_method_hits_top5,
    _count_stmt_hits,
    _infer_extraction,
    _infer_fl,
    _infer_sr,
    compute_coverage,
    file_to_step,
    update_single_bug,
)

from src.common import tracker as tracker_mod
from src.common.tracker import TRACKER_FILENAME, _empty_tracker, save_tracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def processed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a realistic processed-bug directory with hand-crafted outputs."""
    bug_dir = tmp_path / "TestProj" / "42"
    bug_dir.mkdir(parents=True)

    # Redirect tracker IO to this dir
    def fake_get_processed_dir(project: str, bug_id, *, dataset: str = "defects4j") -> Path:
        return bug_dir

    monkeypatch.setattr(tracker_mod, "get_processed_dir", fake_get_processed_dir)

    return bug_dir


def _populate_extraction(bug_dir: Path) -> None:
    """Populate extraction outputs in the fixture directory."""
    from src.extraction.d4j import EXPORT_PROPERTIES

    for prop in EXPORT_PROPERTIES:
        (bug_dir / prop).write_text(f"value-of-{prop}\n")

    # method_signatures.csv
    corpus_csv = bug_dir / "method_signatures.csv"
    corpus_csv.write_text(
        "corpus_id;path;startLine;endLine\n"
        "org$Foo.bar(int);org/Foo.java;10;20\n"
        "org$Foo.baz();org/Foo.java;25;35\n"
    )

    # Test files
    (bug_dir / "relevant_tests.txt").write_text("com.example.FooTest#testBar\n")
    (bug_dir / "junit_tests.txt").write_text("JUNIT,com.example.FooTest#testBar\n")
    (bug_dir / "all_tests.txt").write_text("com.example.FooTest#testBar\n")
    (bug_dir / "failing_tests").write_text("--- com.example.FooTest::testBar\n  stack trace\n")
    (bug_dir / "failing_tests.txt").write_text("com.example.FooTest::testBar\n")

    # GZoltar
    sfl_dir = bug_dir / "sfl" / "sfl" / "txt"
    sfl_dir.mkdir(parents=True)
    (sfl_dir / "spectra.csv").write_text("name\norg.Foo#bar(int):10\n")
    (sfl_dir / "matrix.txt").write_text("0 1\n1 0\n")
    (sfl_dir / "tests.csv").write_text(
        "name,outcome,runtime,stacktrace\ncom.example.FooTest#testBar,PASS,1.0,\n"
    )
    (sfl_dir / "ochiai.ranking.csv").write_text("name;suspiciousness\norg.Foo#bar(int):10;0.5\n")

    # Faults
    (bug_dir / "faults.txt").write_text("org/Foo.java 15\n")
    (bug_dir / "faults.csv").write_text("path,line,signature\norg/Foo.java,15,org$Foo.bar(int)\n")

    # Bug report
    (bug_dir / "bug_report.json").write_text(
        json.dumps({"title": "NullPointerException", "description": "NPE in Foo.bar"})
    )


def _populate_fl(bug_dir: Path) -> None:
    """Populate FL outputs."""
    # Ochiai
    ochiai_dir = bug_dir / "FL" / "Ochiai"
    ochiai_dir.mkdir(parents=True)
    (ochiai_dir / "stmt-susps.txt").write_text("Statement,Suspiciousness\norg.Foo#15,0.7\n")
    (ochiai_dir / "sbfl_ochiai.json").write_text(json.dumps({"org.Foo#15": 0.7}))

    # BoostN
    boostn_dir = bug_dir / "FL" / "BoostN"
    boostn_dir.mkdir(parents=True)
    (boostn_dir / "boostn-method-susps.csv").write_text(
        "Signature,Suspiciousness\norg$Foo.bar(int),0.6\n"
    )
    (boostn_dir / "boostn.json").write_text(json.dumps({"org$Foo.bar(int)": 0.6}))

    # SBIR
    sbir_dir = bug_dir / "FL" / "SBIR"
    sbir_dir.mkdir(parents=True)
    (sbir_dir / "sbir-susps.txt").write_text("Statement,Suspiciousness\norg.Foo#15,0.65\n")
    (sbir_dir / "sbir_scores.json").write_text(json.dumps({"org.Foo#15": 0.65}))

    # Rankings
    rankings_dir = bug_dir / "FlexFL" / "SR" / "rankings"
    rankings_dir.mkdir(parents=True)
    (rankings_dir / "top15.txt").write_text("org$Foo.bar(int)\n")
    (rankings_dir / "top15.csv").write_text(
        "rank,method,signature,path,startLine,endLine\n"
        "1,Ochiai,org$Foo.bar(int),org/Foo.java,10,20\n"
    )
    for source_csv in ("ochiai.csv", "sbir.csv", "boostn.csv"):
        (rankings_dir / source_csv).write_text(
            "rank;signature;path;startLine;endLine;score\n"
            "1;org$Foo.bar(int);org/Foo.java;10;20;0.7\n"
        )


def _populate_sr(bug_dir: Path) -> None:
    """Populate Agent4SR output directory."""
    sr_dir = bug_dir / "FlexFL" / "SR" / "Agent4SR" / "llama3.1_8b"
    sr_dir.mkdir(parents=True)
    (sr_dir / "sr_result.json").write_text(
        json.dumps(
            {
                "model": "llama3.1:8b",
                "temperature": 0.0,
                "iterations": 10,
                "base_url": "http://localhost:11434",
                "input": ["bug_report", "ochiai", "boostn", "sbir"],
            }
        )
    )
    (sr_dir / "top5.txt").write_text("org$Foo.bar(int)\norg$Foo.baz()\n")


# ---------------------------------------------------------------------------
# Tests: file_to_step routing
# ---------------------------------------------------------------------------


class TestFileToStep:
    def test_property_maps_to_properties(self) -> None:
        assert file_to_step("dir.src.classes") == "properties"

    def test_test_file_maps_to_relevant_tests(self) -> None:
        assert file_to_step("relevant_tests.txt") == "relevant_tests"

    def test_failing_test_files_map_to_failing_tests(self) -> None:
        assert file_to_step("failing_tests") == "failing_tests"
        assert file_to_step("failing_tests.txt") == "failing_tests"

    def test_gzoltar_file_maps_to_gzoltar(self) -> None:
        assert file_to_step("spectra.csv") == "gzoltar"

    def test_faults_maps_to_faults(self) -> None:
        assert file_to_step("faults.txt") == "faults"

    def test_bug_report_maps_to_bug_report(self) -> None:
        assert file_to_step("bug_report.json") == "bug_report"

    def test_unknown_file_maps_to_unknown(self) -> None:
        assert file_to_step("random_file.xyz") == "unknown"


# ---------------------------------------------------------------------------
# Tests: extraction inference
# ---------------------------------------------------------------------------


class TestInferExtraction:
    def test_all_steps_completed(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        t = _empty_tracker()
        _infer_extraction(processed_dir, t)
        assert "properties" in t["extraction"]["completed"]
        assert "signatures" in t["extraction"]["completed"]
        assert "relevant_tests" in t["extraction"]["completed"]
        assert "failing_tests" in t["extraction"]["completed"]
        assert "gzoltar" in t["extraction"]["completed"]
        assert "faults" in t["extraction"]["completed"]
        assert "bug_report" in t["extraction"]["completed"]

    def test_missing_properties(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        (processed_dir / "dir.src.classes").unlink()
        t = _empty_tracker()
        _infer_extraction(processed_dir, t)
        assert "properties" not in t["extraction"]["completed"]

    def test_missing_bug_report(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        (processed_dir / "bug_report.json").unlink()
        t = _empty_tracker()
        _infer_extraction(processed_dir, t)
        assert "bug_report" not in t["extraction"]["completed"]
        # Should have a validation warning instead
        assert "bug_report" in t["extraction"]["warnings"]

    def test_bug_report_with_error_key(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        (processed_dir / "bug_report.json").write_text(json.dumps({"error": "No tracker URL"}))
        t = _empty_tracker()
        _infer_extraction(processed_dir, t)
        assert "bug_report" not in t["extraction"]["completed"]

    def test_failing_tests_backfilled_when_raw_exists(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        (processed_dir / "failing_tests.txt").unlink()
        t = _empty_tracker()
        _infer_extraction(processed_dir, t)
        assert "relevant_tests" in t["extraction"]["completed"]
        assert "failing_tests" in t["extraction"]["completed"]
        assert (processed_dir / "failing_tests.txt").read_text() == (
            "com.example.FooTest::testBar\n"
        )

    def test_relevant_tests_independent_when_failing_files_missing(
        self, processed_dir: Path
    ) -> None:
        _populate_extraction(processed_dir)
        (processed_dir / "failing_tests").unlink()
        (processed_dir / "failing_tests.txt").unlink()
        t = _empty_tracker()
        _infer_extraction(processed_dir, t)
        assert "relevant_tests" in t["extraction"]["completed"]
        assert "failing_tests" not in t["extraction"]["completed"]


# ---------------------------------------------------------------------------
# Tests: FL inference
# ---------------------------------------------------------------------------


class TestInferFL:
    def test_all_fl_steps_completed(self, processed_dir: Path) -> None:
        _populate_fl(processed_dir)
        t = _empty_tracker()
        _infer_fl(processed_dir, t)
        assert "ochiai" in t["fl"]["completed"]
        assert "boostn" in t["fl"]["completed"]
        assert "sbir" in t["fl"]["completed"]
        assert "top15" in t["fl"]["completed"]

    def test_missing_boostn(self, processed_dir: Path) -> None:
        _populate_fl(processed_dir)
        (processed_dir / "FL" / "BoostN" / "boostn.json").unlink()
        t = _empty_tracker()
        _infer_fl(processed_dir, t)
        assert "boostn" not in t["fl"]["completed"]


# ---------------------------------------------------------------------------
# Tests: SR inference
# ---------------------------------------------------------------------------


class TestInferSR:
    def test_infers_sr_from_result_json(self, processed_dir: Path) -> None:
        _populate_sr(processed_dir)
        t = _empty_tracker()
        _infer_sr(processed_dir, t)
        assert "llama3.1_8b" in t["sr"]
        assert t["sr"]["llama3.1_8b"]["model"] == "llama3.1:8b"

    def test_no_sr_dir(self, processed_dir: Path) -> None:
        t = _empty_tracker()
        _infer_sr(processed_dir, t)
        assert t["sr"] == {}


# ---------------------------------------------------------------------------
# Tests: coverage computation
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_counts_from_faults_csv(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        _populate_fl(processed_dir)
        t = _empty_tracker()
        compute_coverage(processed_dir, t)
        assert t["coverage"]["s_faults"]["count"] == 1
        assert t["coverage"]["m_faults"]["count"] == 1
        assert t["coverage"]["r_tests"]["count"] == 1
        assert t["coverage"]["f_tests"]["count"] == 1

    def test_empty_dir_produces_zero_counts(self, processed_dir: Path) -> None:
        t = _empty_tracker()
        compute_coverage(processed_dir, t)
        assert t["coverage"].get("s_faults", {}).get("count", 0) == 0


# ---------------------------------------------------------------------------
# Tests: statement / method hit counting
# ---------------------------------------------------------------------------


class TestHitCounting:
    def test_count_stmt_hits(self, tmp_path: Path) -> None:
        stmt_csv = tmp_path / "stmt.txt"
        stmt_csv.write_text(
            "Statement,Suspiciousness\norg.Foo#10,0.5\norg.Foo#15,0.7\norg.Bar#20,0.3\n"
        )
        faults = {("org.Foo", 15), ("org.Bar", 20)}
        assert _count_stmt_hits(stmt_csv, faults) == 2

    def test_count_stmt_hits_no_matches(self, tmp_path: Path) -> None:
        stmt_csv = tmp_path / "stmt.txt"
        stmt_csv.write_text("Statement,Suspiciousness\norg.Foo#10,0.5\n")
        faults = {("org.Baz", 99)}
        assert _count_stmt_hits(stmt_csv, faults) == 0

    def test_count_method_hits_ranking(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ranking.csv"
        csv_path.write_text(
            "rank;signature;path;startLine;endLine;score\n"
            "1;org$Foo.bar(int);org/Foo.java;10;20;0.7\n"
            "2;org$Foo.baz();org/Foo.java;25;35;0.5\n"
        )
        faults = {"org$Foo.bar(int)"}
        assert _count_method_hits_ranking(csv_path, faults) == 1

    def test_count_method_hits_top5(self, tmp_path: Path) -> None:
        top5 = tmp_path / "top5.txt"
        top5.write_text("org$Foo.bar(int)\norg$Foo.baz()\n")
        faults = {"org$Foo.bar(int)", "org$Foo.baz()"}
        assert _count_method_hits_top5(top5, faults) == 2


# ---------------------------------------------------------------------------
# Tests: update_single_bug (integration)
# ---------------------------------------------------------------------------


class TestUpdateSingleBug:
    def test_full_pipeline_produces_tracker(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        _populate_fl(processed_dir)
        _populate_sr(processed_dir)
        tracker = update_single_bug(processed_dir, "TestProj", 42, no_backup=True)

        assert "properties" in tracker["extraction"]["completed"]
        assert "ochiai" in tracker["fl"]["completed"]
        assert "llama3.1_8b" in tracker["sr"]
        assert "s_faults" in tracker["coverage"]

        # Verify file on disk
        tracker_file = processed_dir / TRACKER_FILENAME
        assert tracker_file.exists()
        on_disk = json.loads(tracker_file.read_text())
        assert on_disk["schema_version"] == 2

    def test_dry_run_does_not_write(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        tracker = update_single_bug(processed_dir, "TestProj", 42, dry_run=True)
        assert "properties" in tracker["extraction"]["completed"]
        assert not (processed_dir / TRACKER_FILENAME).exists()

    def test_backup_created_by_default(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        # Create an initial tracker
        save_tracker(_empty_tracker(), "TestProj", 42)
        assert (processed_dir / TRACKER_FILENAME).exists()

        update_single_bug(processed_dir, "TestProj", 42)
        # Backup should exist
        backups = list(processed_dir.glob("tracker.*.json"))
        assert len(backups) == 1
        assert backups[0].name.startswith("tracker.") and backups[0].name.endswith(".json")

    def test_no_backup_flag(self, processed_dir: Path) -> None:
        _populate_extraction(processed_dir)
        save_tracker(_empty_tracker(), "TestProj", 42)
        update_single_bug(processed_dir, "TestProj", 42, no_backup=True)
        backups = list(processed_dir.glob("tracker.*.json"))
        # No backup files (only tracker.json itself)
        assert len(backups) == 0

    def test_empty_dir_still_writes_tracker(self, processed_dir: Path) -> None:
        tracker = update_single_bug(processed_dir, "TestProj", 42, no_backup=True)
        assert tracker["extraction"]["completed"] == []
        assert tracker["fl"]["completed"] == []
        assert (processed_dir / TRACKER_FILENAME).exists()
