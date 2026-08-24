"""Tests for the parallel BugsInPy extraction driver helpers (run_bip_extraction_all).

These cover the pure/host-side helpers — lane naming, per-bug log-dir isolation, lane-death
detection, bug enumeration, and the manifest summary — without spawning containers or subprocesses.
"""

from __future__ import annotations

from pathlib import Path

import run_bip_extraction_all as driver
from run_bip_extraction_all import (
    BugOutcome,
    _driver_out_signals_lane_death,
    _lane_name,
    _per_bug_log_dir,
    _resolve_pairs,
    _write_manifest,
)


def test_lane_name() -> None:
    assert _lane_name(0) == "bugsinpy-cefl-lane-0"
    assert _lane_name(4) == "bugsinpy-cefl-lane-4"


def test_per_bug_log_dir_is_isolated() -> None:
    a = _per_bug_log_dir("run1", "black", 1)
    b = _per_bug_log_dir("run1", "black", 2)
    c = _per_bug_log_dir("run1", "fastapi", 1)
    # A distinct directory per (project, bug) so same-project/same-second log files never collide.
    assert a != b != c
    assert a.parent == b.parent  # same project dir
    assert a.name == "1" and a.parent.name == "black"


def test_resolve_pairs_enumerates_all_bugs() -> None:
    pairs = _resolve_pairs(None, resume=False)
    assert len(pairs) == 501
    assert len({p for p, _ in pairs}) == 17
    assert ("cookiecutter", 4) in pairs


def test_resolve_pairs_restricts_projects() -> None:
    pairs = _resolve_pairs(["black"], resume=False)
    assert pairs
    assert all(p == "black" for p, _ in pairs)


def test_resolve_pairs_resume_filters_complete(monkeypatch) -> None:
    # Mark every even bug id "complete" -> resume keeps only odd ids.
    monkeypatch.setattr(driver, "is_bug_complete", lambda repo: repo.bug_id % 2 == 0)
    pairs = _resolve_pairs(["black"], resume=True)
    assert pairs
    assert all(b % 2 == 1 for _, b in pairs)


def test_driver_out_lane_death_detection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "EXTRACTION_LOGS", tmp_path)
    log_dir = tmp_path / "run1" / "black" / "1"
    log_dir.mkdir(parents=True)
    (log_dir / "driver.out").write_text(
        "blah\nError response from daemon: container ... is not running\n", encoding="utf-8"
    )
    assert _driver_out_signals_lane_death("run1", "black", 1) is True


def test_driver_out_no_death_for_ordinary_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "EXTRACTION_LOGS", tmp_path)
    log_dir = tmp_path / "run1" / "scrapy" / "1"
    log_dir.mkdir(parents=True)
    (log_dir / "driver.out").write_text(
        "ValueError: no option named '--reactor'\n", encoding="utf-8"
    )
    assert _driver_out_signals_lane_death("run1", "scrapy", 1) is False


def test_write_manifest(tmp_path: Path) -> None:
    results = [
        BugOutcome("black", 1, 0, 1, 0, "ok", 12.0),
        BugOutcome("black", 2, 1, 1, 1, "failed", 30.0),
        BugOutcome("luigi", 1, 2, 2, None, "timeout", 2400.0),
    ]
    summary = _write_manifest(tmp_path, results, "run1", 5, True, False, 2400)
    assert summary == {"total": 3, "ok": 1, "failed": 1, "timeout": 1}
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert '"run_id": "run1"' in manifest
    assert '"lanes": 5' in manifest
