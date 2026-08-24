"""Tests for the BugsInPy SR sweep driver helpers (run_bip_sr_sweep).

These cover the pure/host-side helpers — bug enumeration with resume filtering, SR-result
detection, per-bug log-dir isolation, the per-stage command builders, and the ETA/summary
formatting — without spawning containers or subprocesses.
"""

from __future__ import annotations

from pathlib import Path

import run_bip_sr_sweep as sweep
from run_bip_sr_sweep import (
    BugResult,
    StageResult,
    SweepConfig,
    _corpus_cmd,
    _extraction_cmd,
    _fmt_eta,
    _per_bug_log_dir,
    _resolve_pairs,
    _sr_cmd,
    _sr_result_exists,
    _summary_line,
)

_CFG = SweepConfig(
    model="llama3.1:8b",
    base_url="https://cis-ollama.auth.ad.bgu.ac.il",
    iterations=10,
    no_verify=True,
    force=False,
    extract_timeout_sec=2400,
    corpus_timeout_sec=1200,
    sr_timeout_sec=1800,
    max_attempts=2,
)


def test_per_bug_log_dir_is_isolated() -> None:
    a = _per_bug_log_dir("run1", "black", 1)
    b = _per_bug_log_dir("run1", "black", 2)
    assert a != b
    assert a.name == "1" and a.parent.name == "black"


def test_resolve_pairs_enumerates_all_bugs(monkeypatch) -> None:
    monkeypatch.setattr(sweep, "iter_all_bugs", lambda projects: [("black", 1), ("black", 2)])
    pairs = _resolve_pairs(None, resume=False, limit=None)
    assert pairs == [("black", 1), ("black", 2)]


def test_resolve_pairs_limit(monkeypatch) -> None:
    monkeypatch.setattr(sweep, "iter_all_bugs", lambda projects: [("a", 1), ("a", 2), ("a", 3)])
    assert _resolve_pairs(None, resume=False, limit=2) == [("a", 1), ("a", 2)]


def test_resolve_pairs_resume_drops_sr_complete(monkeypatch) -> None:
    monkeypatch.setattr(sweep, "iter_all_bugs", lambda projects: [("a", 1), ("a", 2)])
    # Bug 1 is SR-complete AND extraction-complete -> dropped; bug 2 kept.
    monkeypatch.setattr(sweep, "_sr_result_exists", lambda p, b: b == 1)
    monkeypatch.setattr(sweep, "is_bug_complete", lambda repo: True)
    assert _resolve_pairs(None, resume=True, limit=None) == [("a", 2)]


def test_resolve_pairs_force_bypasses_resume_filter(monkeypatch) -> None:
    # --force must re-run every bug, even ones already SR-complete on disk.
    monkeypatch.setattr(sweep, "iter_all_bugs", lambda projects: [("a", 1), ("a", 2)])
    monkeypatch.setattr(sweep, "_sr_result_exists", lambda p, b: True)
    monkeypatch.setattr(sweep, "is_bug_complete", lambda repo: True)
    assert _resolve_pairs(None, resume=True, force=True, limit=None) == [("a", 1), ("a", 2)]


def test_resolve_pairs_resume_keeps_sr_complete_but_extraction_incomplete(monkeypatch) -> None:
    # A stray sr_result without a complete extraction should NOT be treated as done.
    monkeypatch.setattr(sweep, "iter_all_bugs", lambda projects: [("a", 1)])
    monkeypatch.setattr(sweep, "_sr_result_exists", lambda p, b: True)
    monkeypatch.setattr(sweep, "is_bug_complete", lambda repo: False)
    assert _resolve_pairs(None, resume=True, limit=None) == [("a", 1)]


def test_sr_result_exists(tmp_path: Path, monkeypatch) -> None:
    proc = tmp_path / "PySnooper" / "3"
    monkeypatch.setattr(sweep, "get_processed_dir", lambda project, bug_id, dataset: proc)
    assert _sr_result_exists("PySnooper", 3) is False
    model_dir = proc / "FlexFL" / "SR" / "Agent4SR" / "llama3.1_8b"
    model_dir.mkdir(parents=True)
    (model_dir / "sr_result.json").write_text("{}", encoding="utf-8")
    assert _sr_result_exists("PySnooper", 3) is True


def test_sr_cmd_passes_endpoint_and_no_verify() -> None:
    cmd = _sr_cmd("black", 1, _CFG)
    assert "run" in cmd
    assert cmd[cmd.index("--base-url") + 1] == _CFG.base_url
    assert cmd[cmd.index("--model") + 1] == "llama3.1:8b"
    assert "--no-verify" in cmd
    assert "--force" not in cmd  # cfg.force is False


def test_extraction_and_corpus_cmds_target_bugsinpy() -> None:
    ext = _extraction_cmd("black", 1, Path("/tmp/logs"), force=False)
    assert ext[ext.index("--benchmark") + 1] == "bugsinpy"
    assert "--force" not in ext
    cor = _corpus_cmd("black", 1, force=True)
    assert cor[2] == "corpus"
    assert cor[cor.index("--benchmark") + 1] == "bugsinpy"
    assert "--force" in cor


def test_fmt_eta() -> None:
    assert _fmt_eta(45) == "45s"
    assert _fmt_eta(90) == "1m30s"
    assert _fmt_eta(3700) == "1h01m"


def _bug(outcome: str, reason: str = "", **stages) -> BugResult:
    na = StageResult("n/a")
    return BugResult(
        project="black",
        bug_id=1,
        lane=0,
        outcome=outcome,
        reason=reason,
        extraction=stages.get("extraction", na),
        corpus=stages.get("corpus", na),
        sr=stages.get("sr", na),
        duration_sec=12.0,
    )


def test_summary_line_ok_shows_stage_timings() -> None:
    line = _summary_line(
        _bug(
            "ok",
            "sr produced",
            extraction=StageResult("ran", 100.0),
            corpus=StageResult("ran", 20.0),
            sr=StageResult("ran", 30.0),
        )
    )
    assert line.startswith("✓ black/1")
    assert "ext 100s" in line and "cor 20s" in line and "sr 30s" in line


def test_summary_line_failure_shows_reason() -> None:
    line = _summary_line(_bug("failed", "extraction failed (rc=1)"))
    assert line.startswith("✗ black/1")
    assert "extraction failed (rc=1)" in line
