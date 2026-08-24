"""Tests for run_sr orchestration behavior (formerly run_all)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import run_sr


def test_run_single_bug_d4j_infers_tracker_after_repo_setup(tmp_path) -> None:
    events: list[str] = []
    repo = MagicMock()

    def make_repo(project: str, bug_id: int) -> MagicMock:
        events.append("repo")
        return repo

    def infer_tracker(*args, **kwargs):
        events.append("update")
        raise RuntimeError("stop after inference")

    with (
        patch("run_sr.get_processed_dir", return_value=tmp_path / "processed" / "Chart" / "1"),
        patch("run_sr.D4JRepo", side_effect=make_repo),
        patch("run_sr.update_single_bug", side_effect=infer_tracker),
    ):
        summary = run_sr.run_single_bug_d4j(
            "Chart",
            "1",
            sr_cfg=MagicMock(),
            use_ce=False,
            ce_max_iter=1000,
            ce_pop_size=100,
            skip_existing=True,
            keep_checkouts=True,
        )

    assert events == ["repo", "update"]
    assert summary["error"] == "stop after inference"


def test_run_single_bug_d4j_requires_failing_tests_for_extraction_skip(tmp_path) -> None:
    repo = MagicMock()
    inferred_tracker = {
        "extraction": {
            "completed": [
                "properties",
                "signatures",
                "relevant_tests",
                "gzoltar",
                "faults",
                "bug_report",
            ],
        },
        "fl": {"completed": []},
        "sr": {},
    }

    with (
        patch("run_sr.get_processed_dir", return_value=tmp_path / "processed" / "Chart" / "1"),
        patch("run_sr.D4JRepo", return_value=repo),
        patch("run_sr.update_single_bug", return_value=inferred_tracker),
        patch("run_sr._run_extraction_d4j", side_effect=RuntimeError("missing failing_tests")),
    ):
        summary = run_sr.run_single_bug_d4j(
            "Chart",
            "1",
            sr_cfg=MagicMock(),
            use_ce=False,
            ce_max_iter=1000,
            ce_pop_size=100,
            skip_existing=True,
            keep_checkouts=True,
        )

    repo.checkout.assert_called_once_with(skip_existing=True)
    assert summary["stages"]["extraction"]["status"] == "FAILED"
    assert summary["error"] == "missing failing_tests"
