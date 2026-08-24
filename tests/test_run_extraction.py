"""Tests for run_extraction orchestration behavior."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import run_extraction


def _mock_repo(tmp_path: Path) -> MagicMock:
    repo = MagicMock()
    repo.repo_dir = tmp_path / "repos" / "Chart" / "1"
    repo.output_dir = tmp_path / "processed" / "Chart" / "1"
    repo.repo_dir.mkdir(parents=True)
    repo.output_dir.mkdir(parents=True)
    return repo


def _tracker_path_under(tmp_path: Path) -> Path:
    """Where the patched tracker_path() should write — inside tmp_path."""
    # Reason: TrackerStep, load_tracker, and save_tracker all route through
    # src.common.tracker.tracker_path → get_processed_dir, which resolves the
    # real data/D4J/... path regardless of repo.output_dir. Redirect it here
    # so failing-step exceptions and compute_coverage results don't leak onto
    # the real on-disk tracker.
    out = tmp_path / "processed" / "Chart" / "1"
    out.mkdir(parents=True, exist_ok=True)
    return out / "tracker.json"


_PIPELINE_PATCHES = (
    "src.extraction.pipeline.run_corpus_method_extraction",
    "src.extraction.pipeline.write_parsed_failing_tests",
    "src.extraction.pipeline.save_trigger_test_clean",
    "src.extraction.pipeline.save_fault_lines",
    "src.extraction.pipeline.save_first_fault_lines",
    "src.extraction.pipeline.save_bug_report",
)


def _patch_pipeline(stack: ExitStack) -> None:
    for target in _PIPELINE_PATCHES:
        stack.enter_context(patch(target))


def test_process_bug_cleans_checkout_on_failure(tmp_path: Path) -> None:
    repo = _mock_repo(tmp_path)

    with ExitStack() as stack:
        stack.enter_context(patch("run_extraction.D4JRepo", return_value=repo))
        stack.enter_context(
            patch(
                "src.extraction.pipeline.run_gzoltar_pipeline",
                side_effect=RuntimeError("boom"),
            )
        )
        _patch_pipeline(stack)
        stack.enter_context(
            patch("src.extraction.pipeline.validate_extraction_outputs", return_value=[])
        )
        stack.enter_context(
            patch(
                "src.common.tracker.tracker_path",
                return_value=_tracker_path_under(tmp_path),
            )
        )
        with pytest.raises(RuntimeError, match="boom"):
            run_extraction.process_bug("Chart", 1, cleanup_checkouts=True)

    repo.remove_repo.assert_called_once()


def test_process_bug_can_keep_checkout_for_debug(tmp_path: Path) -> None:
    repo = _mock_repo(tmp_path)

    with ExitStack() as stack:
        stack.enter_context(patch("run_extraction.D4JRepo", return_value=repo))
        stack.enter_context(
            patch(
                "src.extraction.pipeline.run_gzoltar_pipeline",
                side_effect=RuntimeError("boom"),
            )
        )
        _patch_pipeline(stack)
        stack.enter_context(
            patch("src.extraction.pipeline.validate_extraction_outputs", return_value=[])
        )
        stack.enter_context(
            patch(
                "src.common.tracker.tracker_path",
                return_value=_tracker_path_under(tmp_path),
            )
        )
        with pytest.raises(RuntimeError, match="boom"):
            run_extraction.process_bug("Chart", 1, cleanup_checkouts=False)

    repo.remove_repo.assert_not_called()


def test_process_bug_runs_validation_with_expected_flags(tmp_path: Path) -> None:
    repo = _mock_repo(tmp_path)

    validate_mock = MagicMock(return_value=[])
    with ExitStack() as stack:
        stack.enter_context(patch("run_extraction.D4JRepo", return_value=repo))
        stack.enter_context(patch("src.extraction.pipeline.run_gzoltar_pipeline"))
        _patch_pipeline(stack)
        stack.enter_context(
            patch("src.extraction.pipeline.validate_extraction_outputs", validate_mock)
        )
        stack.enter_context(
            patch(
                "src.common.tracker.tracker_path",
                return_value=_tracker_path_under(tmp_path),
            )
        )
        result = run_extraction.process_bug(
            "Chart",
            1,
            cleanup_checkouts=True,
            validate_outputs=True,
        )

    validate_mock.assert_called_once_with(
        repo.output_dir,
        expect_gzoltar=True,
        expect_faults=True,
        expect_bug_report=True,
        dataset="defects4j",
    )
    assert result["cleanup_status"] == "removed"
