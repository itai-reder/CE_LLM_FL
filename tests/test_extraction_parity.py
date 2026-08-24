"""Parity tests for the D4J extraction pipeline.

Both ``run_extraction.process_bug_d4j`` and ``run_sr._run_extraction_d4j``
route through :func:`src.extraction.pipeline.ensure_d4j_outputs`.  These
tests confirm the run-all step sequence is identical between the two
entry points by monkey-patching each underlying step to append its name
to a recorder list.

We don't need a real Defects4J container: the helper is pure orchestration
and every step takes a ``D4JRepo`` we can stub with a minimal fake.
"""

from __future__ import annotations

from typing import cast

import pytest

from src.extraction.d4j import D4JRepo


@pytest.fixture
def step_recorder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every step function with a recorder; return the call log."""
    recorded: list[str] = []

    def make_recorder(name: str):
        def _record(*args: object, **kwargs: object) -> None:
            recorded.append(name)

        return _record

    # repo_setup steps live on the D4JRepo object (handled in the fake).
    # The remaining steps are module-level functions invoked by the helper.
    targets = [
        ("src.extraction.pipeline.run_corpus_method_extraction", "signatures.corpus"),
        ("src.extraction.pipeline.write_parsed_failing_tests", "tests.failing_tests"),
        ("src.extraction.pipeline.save_trigger_test_clean", "tests.trigger_test_clean"),
        ("src.extraction.pipeline.run_gzoltar_pipeline", "gzoltar"),
        ("src.extraction.pipeline.save_fault_lines", "faults.fault_lines"),
        ("src.extraction.pipeline.save_first_fault_lines", "faults.first_fault_lines"),
        ("src.extraction.pipeline.save_bug_report", "bug_report"),
        ("src.extraction.pipeline.validate_extraction_outputs", "validate"),
    ]
    for target, name in targets[:-1]:
        monkeypatch.setattr(target, make_recorder(name))

    # validation needs to return an empty list to satisfy callers.
    def _validate(*args: object, **kwargs: object) -> list[dict[str, str]]:
        recorded.append("validate")
        return []

    monkeypatch.setattr("src.extraction.pipeline.validate_extraction_outputs", _validate)
    return recorded


class _FakeD4JRepo:
    """Minimal stand-in for D4JRepo that records lifecycle calls."""

    def __init__(self, recorder: list[str], output_dir: object) -> None:
        self._recorder = recorder
        self.output_dir = output_dir

    def checkout(self, *args: object, **kwargs: object) -> None:
        self._recorder.append("repo.checkout")

    def compile(self, *args: object, **kwargs: object) -> None:
        self._recorder.append("repo.compile")

    def export_all_properties(self, *args: object, **kwargs: object) -> None:
        self._recorder.append("repo.export_properties")

    def get_relevant_test_methods(self, *args: object, **kwargs: object) -> list[str]:
        self._recorder.append("repo.relevant_tests")
        return []


_CANONICAL_RUN_ALL_ORDER = [
    "repo.checkout",
    "repo.compile",
    "repo.export_properties",
    "signatures.corpus",
    "repo.relevant_tests",
    "tests.failing_tests",
    "tests.trigger_test_clean",
    "gzoltar",
    "faults.fault_lines",
    "faults.first_fault_lines",
    "bug_report",
    "validate",
]


def test_run_all_invokes_canonical_sequence(step_recorder: list[str], tmp_path: object) -> None:
    """``ensure_d4j_outputs`` (run-all, fresh checkout) hits every step in order."""
    from src.extraction.pipeline import ensure_d4j_outputs

    repo = cast(D4JRepo, _FakeD4JRepo(step_recorder, output_dir=tmp_path))
    issues = ensure_d4j_outputs(repo, project="Lang", bug_id=1, skip_existing=False)

    assert issues == []
    assert step_recorder == _CANONICAL_RUN_ALL_ORDER


def test_run_extraction_and_run_sr_use_same_pipeline_helper() -> None:
    """Both entry points import and call ``ensure_d4j_outputs``."""
    import run_extraction
    import run_sr

    assert run_extraction.ensure_d4j_outputs is run_sr.ensure_d4j_outputs


def test_checked_out_caller_skips_repo_checkout(step_recorder: list[str], tmp_path: object) -> None:
    """``run_sr`` invokes the helper with ``checked_out=True``; the helper
    must NOT call ``repo.checkout`` again."""
    from src.extraction.pipeline import ensure_d4j_outputs

    repo = cast(D4JRepo, _FakeD4JRepo(step_recorder, output_dir=tmp_path))
    ensure_d4j_outputs(repo, project="Lang", bug_id=1, skip_existing=False, checked_out=True)

    assert "repo.checkout" not in step_recorder
    # Everything else still happens.
    assert step_recorder[0] == "repo.compile"
    assert step_recorder[-1] == "validate"


def test_partial_subset_skips_unselected_steps(step_recorder: list[str], tmp_path: object) -> None:
    """``--gzoltar-only`` maps to steps=('repo_setup', 'signatures', 'tests', 'gzoltar');
    faults + bug_report must NOT run."""
    from src.extraction.pipeline import ensure_d4j_outputs

    repo = cast(D4JRepo, _FakeD4JRepo(step_recorder, output_dir=tmp_path))
    ensure_d4j_outputs(
        repo,
        project="Lang",
        bug_id=1,
        skip_existing=False,
        steps=("repo_setup", "signatures", "tests", "gzoltar"),
    )

    assert "faults.fault_lines" not in step_recorder
    assert "faults.first_fault_lines" not in step_recorder
    assert "bug_report" not in step_recorder
    assert "gzoltar" in step_recorder
