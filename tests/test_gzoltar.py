"""Tests for GZoltar extraction helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.extraction.gzoltar import run_gzoltar_coverage


def test_run_gzoltar_coverage_retries_after_access_fix_failure(tmp_path: Path, monkeypatch) -> None:
    repo = MagicMock()
    repo.project = "JacksonDatabind"
    repo.bug_id = 75
    repo.output_dir = tmp_path / "processed" / "JacksonDatabind" / "75"
    repo.repo_dir = tmp_path / "repos" / "JacksonDatabind" / "75"
    repo.get_bin_class_dir.return_value = tmp_path / "repo_build"
    repo.get_cp_test.return_value = "cp_test"
    repo.get_relevant_classes.return_value = ["com.fasterxml.jackson.databind.ObjectMapper"]

    repo.output_dir.mkdir(parents=True)
    repo.repo_dir.mkdir(parents=True)
    junit_tests = repo.output_dir / "junit_tests.txt"
    junit_tests.write_text(
        "JUNIT,com.fasterxml.jackson.databind.misc.AccessFixTest#testAccess\n"
        "JUNIT,com.fasterxml.jackson.databind.ObjectMapperTest#testMapper\n"
    )

    monkeypatch.setattr(
        "src.extraction.gzoltar.ensure_java_utils_in_workspace",
        lambda: ("agent.jar", "cli.jar"),
    )
    monkeypatch.setattr("src.extraction.gzoltar.to_container_path", lambda p: str(p))

    run_calls: list[list[str]] = []
    coverage_ser = repo.output_dir / "gzoltar" / "coverage" / "coverage.ser"

    def _run_java(command: list[str], cwd: Path) -> None:
        run_calls.append(command)
        if len(run_calls) == 1:
            raise RuntimeError(
                "ClassCircularityError: java/security/Permission\n"
                "AccessFixTest$CauseBlockingSecurityManager"
            )

        assert cwd == repo.repo_dir
        coverage_ser.parent.mkdir(parents=True, exist_ok=True)
        coverage_ser.write_text("ok")

    monkeypatch.setattr("src.extraction.gzoltar.run_java_in_container", _run_java)

    run_gzoltar_coverage(repo)

    assert len(run_calls) == 2
    assert "junit_tests.filtered.txt" in run_calls[1][run_calls[1].index("--testMethods") + 1]
    filtered = repo.output_dir / "junit_tests.filtered.txt"
    assert filtered.exists()
    assert "AccessFixTest" not in filtered.read_text()
