"""Tests for src.extraction.docker — path translation helpers.

Only tests pure functions that do not require a running Docker container.
"""

from __future__ import annotations

from src.common.config import CONTAINER_WORKSPACE, get_benchmark_data_dir
from src.extraction.docker import to_container_path, translate_cmd_paths

_DATA_DIR = get_benchmark_data_dir()


class TestToContainerPath:
    """Tests for to_container_path()."""

    def test_data_dir_path(self) -> None:
        host = _DATA_DIR / "repos" / "Chart" / "1"
        result = to_container_path(host)
        assert result == f"{CONTAINER_WORKSPACE}/repos/Chart/1"

    def test_processed_dir_path(self) -> None:
        host = _DATA_DIR / "processed" / "Math" / "42"
        result = to_container_path(host)
        assert result == f"{CONTAINER_WORKSPACE}/processed/Math/42"

    def test_non_data_path_unchanged(self) -> None:
        host = "/usr/local/bin/java"
        result = to_container_path(host)
        assert result == host

    def test_string_input(self) -> None:
        host = str(_DATA_DIR / "repos" / "Lang" / "5")
        result = to_container_path(host)
        assert result == f"{CONTAINER_WORKSPACE}/repos/Lang/5"

    def test_data_dir_root(self) -> None:
        result = to_container_path(_DATA_DIR)
        assert result == CONTAINER_WORKSPACE


class TestTranslateCmdPaths:
    """Tests for translate_cmd_paths()."""

    def test_translates_data_paths(self) -> None:
        cmd = [
            "checkout",
            "-p",
            "Chart",
            "-v",
            "1b",
            "-w",
            str(_DATA_DIR / "repos" / "Chart" / "1"),
        ]
        result = translate_cmd_paths(cmd)
        assert result[0] == "checkout"
        assert result[-1] == f"{CONTAINER_WORKSPACE}/repos/Chart/1"

    def test_preserves_non_paths(self) -> None:
        cmd = ["-p", "Chart", "-v", "1b", "--flag"]
        result = translate_cmd_paths(cmd)
        assert result == cmd

    def test_preserves_non_data_abs_paths(self) -> None:
        cmd = ["/usr/bin/java", "-cp", "/some/classpath"]
        result = translate_cmd_paths(cmd)
        assert result == cmd

    def test_empty_list(self) -> None:
        assert translate_cmd_paths([]) == []

    def test_path_objects_converted(self) -> None:
        cmd = ["export", "-o", str(_DATA_DIR / "processed" / "Cli" / "3" / "cp.test")]
        result = translate_cmd_paths(cmd)
        assert CONTAINER_WORKSPACE in result[-1]
