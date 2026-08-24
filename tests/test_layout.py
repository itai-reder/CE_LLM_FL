"""Tests for benchmark-aware layout and adapter registry."""

from __future__ import annotations

from pathlib import Path

from src.benchmarks.registry import get_benchmark_adapter, supported_benchmarks
from src.core.layout import DatasetLayout, normalize_benchmark_name


def test_normalize_defects4j_aliases() -> None:
    assert normalize_benchmark_name("defects4j") == "D4J"
    assert normalize_benchmark_name("D4J") == "D4J"


def test_dataset_layout_paths() -> None:
    layout = DatasetLayout(Path("/tmp/repo"))
    assert layout.benchmark_root("defects4j") == Path("/tmp/repo/data/D4J")
    assert layout.processed_root("defects4j") == Path("/tmp/repo/data/D4J/processed")
    assert layout.repos_root("defects4j") == Path("/tmp/repo/data/D4J/repos")


def test_registry_resolves_defects4j() -> None:
    assert "defects4j" in supported_benchmarks()
    adapter = get_benchmark_adapter("defects4j")
    assert adapter.benchmark_key == "defects4j"
    assert adapter.benchmark_folder == "D4J"
