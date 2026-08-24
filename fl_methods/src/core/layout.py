"""Benchmark-aware repository/data layout helpers.

This module introduces a benchmark-agnostic path layer while keeping the
existing Defects4J on-disk layout unchanged (``data/D4J/...``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

BENCHMARK_ALIASES: dict[str, str] = {
    "d4j": "D4J",
    "defects4j": "D4J",
    "bugsinpy": "BIP",
    "bip": "BIP",
}


def normalize_benchmark_name(name: str) -> str:
    """Normalize benchmark name to a canonical data folder name."""
    key = name.strip().lower()
    if key in BENCHMARK_ALIASES:
        return BENCHMARK_ALIASES[key]
    return name.strip()


@dataclass(frozen=True)
class DatasetLayout:
    """Filesystem layout for benchmark data roots under ``<repo>/data``.

    ``benchmark_overrides`` lets callers point a specific benchmark's data root
    somewhere outside the repo (e.g. ``CEFL_D4J_WORKSPACE`` on the cluster
    pointing at ``$SLURM_SCRATCH_DIR/data/D4J``). Keys are canonical names
    (``"D4J"``); values are absolute host paths.
    """

    project_root: Path
    benchmark_overrides: Mapping[str, Path] = field(default_factory=dict)

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    def benchmark_root(self, benchmark: str) -> Path:
        canonical = normalize_benchmark_name(benchmark)
        if canonical in self.benchmark_overrides:
            return self.benchmark_overrides[canonical]
        return self.data_root / canonical

    def processed_root(self, benchmark: str) -> Path:
        return self.benchmark_root(benchmark) / "processed"

    def repos_root(self, benchmark: str) -> Path:
        return self.benchmark_root(benchmark) / "repos"

    def fixed_root(self, benchmark: str) -> Path:
        return self.benchmark_root(benchmark) / "fixed"

    def exports_root(self, benchmark: str) -> Path:
        return self.benchmark_root(benchmark) / "exports"
