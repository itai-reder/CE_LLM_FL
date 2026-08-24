"""Benchmark adapter registry."""

from __future__ import annotations

from typing import Any, Protocol

from src.benchmarks.bugsinpy.adapter import BugsInPyAdapter
from src.benchmarks.defects4j.adapter import Defects4JAdapter


class BenchmarkAdapter(Protocol):
    """Structural contract every benchmark adapter satisfies.

    ``build_repo`` is typed ``-> Any`` because each benchmark returns its own repo
    class; ``run_extraction.py`` already treats the repo factory as ``Any``. Extracting
    a full Repo protocol is deferred.
    """

    # Declared read-only so frozen-dataclass adapters (whose fields are immutable)
    # structurally satisfy the protocol — a mutable attribute requirement would not.
    @property
    def benchmark_key(self) -> str: ...
    @property
    def benchmark_folder(self) -> str: ...

    def list_projects(self) -> list[str]: ...
    def list_cases(self, project: str) -> list[int]: ...
    def build_repo(self, project: str, case_id: int, *, buggy: bool = ...) -> Any: ...


_REGISTRY: dict[str, BenchmarkAdapter] = {
    "defects4j": Defects4JAdapter(),
    "d4j": Defects4JAdapter(),
    "bugsinpy": BugsInPyAdapter(),
    "bip": BugsInPyAdapter(),
}


def supported_benchmarks() -> tuple[str, ...]:
    """Return supported benchmark keys accepted by the CLI."""
    return tuple(sorted(_REGISTRY.keys()))


def get_benchmark_adapter(name: str) -> BenchmarkAdapter:
    """Resolve a benchmark adapter by user-provided key."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unsupported benchmark: {name}")
    return _REGISTRY[key]
