"""BugsInPy benchmark adapter.

Mirrors :class:`src.benchmarks.defects4j.adapter.Defects4JAdapter`, exposing the
benchmark-generic surface (``list_projects`` / ``list_cases`` / ``build_repo``) over the
BugsInPy extraction APIs in :mod:`src.extraction.bugsinpy`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.extraction.bugsinpy import BugsInPyRepo, get_bip_bids, get_bip_pids


@dataclass(frozen=True)
class BugsInPyAdapter:
    """Thin adapter that exposes benchmark-generic methods for BugsInPy."""

    benchmark_key: str = "bugsinpy"
    benchmark_folder: str = "BIP"

    def list_projects(self) -> list[str]:
        return get_bip_pids()

    def list_cases(self, project: str) -> list[int]:
        return get_bip_bids(project)

    def build_repo(self, project: str, case_id: int, *, buggy: bool = True) -> BugsInPyRepo:
        return BugsInPyRepo(project, case_id, buggy=buggy)
