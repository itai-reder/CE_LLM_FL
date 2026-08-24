"""Defects4J adapter over existing extraction/repo APIs."""

from __future__ import annotations

from dataclasses import dataclass

from src.extraction.d4j import D4JRepo, get_d4j_bids, get_d4j_pids


@dataclass(frozen=True)
class Defects4JAdapter:
    """Thin adapter that exposes benchmark-generic methods."""

    benchmark_key: str = "defects4j"
    benchmark_folder: str = "D4J"

    def list_projects(self) -> list[str]:
        return get_d4j_pids()

    def list_cases(self, project: str) -> list[int]:
        return get_d4j_bids(project)

    def build_repo(self, project: str, case_id: int, *, buggy: bool = True) -> D4JRepo:
        return D4JRepo(project, case_id, buggy=buggy)
