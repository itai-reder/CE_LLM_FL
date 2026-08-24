"""Cross-checks for the per-project FauxPy config against the authors' reference.

``FAUXPY_PROJECT_CONFIG`` (``src.extraction.bip_fauxpy_config``) encodes the FauxPy authors'
constant per-project ``--src`` / ``--exclude`` (from ``subject_info.csv``) for the 13 reference
projects plus layout-derived values for the 4 BugsInPy projects absent from the reference
(ansible, matplotlib, scrapy, PySnooper). These tests guard against drift from the reference and
keep the config internally consistent (every BIP project covered, paths source-tree-relative, the
thefuck fixture set synced to the shipped fixtures).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from src.common.config import BIP_INDEX_CSV
from src.extraction.bip_fauxpy_config import (
    FAUXPY_FIXTURES_ROOT,
    FAUXPY_PROJECT_CONFIG,
    SUBJECT_INFO_CSV,
    _fixture_bug_ids,
)

# The 13 projects with authoritative reference rows (BENCHMARK_NAME == BIP project name).
_REFERENCE_PROJECTS = {
    "black",
    "cookiecutter",
    "fastapi",
    "httpie",
    "keras",
    "luigi",
    "pandas",
    "sanic",
    "spacy",
    "thefuck",
    "tornado",
    "tqdm",
    "youtube-dl",
}


def _parse_exclude(raw: str) -> set[str]:
    """Parse a ``subject_info.csv`` EXCLUDE cell (``;``-separated, ``-`` = empty, may wrap lines)."""
    raw = (raw or "").strip()
    if not raw or raw == "-":
        return set()
    return {part.strip() for part in raw.split(";") if part.strip()}


def _reference_rows() -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SUBJECT_INFO_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):  # csv handles the EXCLUDE cells' embedded newlines
            rows[row["BENCHMARK_NAME"]].append(row)
    return rows


@pytest.mark.parametrize("project", sorted(_REFERENCE_PROJECTS))
def test_src_and_exclude_match_reference(project: str) -> None:
    """Each reference project's config ``src``/``exclude`` equals subject_info.csv, and is constant."""
    rows = _reference_rows()[project]
    assert rows, f"no subject_info.csv rows for {project}"

    target_dirs = {r["TARGET_DIR"] for r in rows}
    excludes = {frozenset(_parse_exclude(r["EXCLUDE"])) for r in rows}
    assert len(target_dirs) == 1, f"{project}: TARGET_DIR varies across bugs: {target_dirs}"
    assert len(excludes) == 1, f"{project}: EXCLUDE varies across bugs: {excludes}"

    cfg = FAUXPY_PROJECT_CONFIG[project]
    assert cfg.src == target_dirs.pop(), f"{project}: src diverges from reference TARGET_DIR"
    assert set(cfg.exclude) == set(excludes.pop()), f"{project}: exclude diverges from reference"


def test_all_bip_projects_have_a_config() -> None:
    """Every BugsInPy project in the index has a FauxPy config entry, and vice versa."""
    with BIP_INDEX_CSV.open(encoding="utf-8") as fh:
        bip_projects = {row["repo"] for row in csv.DictReader(fh)}
    assert bip_projects, "BIP index produced no projects"
    missing = bip_projects - set(FAUXPY_PROJECT_CONFIG)
    extra = set(FAUXPY_PROJECT_CONFIG) - bip_projects
    assert not missing, f"BIP projects without a FauxPy config: {sorted(missing)}"
    assert not extra, f"FauxPy config keys that are not BIP projects: {sorted(extra)}"


@pytest.mark.parametrize("project", sorted(FAUXPY_PROJECT_CONFIG))
def test_paths_are_source_tree_relative(project: str) -> None:
    """``--src``/``--exclude``/``extra_pythonpath`` must be relative (resolved against pytest cwd)."""
    cfg = FAUXPY_PROJECT_CONFIG[project]
    for path in (cfg.src, *cfg.exclude, *cfg.extra_pythonpath):
        assert not Path(path).is_absolute(), f"{project}: absolute path not allowed: {path!r}"


def test_thefuck_fixture_set_tracks_disk() -> None:
    """The thefuck conftest swap set is exactly the on-disk ``B*`` fixture dirs (incl. B9)."""
    cfg = FAUXPY_PROJECT_CONFIG["thefuck"]
    (fixture,) = cfg.host.copy_fixtures
    assert fixture.bugs == _fixture_bug_ids("thefuck")
    assert fixture.bugs, "expected thefuck conftest fixtures on disk"
    assert 9 in fixture.bugs, "B9 has a conftest fixture and must be included"


def test_every_referenced_fixture_file_exists() -> None:
    """Every (project, bug) fixture the config points at resolves to a real file in the clone."""
    assert FAUXPY_FIXTURES_ROOT.is_dir(), f"fixtures root missing: {FAUXPY_FIXTURES_ROOT}"
    for project, cfg in FAUXPY_PROJECT_CONFIG.items():
        for fixture in cfg.host.copy_fixtures:
            bugs = fixture.bugs if fixture.bugs is not None else set()
            for bug_id in bugs:
                src = fixture.source_for(bug_id)
                assert src.is_file(), f"{project} B{bug_id}: missing fixture {src}"
