#!/usr/bin/env python3
"""Package per-project processed data into Google Drive distribution zips.

Produces ``<out_dir>/<Benchmark>/<Project>.zip`` whose internal layout is
``<Project>/<BugId>/<full processed tree>`` — matching the layout documented
in the README "Data" section. Benchmark-level dirs (``_evaluation``,
``_logs``) and heavyweight reproducible intermediates are skipped.

Run from the repo root (PYTHONPATH not required)::

    python utils/package_drive.py -p Lang                       # one project
    python utils/package_drive.py --benchmark bip --all-projects
    python utils/package_drive.py -p Lang --out-dir /tmp/drive
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_FOLDERS = {"d4j": "D4J", "defects4j": "D4J", "bip": "BIP", "bugsinpy": "BIP"}

# Reproducible or oversized intermediates excluded from the archives
# (mirrors the repo .gitignore for processed trees).
EXCLUDED_NAMES = {"coverage.ser", "matrix.txt"}
EXCLUDED_DIRS = {
    "raw",  # FauxPy/raw — untouched FauxPy reports, huge
    "Evaluation",  # legacy per-bug metrics; superseded by results/<BM>/<P>/<B>/evaluation/
}


def _iter_files(project_dir: Path):
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir)
        if path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        yield path, rel


def package_project(benchmark_folder: str, project: str, out_dir: Path) -> Path:
    processed_root = REPO_ROOT / "data" / benchmark_folder / "processed"
    project_dir = processed_root / project
    if not project_dir.is_dir():
        raise SystemExit(f"no processed data for {project!r} under {processed_root}")

    bug_dirs = [d for d in project_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not bug_dirs:
        raise SystemExit(f"{project_dir} contains no numeric bug dirs; not a project tree")
    if not any(
        (d / "method_signatures.csv").exists() or (d / "faults.csv").exists() for d in bug_dirs
    ):
        raise SystemExit(
            f"{project_dir} has no extracted bug data; refusing to package a stray dir"
        )

    target_dir = out_dir / benchmark_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{project}.zip"

    n = 0
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path, rel in _iter_files(project_dir):
            zf.write(path, arcname=str(Path(project) / rel))
            n += 1
    print(f"{target}  ({n} files, {target.stat().st_size / 1e6:.1f} MB)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Package processed data for Drive upload")
    parser.add_argument(
        "--benchmark",
        default="d4j",
        choices=sorted(BENCHMARK_FOLDERS),
        help="Benchmark whose processed tree to package.",
    )
    parser.add_argument("-p", "--project", default=None, help="Project to package.")
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Package every project under the benchmark's processed tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "drive_packages",
        help="Where to write the zips (default: <repo>/drive_packages/).",
    )
    args = parser.parse_args()

    if bool(args.project) == bool(args.all_projects):
        parser.error("exactly one of -p/--project or --all-projects is required")

    folder = BENCHMARK_FOLDERS[args.benchmark]
    if args.project:
        projects = [args.project]
    else:
        root = REPO_ROOT / "data" / folder / "processed"
        if not root.is_dir():
            raise SystemExit(f"no processed tree at {root}")
        projects = sorted(
            d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
        )

    failures = []
    for project in projects:
        try:
            package_project(folder, project, args.out_dir)
        except SystemExit as exc:
            print(f"SKIP {project}: {exc}", file=sys.stderr)
            failures.append(project)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
