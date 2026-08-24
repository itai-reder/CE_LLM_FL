"""CLI: drive the full BugsInPy extraction across N isolated container lanes.

The extraction pipeline itself is single-bug and not concurrency-safe under a shared container:
per-bug conda envs are keyed by ``md5(python_version+requirements)`` and the FauxPy step mutates
them (installs FauxPy, pins setuptools, uninstalls pytest-sugar). Two bugs touching the same env
concurrently corrupt each other. Conda envs live *inside* each container, so the safe unit of
parallelism is **one container per lane**: this driver pre-creates ``bugsinpy-cefl-lane-{0..N-1}``
(all sharing the ``bugsinpy:cefl`` image and the ``data/BIP`` mount) and runs each bug as a
subprocess of ``run_extraction.py`` with ``CEFL_BIP_CONTAINER`` pinned to the worker's lane. Bugs
are serial within a lane (mandatory — env hashes are shared across bugs) and concurrent across
lanes, dispatched from a shared work queue for load balancing.

Each bug gets an isolated per-bug log directory (``logs/BIP/extraction/<run_id>/<project>/<bug>/``) so
the 1-second-granularity run-report filenames never collide; the subprocess's own stdout/stderr is
teed to ``driver.out`` there. After the run, a completeness audit (see ``bip_audit``) is written to
the run directory.

Usage examples::

    # Full force re-run of all 501 bugs across 5 lanes (then audit)
    python fl_methods/run_bip_extraction_all.py --run-id 20260625_120000

    # Resume an interrupted run (skip already-complete bugs), 3 lanes
    python fl_methods/run_bip_extraction_all.py --run-id 20260625_120000 --resume --lanes 3

    # Dry-run: just show the plan
    python fl_methods/run_bip_extraction_all.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.common.config import get_logs_dir
from src.extraction.bip_audit import (
    audit_all,
    is_bug_complete,
    iter_all_bugs,
    summarize,
    write_reports,
)
from src.extraction.bugsinpy import BugsInPyRepo

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_EXTRACTION = "fl_methods/run_extraction.py"
SETUP_SCRIPT = REPO_ROOT / "utils" / "docker" / "bugsinpy" / "setup-docker-bip.sh"
EXTRACTION_LOGS = get_logs_dir("extraction", "bugsinpy")
LANE_PREFIX = "bugsinpy-cefl-lane-"

# Signatures in a failed bug's driver.out that indicate the lane container (not the bug) died.
_DOCKER_DEATH_SIGNATURES = (
    "Cannot connect to the Docker daemon",
    "Error response from daemon",
    "No such container",
    "is not running",
)


def _lane_name(lane_idx: int) -> str:
    return f"{LANE_PREFIX}{lane_idx}"


@dataclass
class BugOutcome:
    """The result of one extraction subprocess (one attempt)."""

    project: str
    bug_id: int
    lane: int
    attempt: int
    exit_code: int | None  # None on timeout
    outcome: str  # ok | failed | timeout | retried | error
    duration_sec: float


# ---------------------------------------------------------------------------
# Lane container lifecycle
# ---------------------------------------------------------------------------


def prepare_lanes(lanes: int, run_dir: Path) -> None:
    """Create one container per lane from the shared image (serial — builds the image once)."""
    setup_log = run_dir / "lane-setup.log"
    with setup_log.open("a", encoding="utf-8") as fh:
        for i in range(lanes):
            lane = _lane_name(i)
            logger.info("Preparing lane container %s (this builds the image on first lane)", lane)
            fh.write(f"\n===== setup {lane} =====\n")
            fh.flush()
            env = {**os.environ, "CEFL_BIP_CONTAINER": lane}
            proc = subprocess.run(
                [str(SETUP_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Failed to prepare lane container {lane} (rc={proc.returncode}); see {setup_log}"
                )


def _restart_container(lane: int) -> None:
    """Restart a lane container (kills orphaned in-container processes; keeps conda envs)."""
    subprocess.run(
        ["docker", "restart", _lane_name(lane)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _recreate_container(lane: int, run_dir: Path) -> None:
    """Recreate a dead lane container from the shared image (envs are lost; rebuilt on demand)."""
    with (run_dir / "lane-setup.log").open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== recover {_lane_name(lane)} =====\n")
        fh.flush()
        subprocess.run(
            [str(SETUP_SCRIPT)],
            cwd=REPO_ROOT,
            env={**os.environ, "CEFL_BIP_CONTAINER": _lane_name(lane)},
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )


# ---------------------------------------------------------------------------
# Single-bug dispatch
# ---------------------------------------------------------------------------


def _per_bug_log_dir(run_id: str, project: str, bug_id: int) -> Path:
    return EXTRACTION_LOGS / run_id / project / str(bug_id)


def _run_one_bug(
    project: str,
    bug_id: int,
    lane_idx: int,
    attempt: int,
    *,
    run_id: str,
    force: bool,
    timeout_sec: int,
) -> BugOutcome:
    """Run one bug's extraction as a subprocess pinned to *lane_idx*'s container."""
    lane = _lane_name(lane_idx)
    log_dir = _per_bug_log_dir(run_id, project, bug_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        RUN_EXTRACTION,
        "--benchmark",
        "bugsinpy",
        "-p",
        project,
        "-v",
        str(bug_id),
        "--log-dir",
        str(log_dir),
    ]
    if force:
        cmd.append("--force")
    env = {**os.environ, "CEFL_BIP_CONTAINER": lane}

    started = time.time()
    exit_code: int | None
    with (log_dir / "driver.out").open("w", encoding="utf-8") as fh:
        fh.write(f"$ CEFL_BIP_CONTAINER={lane} {' '.join(cmd)}\n\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            exit_code = proc.returncode
            outcome = "ok" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            exit_code = None
            outcome = "timeout"
            fh.write(f"\n----- TIMEOUT after {timeout_sec}s; subprocess killed -----\n")

    return BugOutcome(
        project=project,
        bug_id=bug_id,
        lane=lane_idx,
        attempt=attempt,
        exit_code=exit_code,
        outcome=outcome,
        duration_sec=round(time.time() - started, 2),
    )


def _driver_out_signals_lane_death(run_id: str, project: str, bug_id: int) -> bool:
    """True if the bug's driver.out tail contains a Docker-daemon/container-death signature."""
    path = _per_bug_log_dir(run_id, project, bug_id) / "driver.out"
    if not path.exists():
        return False
    tail = path.read_text(encoding="utf-8", errors="ignore")[-4000:]
    return any(sig in tail for sig in _DOCKER_DEATH_SIGNATURES)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_pairs(projects: list[str] | None, resume: bool) -> list[tuple[str, int]]:
    pairs = iter_all_bugs(projects)
    if not resume:
        return pairs
    todo = [(p, b) for (p, b) in pairs if not is_bug_complete(BugsInPyRepo(p, b))]
    logger.info(
        "resume: %d/%d bugs already complete; %d to run",
        len(pairs) - len(todo),
        len(pairs),
        len(todo),
    )
    return todo


def run(
    *,
    projects: list[str] | None,
    lanes: int,
    run_id: str,
    force: bool,
    resume: bool,
    timeout_sec: int,
    max_attempts: int,
    do_audit: bool,
) -> dict[str, int]:
    """Execute the full parallel extraction and (optionally) the post-run audit."""
    run_dir = EXTRACTION_LOGS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pairs = _resolve_pairs(projects, resume)
    if not pairs:
        logger.info("Nothing to run.")
        return {"total": 0}

    prepare_lanes(lanes, run_dir)

    work: queue.Queue[tuple[str, int, int] | None] = queue.Queue()
    for project, bug_id in pairs:
        work.put((project, bug_id, 1))

    results: list[BugOutcome] = []
    results_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress_path = run_dir / "progress.jsonl"
    done = [0]
    total = len(pairs)

    def _record(outcome: BugOutcome, *, final: bool) -> None:
        with progress_lock, progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(outcome)) + "\n")
        if final:
            with results_lock:
                results.append(outcome)
                done[0] += 1
                n = done[0]
            logger.info(
                "[%d/%d] %s/%s lane=%d -> %s (%.0fs)",
                n,
                total,
                outcome.project,
                outcome.bug_id,
                outcome.lane,
                outcome.outcome,
                outcome.duration_sec,
            )

    def _worker(lane_idx: int) -> None:
        while True:
            item = work.get()
            try:
                if item is None:
                    return
                project, bug_id, attempt = item
                try:
                    outcome = _run_one_bug(
                        project,
                        bug_id,
                        lane_idx,
                        attempt,
                        run_id=run_id,
                        force=force,
                        timeout_sec=timeout_sec,
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception("lane %d crashed on %s/%s", lane_idx, project, bug_id)
                    _record(
                        BugOutcome(project, bug_id, lane_idx, attempt, None, "error", 0.0),
                        final=True,
                    )
                    continue

                lane_dead = outcome.outcome == "failed" and _driver_out_signals_lane_death(
                    run_id, project, bug_id
                )
                retryable = outcome.outcome == "timeout" or lane_dead
                if retryable and attempt < max_attempts:
                    outcome.outcome = "retried"
                    _record(outcome, final=False)
                    if lane_dead:
                        _recreate_container(lane_idx, run_dir)
                    else:
                        _restart_container(lane_idx)
                    work.put((project, bug_id, attempt + 1))
                else:
                    _record(outcome, final=True)
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=_worker, args=(i,), name=f"lane-{i}", daemon=True)
        for i in range(lanes)
    ]
    for t in threads:
        t.start()
    work.join()  # wait for all bugs (including requeued retries) to finish
    for _ in threads:
        work.put(None)  # sentinel per worker
    for t in threads:
        t.join()

    summary = _write_manifest(run_dir, results, run_id, lanes, force, resume, timeout_sec)
    logger.info("Extraction summary: %s", summary)

    if do_audit:
        logger.info("Auditing results...")
        audit_results = audit_all(projects, log_root=run_dir)
        csv_path, json_path = write_reports(audit_results, run_dir)
        logger.info("Audit: %s", summarize(audit_results))
        logger.info("wrote %s and %s", csv_path, json_path)
    return summary


def _write_manifest(
    run_dir: Path,
    results: list[BugOutcome],
    run_id: str,
    lanes: int,
    force: bool,
    resume: bool,
    timeout_sec: int,
) -> dict[str, int]:
    """Single-writer (main thread) aggregate manifest of the run."""
    summary: dict[str, int] = {"total": len(results)}
    for r in results:
        summary[r.outcome] = summary.get(r.outcome, 0) + 1
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lanes": lanes,
        "force": force,
        "resume": resume,
        "timeout_sec": timeout_sec,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drive the full BugsInPy extraction in parallel.")
    parser.add_argument("--projects", nargs="*", default=None, help="Restrict to these projects.")
    parser.add_argument(
        "--lanes", type=int, default=5, help="Concurrent container lanes (default 5)."
    )
    parser.add_argument("--run-id", default=None, help="Run id (default: timestamp).")
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Do not pass --force (skip already-extracted steps).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip bugs whose outputs are already complete (per the audit classifier).",
    )
    parser.add_argument(
        "--timeout-min", type=int, default=40, help="Per-bug wall-clock timeout (default 40 min)."
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Attempts per bug; >1 enables one retry on timeout/lane-death (default 2).",
    )
    parser.add_argument(
        "--no-audit", dest="audit", action="store_false", help="Skip the post-run audit."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(force=True, audit=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs = _resolve_pairs(args.projects, args.resume)
    logger.info(
        "Run %s: %d bugs, %d lanes, force=%s, timeout=%dmin",
        run_id,
        len(pairs),
        args.lanes,
        args.force,
        args.timeout_min,
    )
    if args.dry_run:
        for project, bug_id in pairs:
            print(f"{project}/{bug_id}")
        print(
            f"\n{len(pairs)} bugs across {args.lanes} lanes (lane containers {LANE_PREFIX}0..{args.lanes - 1})"
        )
        return 0

    summary = run(
        projects=args.projects,
        lanes=args.lanes,
        run_id=run_id,
        force=args.force,
        resume=args.resume,
        timeout_sec=args.timeout_min * 60,
        max_attempts=args.max_attempts,
        do_audit=args.audit,
    )
    # Exit non-zero if any bug did not finish cleanly, so CI/cron can detect it.
    unclean = summary.get("total", 0) - summary.get("ok", 0)
    return 1 if unclean else 0


if __name__ == "__main__":
    sys.exit(main())
