"""CLI: sweep the full BugsInPy pipeline (extraction -> corpus -> Agent4SR) for every bug.

This is the end-to-end companion to :mod:`run_bip_extraction_all` (which stops after
extraction). For each bug it runs three stages in order:

1. **extraction** -- ``run_extraction.py --benchmark bugsinpy`` (container-bound: checkout,
   conda env, FauxPy coverage). Skipped when the bug already audits complete.
2. **corpus** -- ``run_agent4sr.py corpus`` (container-bound checkout; builds
   ``FlexFL/SR/corpus_methods.txt``). Self-gates: only audit-complete bugs get a corpus.
3. **sr** -- ``run_agent4sr.py run`` (host-side HTTPS to the Ollama endpoint; no container).
   Self-gates on the corpus + a blank-trigger skip, then writes ``sr_result.json``.

Concurrency follows the same lane model as ``run_bip_extraction_all``: extraction and corpus
are **not** safe to parallelise under one container (per-bug conda envs are shared by hash and
FauxPy mutates them), so each lane owns its own ``bugsinpy-cefl-lane-{i}`` container and runs a
bug fully (all three stages) before taking the next from a shared queue. The SR stage is a
host-side HTTP call, so it rides along on the lane's worker thread without touching the container.

Progress, elapsed time and ETA are rendered live with ``tqdm`` (one bar over all bugs, advanced as
each bug finishes), and every finished bug prints a one-line summary above the bar. A per-bug log
dir, a ``progress.jsonl`` stream and a final ``manifest.json`` land under
``logs/BIP/sr_sweep/<run_id>/``.

Usage examples::

    # Resume sweep of all 501 bugs across 3 lanes against the BGU Ollama endpoint
    python fl_methods/run_bip_sr_sweep.py

    # Smoke-test: one project, one lane, no resume-skip
    python fl_methods/run_bip_sr_sweep.py --projects PySnooper --lanes 1 --limit 1 --force

    # Just show the plan
    python fl_methods/run_bip_sr_sweep.py --dry-run
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

# Reuse the proven lane-container lifecycle from the extraction driver (single source of truth).
from run_bip_extraction_all import (
    _DOCKER_DEATH_SIGNATURES,
    _lane_name,
    _recreate_container,
    _restart_container,
    prepare_lanes,
)
from tqdm import tqdm  # type: ignore[import-untyped]

from src.common.bip_gate import bip_corpus_exists, classify_bip_bug
from src.common.config import get_logs_dir, get_processed_dir
from src.extraction.bip_audit import is_bug_complete, iter_all_bugs
from src.extraction.bugsinpy import BugsInPyRepo

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_EXTRACTION = "fl_methods/run_extraction.py"
RUN_AGENT4SR = "fl_methods/run_agent4sr.py"
SWEEP_LOGS = get_logs_dir("sr_sweep", "bugsinpy")

DEFAULT_BASE_URL = "https://cis-ollama.auth.ad.bgu.ac.il"
DEFAULT_MODEL = "llama3.1:8b"

# Per-bug outcome glyphs for the live one-line summaries.
_GLYPH = {"ok": "✓", "skipped": "⊘", "invalid": "∅", "failed": "✗"}


@dataclass
class StageResult:
    """One stage's outcome for a bug: ``ran`` | ``skipped`` | ``failed`` | ``n/a``."""

    status: str
    duration_sec: float = 0.0
    exit_code: int | None = None


@dataclass
class BugResult:
    """Aggregate outcome of one bug across all three stages."""

    project: str
    bug_id: int
    lane: int
    outcome: str  # ok | skipped | invalid | failed
    reason: str
    extraction: StageResult
    corpus: StageResult
    sr: StageResult
    duration_sec: float


# ---------------------------------------------------------------------------
# Resume / completion probes
# ---------------------------------------------------------------------------


def _sr_result_exists(project: str, bug_id: int) -> bool:
    """True when an Agent4SR ``sr_result.json`` exists for *any* model slug of this bug."""
    sr_root = get_processed_dir(project, bug_id, dataset="bugsinpy") / "FlexFL" / "SR" / "Agent4SR"
    if not sr_root.is_dir():
        return False
    return any(sr_root.glob("*/sr_result.json"))


def _per_bug_log_dir(run_id: str, project: str, bug_id: int) -> Path:
    return SWEEP_LOGS / run_id / project / str(bug_id)


def _log_signals_lane_death(path: Path) -> bool:
    """True if a stage log's tail carries a Docker-daemon / container-death signature."""
    if not path.exists():
        return False
    tail = path.read_text(encoding="utf-8", errors="ignore")[-4000:]
    return any(sig in tail for sig in _DOCKER_DEATH_SIGNATURES)


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _run_stage(
    cmd: list[str], log_path: Path, *, lane: int, timeout_sec: int
) -> tuple[str, float, int | None]:
    """Run one stage as a subprocess pinned to *lane*'s container.

    Returns ``(status, duration_sec, exit_code)`` where status is ``ran`` (rc 0),
    ``failed`` (rc != 0), or ``timeout`` (killed). Output is teed to *log_path*.
    """
    env = {**os.environ, "CEFL_BIP_CONTAINER": _lane_name(lane)}
    started = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"$ CEFL_BIP_CONTAINER={_lane_name(lane)} {' '.join(cmd)}\n\n")
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
            status = "ran" if proc.returncode == 0 else "failed"
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n----- TIMEOUT after {timeout_sec}s; subprocess killed -----\n")
            status, exit_code = "timeout", None
    return status, round(time.time() - started, 2), exit_code


def _extraction_cmd(project: str, bug_id: int, log_dir: Path, *, force: bool) -> list[str]:
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
    return cmd


def _corpus_cmd(project: str, bug_id: int, *, force: bool) -> list[str]:
    cmd = [
        sys.executable,
        RUN_AGENT4SR,
        "corpus",
        "--benchmark",
        "bugsinpy",
        "-p",
        project,
        "-v",
        str(bug_id),
    ]
    if force:
        cmd.append("--force")
    return cmd


def _sr_cmd(project: str, bug_id: int, cfg: SweepConfig) -> list[str]:
    cmd = [
        sys.executable,
        RUN_AGENT4SR,
        "run",
        "--benchmark",
        "bugsinpy",
        "-p",
        project,
        "-v",
        str(bug_id),
        "--model",
        cfg.model,
        "--base-url",
        cfg.base_url,
        "--iterations",
        str(cfg.iterations),
    ]
    if cfg.no_verify:
        cmd.append("--no-verify")
    if cfg.force:
        cmd.append("--force")
    return cmd


@dataclass(frozen=True)
class SweepConfig:
    """Run-wide knobs shared by every lane worker."""

    model: str
    base_url: str
    iterations: int
    no_verify: bool
    force: bool
    extract_timeout_sec: int
    corpus_timeout_sec: int
    sr_timeout_sec: int
    max_attempts: int


def _process_bug(
    project: str, bug_id: int, lane: int, *, run_id: str, cfg: SweepConfig, run_dir: Path
) -> BugResult:
    """Run extraction -> corpus -> sr for one bug, returning its aggregate outcome."""
    log_dir = _per_bug_log_dir(run_id, project, bug_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    repo = BugsInPyRepo(project, bug_id)
    started = time.time()

    ext = StageResult("n/a")
    corpus = StageResult("n/a")
    sr = StageResult("n/a")

    def _finish(outcome: str, reason: str) -> BugResult:
        return BugResult(
            project=project,
            bug_id=bug_id,
            lane=lane,
            outcome=outcome,
            reason=reason,
            extraction=ext,
            corpus=corpus,
            sr=sr,
            duration_sec=round(time.time() - started, 2),
        )

    # Whole-bug resume short-circuit: SR already done and extraction complete.
    if not cfg.force and _sr_result_exists(project, bug_id) and is_bug_complete(repo):
        ext = StageResult("skipped")
        corpus = StageResult("skipped")
        sr = StageResult("skipped")
        return _finish("skipped", "already SR-complete")

    # --- Stage 1: extraction (with one container-death retry) ---
    if not cfg.force and is_bug_complete(repo):
        ext = StageResult("skipped")
    else:
        for attempt in range(1, cfg.max_attempts + 1):
            log_path = log_dir / "extraction.out"
            status, dur, rc = _run_stage(
                _extraction_cmd(project, bug_id, log_dir, force=cfg.force),
                log_path,
                lane=lane,
                timeout_sec=cfg.extract_timeout_sec,
            )
            ext = StageResult("ran" if status == "ran" else "failed", dur, rc)
            lane_dead = status != "ran" and _log_signals_lane_death(log_path)
            retryable = (status == "timeout" or lane_dead) and attempt < cfg.max_attempts
            if not retryable:
                break
            if lane_dead:
                _recreate_container(lane, run_dir)
            else:
                _restart_container(lane)

    # --- Stage 2: corpus ---
    if not cfg.force and bip_corpus_exists(project, bug_id):
        corpus = StageResult("skipped")
    elif ext.status == "failed":
        # No point building a corpus if extraction died; leave corpus/sr as n/a.
        return _finish("failed", f"extraction failed (rc={ext.exit_code})")
    else:
        status, dur, rc = _run_stage(
            _corpus_cmd(project, bug_id, force=cfg.force),
            log_dir / "corpus.out",
            lane=lane,
            timeout_sec=cfg.corpus_timeout_sec,
        )
        corpus = StageResult("ran" if status == "ran" else "failed", dur, rc)

    # Did the bug actually pass the valid-bug gate? (corpus self-skips invalid bugs.)
    _, gate_reason = classify_bip_bug(project, bug_id)
    if gate_reason is not None:
        return _finish("invalid", gate_reason)
    if not bip_corpus_exists(project, bug_id):
        return _finish("failed", f"corpus missing after build (rc={corpus.exit_code})")

    # --- Stage 3: Agent4SR run (host-side HTTPS; no container) ---
    if not cfg.force and _sr_result_exists(project, bug_id):
        sr = StageResult("skipped")
        return _finish("ok", "sr already present")
    status, dur, rc = _run_stage(
        _sr_cmd(project, bug_id, cfg),
        log_dir / "sr.out",
        lane=lane,
        timeout_sec=cfg.sr_timeout_sec,
    )
    sr = StageResult("ran" if status == "ran" else "failed", dur, rc)
    if _sr_result_exists(project, bug_id):
        return _finish("ok", "sr produced")
    # rc 0 but no result => the run subcommand gate-skipped (e.g. blank trigger).
    if status == "ran":
        return _finish("invalid", "sr produced no result (gate/blank-trigger skip)")
    return _finish("failed", f"sr failed (rc={rc})")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_pairs(
    projects: list[str] | None, *, resume: bool, force: bool = False, limit: int | None
) -> list[tuple[str, int]]:
    """Enumerate (project, bug) pairs, optionally dropping already-SR-complete bugs.

    ``--force`` re-runs every stage, so it must bypass the resume pre-filter — otherwise
    already-SR-complete bugs are dropped from the plan before ``--force`` can apply.
    """
    pairs = iter_all_bugs(projects)
    if resume and not force:
        kept = [
            (p, b)
            for (p, b) in pairs
            if not (_sr_result_exists(p, b) and is_bug_complete(BugsInPyRepo(p, b)))
        ]
        logger.info(
            "resume: %d/%d bugs already SR-complete; %d to run",
            len(pairs) - len(kept),
            len(pairs),
            len(kept),
        )
        pairs = kept
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _summary_line(r: BugResult) -> str:
    """A single dense line printed above the progress bar for each finished bug."""
    glyph = _GLYPH.get(r.outcome, "?")
    stages = []
    for name, st in (("ext", r.extraction), ("cor", r.corpus), ("sr", r.sr)):
        if st.status == "ran":
            stages.append(f"{name} {int(st.duration_sec)}s")
        elif st.status == "skipped":
            stages.append(f"{name} skip")
    detail = "  ".join(stages) if stages else r.reason
    tail = f" — {r.reason}" if r.outcome in {"failed", "invalid"} else ""
    return (
        f"{glyph} {r.project}/{r.bug_id} (lane {r.lane})  {detail}  [{int(r.duration_sec)}s]{tail}"
    )


def run(
    *, pairs: list[tuple[str, int]], lanes: int, run_id: str, cfg: SweepConfig
) -> dict[str, int]:
    """Execute the sweep across *lanes* container lanes with a live progress bar."""
    run_dir = SWEEP_LOGS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(pairs)

    prepare_lanes(lanes, run_dir)

    work: queue.Queue[tuple[str, int] | None] = queue.Queue()
    for pair in pairs:
        work.put(pair)
    results_q: queue.Queue[BugResult] = queue.Queue()

    def _worker(lane: int) -> None:
        while True:
            item = work.get()
            try:
                if item is None:
                    return
                project, bug_id = item
                try:
                    result = _process_bug(
                        project, bug_id, lane, run_id=run_id, cfg=cfg, run_dir=run_dir
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("lane %d crashed on %s/%s", lane, project, bug_id)
                    na = StageResult("n/a")
                    result = BugResult(
                        project, bug_id, lane, "failed", f"driver error: {exc!r}", na, na, na, 0.0
                    )
                results_q.put(result)
            finally:
                work.task_done()

    threads = [
        threading.Thread(target=_worker, args=(i,), name=f"lane-{i}", daemon=True)
        for i in range(lanes)
    ]
    for t in threads:
        t.start()

    counts = {"ok": 0, "skipped": 0, "invalid": 0, "failed": 0}
    progress_path = run_dir / "progress.jsonl"
    results: list[BugResult] = []

    bar = tqdm(total=total, desc="BIP sweep", unit="bug", dynamic_ncols=True, smoothing=0.1)
    with bar, progress_path.open("w", encoding="utf-8") as pfh:
        for _ in range(total):
            r = results_q.get()
            results.append(r)
            counts[r.outcome] = counts.get(r.outcome, 0) + 1
            pfh.write(json.dumps(asdict(r)) + "\n")
            pfh.flush()
            bar.write(_summary_line(r))
            done = len(results)
            avg = bar.format_dict["elapsed"] / done if done else 0.0
            eta = avg * (total - done)
            bar.set_postfix_str(
                f"ok={counts['ok']} skip={counts['skipped']} inv={counts['invalid']} "
                f"fail={counts['failed']} | ETA {_fmt_eta(eta)}"
            )
            bar.update(1)

    # All bugs consumed; release the workers.
    for _ in threads:
        work.put(None)
    for t in threads:
        t.join()

    summary = {"total": total, **counts}
    _write_manifest(run_dir, results, run_id, lanes, cfg, summary)
    return summary


def _write_manifest(
    run_dir: Path,
    results: list[BugResult],
    run_id: str,
    lanes: int,
    cfg: SweepConfig,
    summary: dict[str, int],
) -> None:
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lanes": lanes,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep extraction -> corpus -> Agent4SR over all BugsInPy bugs."
    )
    parser.add_argument("--projects", nargs="*", default=None, help="Restrict to these projects.")
    parser.add_argument("--lanes", type=int, default=3, help="Concurrent container lanes (def 3).")
    parser.add_argument("--run-id", default=None, help="Run id (default: timestamp).")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama model (def {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"Ollama endpoint (def {DEFAULT_BASE_URL})."
    )
    parser.add_argument("--iterations", type=int, default=10, help="Max SR tool-call iterations.")
    parser.add_argument(
        "--verify-ssl",
        dest="no_verify",
        action="store_false",
        help="Verify the Ollama TLS cert (default: --no-verify, for the self-signed BGU endpoint).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every stage even if outputs exist (also bypasses the resume pre-filter).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not pre-filter already-SR-complete bugs from the plan.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N bugs (smoke).")
    parser.add_argument("--extract-timeout-min", type=int, default=40)
    parser.add_argument("--corpus-timeout-min", type=int, default=20)
    parser.add_argument("--sr-timeout-min", type=int, default=30)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Extraction attempts (>1 retries on lane death).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(no_verify=True, resume=True)
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SWEEP_LOGS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(run_dir / "driver.log", encoding="utf-8")],
    )

    pairs = _resolve_pairs(args.projects, resume=args.resume, force=args.force, limit=args.limit)
    print(
        f"Run {run_id}: {len(pairs)} bugs, {args.lanes} lanes, model={args.model}, "
        f"endpoint={args.base_url}",
        file=sys.stderr,
    )
    if args.dry_run:
        for project, bug_id in pairs:
            print(f"{project}/{bug_id}")
        print(
            f"\n{len(pairs)} bugs across {args.lanes} lanes "
            f"(containers bugsinpy-cefl-lane-0..{args.lanes - 1})",
            file=sys.stderr,
        )
        return 0
    if not pairs:
        print("Nothing to run.", file=sys.stderr)
        return 0

    cfg = SweepConfig(
        model=args.model,
        base_url=args.base_url,
        iterations=args.iterations,
        no_verify=args.no_verify,
        force=args.force,
        extract_timeout_sec=args.extract_timeout_min * 60,
        corpus_timeout_sec=args.corpus_timeout_min * 60,
        sr_timeout_sec=args.sr_timeout_min * 60,
        max_attempts=args.max_attempts,
    )
    summary = run(pairs=pairs, lanes=args.lanes, run_id=run_id, cfg=cfg)
    print(
        f"\nDone: {summary['ok']} ok, {summary['skipped']} skipped, "
        f"{summary['invalid']} invalid, {summary['failed']} failed "
        f"(of {summary['total']}). Logs: {run_dir}",
        file=sys.stderr,
    )
    return 1 if summary.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
