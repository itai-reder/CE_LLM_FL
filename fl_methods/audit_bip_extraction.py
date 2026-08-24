"""CLI: audit the completeness of BugsInPy extraction outputs.

Classifies every bug (or a subset of projects) and writes ``audit.{json,csv}`` plus a printed
summary, flagging every bug whose required deliverables are missing/malformed (``real_failure``)
or whose faulty method never lands in the coverage spectra (``fl_unreachable``).

Usage examples::

    # Audit every bug, write reports next to the run's per-bug logs
    python fl_methods/audit_bip_extraction.py --run-id 20260625_120000

    # Audit a couple of projects, print only the bugs that need review
    python fl_methods/audit_bip_extraction.py --projects black fastapi --review-only

    # Audit everything against the current on-disk data (no run id), reports to a chosen dir
    python fl_methods/audit_bip_extraction.py --out-dir /tmp/bip-audit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.common.config import get_logs_dir
from src.extraction.bip_audit import BugAuditResult, audit_all, summarize, write_reports

logger = logging.getLogger(__name__)

_EXTRACTION_LOGS = get_logs_dir("extraction", "bugsinpy")


def _resolve_dirs(run_id: str | None, out_dir: str | None) -> tuple[Path | None, Path]:
    """Return ``(log_root, out_dir)`` for run-report lookup and report writing."""
    log_root = (_EXTRACTION_LOGS / run_id) if run_id else None
    if out_dir:
        out = Path(out_dir)
    elif run_id:
        out = _EXTRACTION_LOGS / run_id
    else:
        out = _EXTRACTION_LOGS
    return log_root, out


def _print_summary(results: list[BugAuditResult], *, review_only: bool) -> None:
    summary = summarize(results)
    order = ["ok", "ok_reduced", "fl_unreachable", "real_failure", "missing"]
    print("\n=== BIP extraction audit ===")
    print(f"total: {summary['total']}   needs review: {summary['needs_review']}")
    for status in order:
        if summary.get(status):
            print(f"  {status:16} {summary[status]:4}")

    flagged = [r for r in results if r.needs_review]
    if not flagged:
        print("\nNothing needs review. ✓")
        return
    print(f"\n--- needs review ({len(flagged)}) ---")
    for r in sorted(flagged, key=lambda x: (x.status, x.project, x.bug_id)):
        detail = r.error_msg or (r.issues[0]["message"] if r.issues else "")
        if r.status == "fl_unreachable":
            detail = f"fault method(s) not in spectra: {', '.join(r.unreached_sigs)}"
        print(f"  [{r.status}] {r.project}/{r.bug_id}: {detail[:140]}")
    if not review_only:
        print("\n(full per-bug issues + fauxpy_run.log tails in audit.json)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit BugsInPy extraction completeness.")
    parser.add_argument(
        "--projects",
        nargs="*",
        default=None,
        help="Restrict to these projects (default: all 17).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Driver run id; locates per-bug run-report logs and defaults the output dir.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write audit.{json,csv} (default: the run-id log dir, "
        "else logs/BIP/extraction).",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Print only the needs-review list (reports are still written in full).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log_root, out_dir = _resolve_dirs(args.run_id, args.out_dir)
    logger.info("Auditing BugsInPy outputs (projects=%s)", args.projects or "all")
    results = audit_all(args.projects, log_root=log_root)
    csv_path, json_path = write_reports(results, out_dir)
    _print_summary(results, review_only=args.review_only)
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
