"""Completeness audit for BugsInPy extraction outputs.

Classifies each processed bug directory (``data/BIP/processed/<project>/<bug>/``) so a full
re-extraction can be reviewed bug-by-bug. Built entirely on the existing
:func:`src.extraction.validation.validate_extraction_outputs`, plus one cross-check that
validation does *not* perform: whether the faulty method actually appears in the coverage spectra.

Statuses:

- ``ok`` — every required deliverable present & well-formed, no warnings, and (when coverage
  exists) the faulty method is represented in the spectra.
- ``ok_reduced`` — no validation *errors*, but either FauxPy is unsupported for this bug
  (the reduced deliverable set, e.g. cookiecutter/4 @ Python 3.5) or only *warnings* are present
  (e.g. ``bug_report.json`` missing on a GitHub rate-limit). Not flagged for review.
- ``fl_unreachable`` — all deliverables present & valid, but **no** faulty method appears in the
  spectra, so FL can never localize it (the known coverage-gap class). Flagged for review.
- ``real_failure`` — at least one required deliverable is missing/malformed (a validation
  ``error``). Flagged for review.
- ``missing`` — the processed dir does not exist (never extracted). Flagged for review.

``is_bug_complete`` (status in {ok, ok_reduced}) is reused by ``run_bip_extraction_all.py`` to
pre-filter the work queue on ``--resume``.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.extraction.bugsinpy import BugsInPyRepo, get_bip_bids, get_bip_pids
from src.extraction.fauxpy import fauxpy_supported
from src.extraction.validation import validate_extraction_outputs

logger = logging.getLogger(__name__)

# Statuses that do NOT require human review (the dataset is usable for this bug).
COMPLETE_STATUSES = frozenset({"ok", "ok_reduced"})
# Statuses surfaced for review (per the user's "flag both" choice, fl_unreachable is included).
REVIEW_STATUSES = frozenset({"real_failure", "fl_unreachable", "missing"})

_FAUXPY_LOG_TAIL_LINES = 40


@dataclass(frozen=True)
class BugAuditResult:
    """The audit verdict for one BugsInPy bug."""

    project: str
    bug_id: int
    status: str
    fauxpy_supported: bool
    issues: list[dict[str, str]] = field(default_factory=list)
    fault_sigs: tuple[str, ...] = ()
    unreached_sigs: tuple[str, ...] = ()
    error_msg: str = ""
    fauxpy_log_tail: str = ""

    @property
    def needs_review(self) -> bool:
        return self.status in REVIEW_STATUSES

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.get("severity") == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.get("severity") == "warning")

    def to_row(self) -> dict[str, str]:
        """Flatten to the audit.csv row schema."""
        return {
            "project": self.project,
            "bug": str(self.bug_id),
            "status": self.status,
            "n_errors": str(self.n_errors),
            "n_warnings": str(self.n_warnings),
            "fauxpy_supported": str(self.fauxpy_supported).lower(),
            "fault_sigs": str(len(self.fault_sigs)),
            "unreached_sigs": str(len(self.unreached_sigs)),
            "error_msg": _one_line(self.error_msg),
        }


AUDIT_CSV_COLUMNS = (
    "project",
    "bug",
    "status",
    "n_errors",
    "n_warnings",
    "fauxpy_supported",
    "fault_sigs",
    "unreached_sigs",
    "error_msg",
)


# ---------------------------------------------------------------------------
# fault.csv <-> spectra cross-check (the silent FL-unreachable class)
# ---------------------------------------------------------------------------


def read_fault_signatures(output_dir: Path) -> set[str]:
    """Return the non-empty method ``signature`` values from ``faults.csv``.

    The signature is a bare method corpus_id (``module$qualname(params)``). Rows whose signature
    is empty (no method mapped) are skipped — they cannot participate in the spectra cross-check.
    """
    path = output_dir / "faults.csv"
    if not path.exists():
        return set()
    sigs: set[str] = set()
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sig = (row.get("signature") or "").strip()
            if sig:
                sigs.add(sig)
    return sigs


def read_spectra_methods(output_dir: Path) -> set[str]:
    """Return the set of method corpus_ids covered in ``FauxPy/coverage/spectra.csv``.

    Spectra rows are ``<corpus_id>:<line>``; the trailing ``:<line>`` is stripped (corpus_ids
    contain no colon) so the result is comparable to ``faults.csv`` signatures by equality.
    """
    path = output_dir / "FauxPy" / "coverage" / "spectra.csv"
    if not path.exists():
        return set()
    methods: set[str] = set()
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or (idx == 0 and stripped == "name"):
            continue
        methods.add(stripped.rsplit(":", 1)[0])
    return methods


def _fauxpy_log_tail(output_dir: Path) -> str:
    """Return the last lines of ``FauxPy/raw/fauxpy_run.log`` (root-cause diagnostic), if present."""
    path = output_dir / "FauxPy" / "raw" / "fauxpy_run.log"
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-_FAUXPY_LOG_TAIL_LINES:])


def _run_report_error(output_dir: Path, log_root: Path | None) -> str:
    """Best-effort: the per-bug ``error`` recorded in a run-report JSON, or ``""``.

    When *log_root* is given (a driver run dir), look at ``<log_root>/<project>/<bug>/`` first
    (the per-bug isolated logs this driver writes). Falls back to scanning the bug's own processed
    tree is unnecessary — run reports never live there.
    """
    if log_root is None:
        return ""
    project = output_dir.parent.name
    bug = output_dir.name
    bug_log_dir = log_root / project / bug
    if not bug_log_dir.is_dir():
        return ""
    for report in sorted(bug_log_dir.glob("extraction_*.json"), reverse=True):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in data.get("results", []):
            if str(entry.get("bug_id")) == bug and entry.get("error"):
                return str(entry["error"])
    return ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def decide_status(
    *,
    has_errors: bool,
    has_warnings: bool,
    fauxpy_supported: bool,
    fault_sigs: set[str],
    unreached: set[str],
) -> str:
    """Pure decision table mapping computed signals to an audit status.

    Precedence: a validation error always wins; then FL-unreachability (only meaningful when
    coverage exists and every faulty method is absent from the spectra); then the reduced/warning
    case (FauxPy-unsupported or warning-only); otherwise clean.
    """
    if has_errors:
        return "real_failure"
    if fauxpy_supported and fault_sigs and unreached == fault_sigs:
        return "fl_unreachable"
    if not fauxpy_supported or has_warnings:
        return "ok_reduced"
    return "ok"


def classify_bug(repo: BugsInPyRepo, *, log_root: Path | None = None) -> BugAuditResult:
    """Classify a single bug's processed outputs into a :class:`BugAuditResult`."""
    output_dir = repo.output_dir
    fxs = fauxpy_supported(repo)

    if not output_dir.exists():
        return BugAuditResult(repo.project, repo.bug_id, "missing", fxs)

    issues = validate_extraction_outputs(
        output_dir,
        expect_gzoltar=fxs,
        expect_faults=True,
        expect_bug_report=True,
        dataset="bugsinpy",
    )
    has_errors = any(i.get("severity") == "error" for i in issues)
    has_warnings = any(i.get("severity") == "warning" for i in issues)

    fault_sigs: set[str] = set()
    unreached: set[str] = set()
    if fxs:
        fault_sigs = read_fault_signatures(output_dir)
        if fault_sigs:
            unreached = fault_sigs - read_spectra_methods(output_dir)

    status = decide_status(
        has_errors=has_errors,
        has_warnings=has_warnings,
        fauxpy_supported=fxs,
        fault_sigs=fault_sigs,
        unreached=unreached,
    )

    error_msg = ""
    log_tail = ""
    if status in REVIEW_STATUSES:
        error_msg = _run_report_error(output_dir, log_root)
        log_tail = _fauxpy_log_tail(output_dir)

    return BugAuditResult(
        project=repo.project,
        bug_id=repo.bug_id,
        status=status,
        fauxpy_supported=fxs,
        issues=issues,
        fault_sigs=tuple(sorted(fault_sigs)),
        unreached_sigs=tuple(sorted(unreached)),
        error_msg=error_msg,
        fauxpy_log_tail=log_tail,
    )


def is_bug_complete(repo: BugsInPyRepo) -> bool:
    """True when the bug's outputs are usable (no review needed) — used for ``--resume``."""
    return classify_bug(repo).status in COMPLETE_STATUSES


def iter_all_bugs(projects: list[str] | None = None) -> list[tuple[str, int]]:
    """Return all ``(project, bug_id)`` pairs from the BugsInPy index, sorted."""
    pids = projects if projects is not None else get_bip_pids()
    pairs: list[tuple[str, int]] = []
    for project in pids:
        for bug_id in get_bip_bids(project):
            pairs.append((project, bug_id))
    return pairs


def audit_all(
    projects: list[str] | None = None, *, log_root: Path | None = None
) -> list[BugAuditResult]:
    """Classify every BugsInPy bug (optionally restricted to *projects*)."""
    results: list[BugAuditResult] = []
    for project, bug_id in iter_all_bugs(projects):
        results.append(classify_bug(BugsInPyRepo(project, bug_id), log_root=log_root))
    return results


def summarize(results: list[BugAuditResult]) -> dict[str, int]:
    """Return a ``{status: count}`` tally plus a ``total`` and ``needs_review`` roll-up."""
    summary: dict[str, int] = {"total": len(results), "needs_review": 0}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
        if r.needs_review:
            summary["needs_review"] += 1
    return summary


def _result_to_dict(r: BugAuditResult) -> dict[str, object]:
    """Full (review-detailed) representation for ``audit.json``."""
    return {
        "project": r.project,
        "bug_id": r.bug_id,
        "status": r.status,
        "fauxpy_supported": r.fauxpy_supported,
        "issues": r.issues,
        "fault_sigs": list(r.fault_sigs),
        "unreached_sigs": list(r.unreached_sigs),
        "error_msg": r.error_msg,
        "fauxpy_log_tail": r.fauxpy_log_tail,
    }


def write_reports(results: list[BugAuditResult], out_dir: Path) -> tuple[Path, Path]:
    """Write ``audit.csv`` (flat, one row per bug) and ``audit.json`` (full detail).

    Returns ``(csv_path, json_path)``. Shared by the audit CLI and the extraction driver.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())

    json_path = out_dir / "audit.json"
    payload = {"summary": summarize(results), "results": [_result_to_dict(r) for r in results]}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _one_line(text: str) -> str:
    """Collapse a multi-line message to a single CSV-safe line."""
    return " ".join(text.split())
