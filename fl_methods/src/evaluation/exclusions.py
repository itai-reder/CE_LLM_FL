"""BugsInPy evaluation exclusions report.

Classifies every (project, bug) in the BugsInPy corpus that does **not**
produce a full set of evaluation rows, and writes the sidecar
``evaluation/BIP/exclusions.csv`` (``Project,BugId,bucket,scope,detail``).
The report is the authoritative explainer for cross-bug summary
denominators: the long/summary CSV schemas stay benchmark-identical, so
partial coverage must be legible *somewhere* — this is that somewhere.

Buckets (first match wins):

1. ``extraction_incomplete`` — audit status ``real_failure``/``missing`` (or
   the bug is absent from the audit altogether).
2. ``fl_unreachable`` — coverage valid but no faulty method in the spectra.
3. ``method_unlocalizable`` — ``faults.csv`` has rows but no method
   signature (the patch is entirely outside any ``def``, e.g. luigi/32,
   tqdm/7). Ranked above the SR/LR buckets because it is intrinsic to the
   bug, not transient sweep state.
4. ``sr_not_run`` — no ``FlexFL/SR/Agent4SR/*/sr_result.json`` (includes the
   blank-trigger / non-mirrorable-failure class; the detail says which).
5. ``lr_readiness_skipped`` — no resolvable SR top-20 (``combine`` not run),
   so there is no candidate universe at all.
6. ``lr_measurement_skipped`` — top-20 present but no fault in it and no LR
   result: baselines evaluate, flexfl slots stay empty ("unlocalizable").
7. ``lr_not_run`` — top-20 present, fault in it, but no LR run yet:
   baselines evaluate, flexfl slots stay empty until the LR sweep.

``scope`` says how much of the evaluation the bucket suppresses: ``all``
(no evaluation rows), ``flexfl`` (baselines evaluated, flexfl slots empty),
or ``aggregates`` (rows exist but carry blank FR and never enter a summary
aggregate). Fully evaluable bugs do not appear in the report.

Not an exclusion: an empty ``faults_first.csv``. Its blank-FR ``*_first``
rows are dropped by the summary aggregation rule by design.

Classification reads the extraction-audit snapshot
(``results/BIP/_meta/audit.csv``, written by ``run_build_results.py`` from
the ``audit_bip_extraction.py`` output) plus cheap checks over the slim
results tree — no container, no processed data, no re-audit. The audit
snapshot can be stale relative to the source data; regenerate it with
``audit_bip_extraction.py`` + ``run_build_results.py`` after extraction
changes.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.benchmarks.registry import get_benchmark_adapter
from src.common.config import get_results_bug_dir, get_results_meta_dir
from src.evaluation.cross_bug import get_cross_bug_dir
from src.evaluation.sources import DEFAULT_SR_MODEL_ID
from src.results.read import RANKINGS_SUBDIR, load_lr_json, load_meta

logger = logging.getLogger(__name__)

EXCLUSIONS_FILENAME = "exclusions.csv"
EXCLUSION_HEADERS = ("Project", "BugId", "bucket", "scope", "detail")

_AUDIT_INCOMPLETE = {"real_failure", "missing"}


def _audit_csv_path(dataset: str = "bugsinpy") -> Path:
    """Extraction-audit snapshot location (written by run_build_results.py)."""
    return get_results_meta_dir(dataset) / "audit.csv"


def _read_audit(audit_csv: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Return ``{(project, bug): row}`` from the audit CSV (empty if absent)."""
    if not audit_csv.exists():
        logger.warning("extraction audit missing: %s (all bugs will need it)", audit_csv)
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with audit_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            project = (row.get("project") or "").strip()
            bug = (row.get("bug") or "").strip()
            if project and bug:
                rows[(project, bug)] = row
    return rows


def _fault_signatures(results_dir: Path) -> tuple[int, set[str]]:
    """Return ``(n_rows, non_empty_signatures)`` from ``faults.csv``."""
    faults_csv = results_dir / "faults.csv"
    if not faults_csv.exists():
        return 0, set()
    n_rows = 0
    sigs: set[str] = set()
    with faults_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            sig = (row.get("signature") or "").strip()
            if sig:
                sigs.add(sig)
    return n_rows, sigs


def _top20_lines(results_dir: Path, sr_model_id: str) -> list[str] | None:
    """Non-blank SR top-20 lines, or ``None`` when the file is missing."""
    top20 = results_dir / RANKINGS_SUBDIR / "top20" / f"{sr_model_id}.txt"
    if not top20.exists():
        return None
    return [ln.strip() for ln in top20.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _has_sr_result(results_dir: Path) -> bool:
    """The builder records this processed-tree signal in ``meta.json``."""
    meta = load_meta(results_dir)
    return bool(meta and meta.get("has_sr_result"))


def _trigger_blank(results_dir: Path) -> bool:
    meta = load_meta(results_dir)
    return bool(meta and meta.get("trigger_blank"))


def _has_lr_result(results_dir: Path) -> bool:
    return bool(load_lr_json(results_dir))


def classify_exclusion(
    results_dir: Path,
    audit_row: dict[str, str] | None,
    *,
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
) -> tuple[str, str, str] | None:
    """Classify one bug; return ``(bucket, scope, detail)`` or ``None`` if evaluable.

    ``results_dir`` is the bug's slim results dir; ``audit_row`` is the bug's
    row from the extraction-audit snapshot (or ``None`` when absent from it).
    """
    # 1/2 — extraction audit gates.
    if audit_row is None:
        return ("extraction_incomplete", "all", "not in extraction audit")
    status = (audit_row.get("status") or "").strip()
    if status in _AUDIT_INCOMPLETE:
        detail = (audit_row.get("error_msg") or "").strip() or f"audit status {status}"
        return ("extraction_incomplete", "all", detail)
    if status == "fl_unreachable":
        n = (audit_row.get("unreached_sigs") or "?").strip()
        return (
            "fl_unreachable",
            "all",
            f"faulty method(s) not covered by spectra (n={n})",
        )

    # 3 — intrinsic: patch never touches a method body.
    n_fault_rows, fault_sigs = _fault_signatures(results_dir)
    if n_fault_rows > 0 and not fault_sigs:
        return (
            "method_unlocalizable",
            "aggregates",
            "no method signature in faults.csv (patch outside any def)",
        )

    # 4 — SR never ran (blank trigger or sweep gap).
    if not _has_sr_result(results_dir):
        if _trigger_blank(results_dir):
            return (
                "sr_not_run",
                "all",
                "blank trigger (non-mirrorable failure); sr_result.json missing",
            )
        return ("sr_not_run", "all", "sr_result.json missing")

    # 5 — no candidate universe (combine not run / produced nothing).
    top20 = _top20_lines(results_dir, sr_model_id)
    if top20 is None:
        return ("lr_readiness_skipped", "all", "SR top-20 missing (combine not run)")
    if not top20:
        return ("lr_readiness_skipped", "all", "SR top-20 blank")

    # 6/7 — flexfl-slot exclusions (baselines still evaluate).
    if not _has_lr_result(results_dir):
        dotted_faults = {sig.replace("$", ".") for sig in fault_sigs}
        if not (dotted_faults & set(top20)):
            return (
                "lr_measurement_skipped",
                "flexfl",
                "fault not in SR top-20; baselines evaluated, flexfl slots empty",
            )
        return ("lr_not_run", "flexfl", "LR not run for any config; baselines evaluated")

    return None


def write_exclusions_report(
    *,
    dataset: str = "bugsinpy",
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
    out_path: Path | None = None,
) -> Path:
    """Classify the full corpus and (re)write ``_evaluation/exclusions.csv``.

    Enumerates every (project, bug) via the benchmark adapter (offline for
    BugsInPy: the ``bugsinpy-index.csv`` host copy or the
    ``results/BIP/_meta/`` snapshot), classifies each, and fully rewrites
    the report (idempotent). Returns the written path.
    """
    adapter = get_benchmark_adapter(dataset)
    audit = _read_audit(_audit_csv_path(dataset))
    target = out_path or (get_cross_bug_dir(dataset) / EXCLUSIONS_FILENAME)

    records: list[tuple[str, int, str, str, str]] = []
    for project in adapter.list_projects():
        for bug_id in adapter.list_cases(project):
            results_dir = get_results_bug_dir(project, bug_id, dataset=dataset)
            audit_row = audit.get((project, str(bug_id)))
            verdict = classify_exclusion(results_dir, audit_row, sr_model_id=sr_model_id)
            if verdict is not None:
                bucket, scope, detail = verdict
                records.append((project, bug_id, bucket, scope, detail))

    records.sort(key=lambda r: (r[0], r[1]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(EXCLUSION_HEADERS)
        for project, bug_id, bucket, scope, detail in records:
            writer.writerow([project, str(bug_id), bucket, scope, detail])
    logger.info("wrote %s (%d excluded bugs)", target, len(records))
    return target
