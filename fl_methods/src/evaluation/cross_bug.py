"""Cross-bug evaluation outputs: long form + summary aggregates.

Two CSV families per evaluation slot (baselines / baselines_first / flexfl /
flexfl_first):

* ``evaluation/<Benchmark>/<slot>.csv`` — long form.
  Columns: ``Project,BugId,Method,FR,AR,Top1..Top5,WE``. One row per
  (Project, BugId, Method). Idempotent on that key: re-running a bug
  replaces all rows matching ``(Project, BugId)`` in place.
* ``evaluation/<Benchmark>/<slot>_summary.csv`` — aggregated.
  Columns: ``Method,Bugs,MFR,MAR,Top1Rate..Top5Rate,MeanWE``. Rows whose
  ``FR`` is blank are dropped from *all* aggregates (those bugs had no
  faulty method in their universe — they aren't a method failure to count).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.common.config import get_evaluation_root
from src.evaluation.per_bug import Metrics

logger = logging.getLogger(__name__)

LONG_HEADERS = (
    "Project",
    "BugId",
    "Method",
    "FR",
    "AR",
    "Top1",
    "Top2",
    "Top3",
    "Top4",
    "Top5",
    "WE",
    "InputTokens",
    "CachedTokens",
    "OutputTokens",
    "CostUSD",
)

SUMMARY_HEADERS = (
    "Method",
    "Bugs",
    "MFR",
    "MAR",
    "Top1Rate",
    "Top2Rate",
    "Top3Rate",
    "Top4Rate",
    "Top5Rate",
    "MeanWE",
    "TotalInputTokens",
    "TotalCachedTokens",
    "TotalOutputTokens",
    "TotalCostUSD",
    "MeanCostUSD",
)

SLOTS = ("baselines", "baselines_first", "flexfl", "flexfl_first")


def get_cross_bug_dir(dataset: str = "defects4j") -> Path:
    """Return ``evaluation/<Benchmark>/`` (creates if absent)."""
    out = get_evaluation_root(dataset)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _metrics_to_long_row(project: str, bug_id: str | int, m: Metrics) -> list[str]:
    def _cell(v: object) -> str:
        return "" if v is None else str(v)

    cost = "" if m.CostUSD is None else f"{m.CostUSD:.6f}"
    return [
        project,
        str(bug_id),
        m.Method,
        _cell(m.FR),
        _cell(m.AR),
        str(m.Top1),
        str(m.Top2),
        str(m.Top3),
        str(m.Top4),
        str(m.Top5),
        _cell(m.WE),
        str(m.InputTokens),
        str(m.CachedTokens),
        str(m.OutputTokens),
        cost,
    ]


def _read_long_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return []
        for row in reader:
            if row:
                rows.append(row)
    return rows


def _write_long_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(LONG_HEADERS)
        for row in rows:
            writer.writerow(row)


def append_to_long_csv(
    out_csv: Path,
    project: str,
    bug_id: str | int,
    metrics: list[Metrics],
) -> None:
    """Replace all rows for ``(project, bug_id)`` with ``metrics``.

    Idempotent: re-running a bug replaces its prior rows **in place** (at
    the position of its first prior row) rather than appending duplicates,
    so an unchanged re-run leaves the file byte-identical. New bugs append.
    """
    existing = _read_long_rows(out_csv)
    bug_id_str = str(bug_id)

    def _is_bug_row(row: list[str]) -> bool:
        return len(row) >= 2 and row[0] == project and row[1] == bug_id_str

    new_rows = [_metrics_to_long_row(project, bug_id, m) for m in metrics]
    insert_at = next((i for i, row in enumerate(existing) if _is_bug_row(row)), len(existing))
    keep_before = [row for row in existing[:insert_at]]
    keep_after = [row for row in existing[insert_at:] if not _is_bug_row(row)]
    _write_long_rows(out_csv, keep_before + new_rows + keep_after)


def _parse_float_cell(cell: str) -> float | None:
    cell = cell.strip()
    if not cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _parse_int_cell(cell: str) -> int:
    cell = cell.strip()
    if not cell:
        return 0
    try:
        return int(cell)
    except ValueError:
        return 0


def write_summary_csv(long_csv: Path, summary_csv: Path) -> None:
    """Aggregate a long CSV into a per-method summary.

    Aggregation rules (per Method group):
      * Rows with blank ``FR`` are dropped from every aggregate — the bug had
        no faulty method in its universe, so it shouldn't penalise the method.
      * ``Bugs`` = count of surviving rows.
      * ``MFR``, ``MAR``, ``MeanWE`` = arithmetic mean over surviving rows
        (skipping blank ``WE`` cells individually).
      * ``TopKRate`` = mean of the 0/1 TopK column over surviving rows.
      * ``Total*Tokens`` / ``TotalCostUSD`` = sum over surviving rows.
      * ``MeanCostUSD`` = arithmetic mean of non-blank cost cells.
    """
    rows = _read_long_rows(long_csv)
    grouped: dict[str, list[list[str]]] = {}
    for row in rows:
        if len(row) < len(LONG_HEADERS):
            continue
        method = row[2]
        grouped.setdefault(method, []).append(row)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    empty_token_tail = ["0", "0", "0", "0.000000", ""]
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_HEADERS)
        for method in sorted(grouped):
            method_rows = grouped[method]
            fr_values: list[float] = []
            ar_values: list[float] = []
            we_values: list[float] = []
            topk_values: list[list[int]] = []
            input_tokens = 0
            cached_tokens = 0
            output_tokens = 0
            cost_values: list[float] = []
            for row in method_rows:
                fr = _parse_float_cell(row[3])
                if fr is None:
                    continue
                fr_values.append(fr)
                ar = _parse_float_cell(row[4])
                if ar is not None:
                    ar_values.append(ar)
                we = _parse_float_cell(row[10])
                if we is not None:
                    we_values.append(we)
                topk_values.append([_parse_int_cell(row[5 + k]) for k in range(5)])
                input_tokens += _parse_int_cell(row[11])
                cached_tokens += _parse_int_cell(row[12])
                output_tokens += _parse_int_cell(row[13])
                cost = _parse_float_cell(row[14])
                if cost is not None:
                    cost_values.append(cost)

            bugs = len(fr_values)
            if bugs == 0:
                writer.writerow([method, 0, "", "", "", "", "", "", "", "", *empty_token_tail])
                continue

            def _mean(xs: list[float]) -> str:
                return f"{sum(xs) / len(xs):.6f}" if xs else ""

            mfr = _mean(fr_values)
            mar = _mean(ar_values)
            mean_we = _mean(we_values)
            topk_rate = ["" for _ in range(5)]
            if topk_values:
                for k in range(5):
                    col = [r[k] for r in topk_values]
                    topk_rate[k] = f"{sum(col) / len(col):.6f}"
            total_cost = sum(cost_values)
            mean_cost = _mean(cost_values)
            writer.writerow(
                [
                    method,
                    bugs,
                    mfr,
                    mar,
                    *topk_rate,
                    mean_we,
                    str(input_tokens),
                    str(cached_tokens),
                    str(output_tokens),
                    f"{total_cost:.6f}",
                    mean_cost,
                ]
            )


def slot_paths(dataset: str = "defects4j") -> dict[str, tuple[Path, Path]]:
    """Return ``{slot: (long_csv, summary_csv)}`` for all four evaluation slots."""
    base = get_cross_bug_dir(dataset)
    return {slot: (base / f"{slot}.csv", base / f"{slot}_summary.csv") for slot in SLOTS}
