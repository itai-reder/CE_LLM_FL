"""Per-bug evaluation orchestration.

Builds the four per-bug ``Evaluation/*.csv`` files for one bug (any
supported benchmark — Defects4J or BugsInPy).

Layout written:

    <processed>/Evaluation/baselines.csv
    <processed>/Evaluation/baselines_first.csv
    <processed>/Evaluation/flexfl.csv
    <processed>/Evaluation/flexfl_first.csv

Each row carries a single FL method's metrics for the bug:
``Method,FR,AR,Top1,Top2,Top3,Top4,Top5,WE``. Empty ``None`` cells are
written blank.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import (
    FLEXFL_SR_SUBDIR,
    get_processed_dir,
    get_results_bug_dir,
)
from src.common.method_entity import MethodEntity, load_method_entities
from src.evaluation.metrics import (
    first_rank,
    mean_rank,
    top_k,
    wasted_effort,
)
from src.evaluation.ranking import score_universe
from src.evaluation.sources import (
    DEFAULT_SR_MODEL_ID,
    Usage,
    discover_agent4lr_configs,
    load_agent4lr_scores,
    load_agent4lr_usage,
    load_baseline_scores,
    load_candidate_universe,
    top5_to_scores,
)

logger = logging.getLogger(__name__)

EVALUATION_SUBDIR = "Evaluation"  # per-bug CSVs inside a processed bug dir
RESULTS_EVALUATION_SUBDIR = "evaluation"  # per-bug CSVs inside a results bug dir

BASELINE_METHODS = ("Ochiai", "SBIR", "BoostN")
_BASELINE_KEY = {"Ochiai": "ochiai", "SBIR": "sbir", "BoostN": "boostn"}

PER_BUG_HEADERS = (
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


@dataclass(frozen=True)
class Metrics:
    """Evaluation metrics for one FL method on one bug.

    Token / cost fields are populated only for Agent4LR rows (the LR
    runner is the source of LLM-billable tokens). Baselines keep the
    zero/None defaults so their cells render as ``0,0,0,`` (blank cost).
    """

    Method: str
    FR: float | None
    AR: float | None
    Top1: int
    Top2: int
    Top3: int
    Top4: int
    Top5: int
    WE: float | None
    InputTokens: int = 0
    CachedTokens: int = 0
    OutputTokens: int = 0
    CostUSD: float | None = None

    def to_row(self) -> list[str]:
        def _cell(value: object) -> str:
            return "" if value is None else str(value)

        cost = "" if self.CostUSD is None else f"{self.CostUSD:.6f}"
        return [
            self.Method,
            _cell(self.FR),
            _cell(self.AR),
            str(self.Top1),
            str(self.Top2),
            str(self.Top3),
            str(self.Top4),
            str(self.Top5),
            _cell(self.WE),
            str(self.InputTokens),
            str(self.CachedTokens),
            str(self.OutputTokens),
            cost,
        ]


def evaluate_method(
    method_name: str,
    scored: dict[MethodEntity, float],
    universe: set[MethodEntity],
    faulty: set[MethodEntity],
    usage: Usage | None = None,
) -> Metrics:
    """Compute ``Metrics`` for one method using already-loaded score/universe sets.

    The universe defines both the rank denominator and the candidate set for
    rank assignment. ``faulty ∩ universe`` is what we actually score against.

    ``usage`` carries pre-aggregated LLM token / cost data (only meaningful
    for Agent4LR rows). When omitted, the resulting ``Metrics`` keeps the
    zero-token, blank-cost defaults — appropriate for baseline FL methods.
    """
    token_fields: dict[str, Any] = {}
    if usage is not None:
        token_fields = {
            "InputTokens": usage.input_tokens,
            "CachedTokens": usage.cached_tokens,
            "OutputTokens": usage.output_tokens,
            "CostUSD": usage.cost_usd,
        }

    universe_size = len(universe)
    if universe_size == 0:
        return Metrics(method_name, None, None, 0, 0, 0, 0, 0, None, **token_fields)

    ranks = score_universe(scored, universe, tiebreak=lambda e: e.corpus_id)
    faulty_in_universe = faulty & universe
    fr = first_rank(ranks, faulty_in_universe)
    ar = mean_rank(ranks, faulty_in_universe)
    we = wasted_effort(ranks, faulty_in_universe, universe_size)
    return Metrics(
        Method=method_name,
        FR=fr,
        AR=ar,
        Top1=top_k(fr, 1),
        Top2=top_k(fr, 2),
        Top3=top_k(fr, 3),
        Top4=top_k(fr, 4),
        Top5=top_k(fr, 5),
        WE=we,
        **token_fields,
    )


def _load_faulty_ids(csv_path: Path) -> set[str]:
    """Read the ``signature`` column from a faults.csv-shaped file."""
    if not csv_path.exists():
        return set()
    ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sig = (row.get("signature") or "").strip()
            if sig:
                ids.add(sig)
    return ids


def _ids_to_entities(ids: set[str], entities: list[MethodEntity]) -> set[MethodEntity]:
    index = {e.corpus_id: e for e in entities}
    return {index[i] for i in ids if i in index}


def _load_tracker(processed_dir: Path) -> dict | None:
    """Return parsed tracker.json or None if missing/malformed."""
    path = processed_dir / "tracker.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class BugInputs:
    """Everything :func:`evaluate_bug_inputs` needs for one bug, pre-loaded.

    Produced by :func:`load_bug_inputs_from_processed` (full processed tree)
    or :func:`load_bug_inputs_from_results` (slim committed results tree);
    both must yield identical metrics for the same bug — the results builder
    asserts exactly that.
    """

    entities: list[MethodEntity]
    universe: set[MethodEntity]
    faulty_all: set[MethodEntity]
    faulty_first: set[MethodEntity]
    baseline_scores: dict[str, dict[MethodEntity, float]]
    lr_scores: dict[str, dict[MethodEntity, float]]
    lr_usage: dict[str, Usage]


def _load_ground_truth(
    base_dir: Path, entities: list[MethodEntity]
) -> tuple[set[MethodEntity], set[MethodEntity]]:
    faulty_all = _ids_to_entities(_load_faulty_ids(base_dir / "faults.csv"), entities)
    faulty_first = _ids_to_entities(_load_faulty_ids(base_dir / "faults_first.csv"), entities)
    return faulty_all, faulty_first


def load_bug_inputs_from_processed(
    processed_dir: Path,
    *,
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
) -> BugInputs:
    """Load evaluation inputs from a full per-bug processed directory."""
    entities = load_method_entities(processed_dir)
    rankings_dir = processed_dir / FLEXFL_SR_SUBDIR / "rankings"

    # Universe — the SR top-20 candidate list fed into Agent4LR. Both panels
    # (all-faults and first-fault-only) share this single candidate set; they
    # differ only in the faulty-method ground truth.
    universe = load_candidate_universe(rankings_dir / "top20" / f"{sr_model_id}.txt", entities)
    faulty_all, faulty_first = _load_ground_truth(processed_dir, entities)

    baseline_scores: dict[str, dict[MethodEntity, float]] = {}
    for label in BASELINE_METHODS:
        baseline_scores[label] = load_baseline_scores(rankings_dir, _BASELINE_KEY[label], entities)

    # Agent4LR configs (label = "Agent4LR-<config_name>")
    tracker = _load_tracker(processed_dir)
    lr_scores: dict[str, dict[MethodEntity, float]] = {}
    lr_usage: dict[str, Usage] = {}
    for config_name, lr_result_path in discover_agent4lr_configs(processed_dir, tracker=tracker):
        label = f"Agent4LR-{config_name}"
        lr_scores[label] = load_agent4lr_scores(lr_result_path, entities)
        lr_usage[label] = load_agent4lr_usage(lr_result_path)

    return BugInputs(
        entities=entities,
        universe=universe,
        faulty_all=faulty_all,
        faulty_first=faulty_first,
        baseline_scores=baseline_scores,
        lr_scores=lr_scores,
        lr_usage=lr_usage,
    )


def load_bug_inputs_from_results(
    results_dir: Path,
    *,
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
) -> BugInputs:
    """Load evaluation inputs from a slim per-bug results directory.

    Mirrors :func:`load_bug_inputs_from_processed` over the committed
    ``results/<Benchmark>/<Project>/<BugId>/`` layout: the same
    ``method_signatures.csv`` / ``faults*.csv`` schemas at the dir root, the
    baseline rankings + SR top-20 under ``rankings/``, and all Agent4LR
    configs consolidated into one ``lr.json``.
    """
    from src.results.read import load_lr_json

    entities = load_method_entities(results_dir)
    rankings_dir = results_dir / "rankings"

    universe = load_candidate_universe(rankings_dir / "top20" / f"{sr_model_id}.txt", entities)
    faulty_all, faulty_first = _load_ground_truth(results_dir, entities)

    baseline_scores: dict[str, dict[MethodEntity, float]] = {}
    for label in BASELINE_METHODS:
        baseline_scores[label] = load_baseline_scores(rankings_dir, _BASELINE_KEY[label], entities)

    lr_scores: dict[str, dict[MethodEntity, float]] = {}
    lr_usage: dict[str, Usage] = {}
    for config_name, entry in load_lr_json(results_dir).items():
        label = f"Agent4LR-{config_name}"
        lr_scores[label] = top5_to_scores(entry.top5, entities, context=config_name)
        lr_usage[label] = entry.usage
    return BugInputs(
        entities=entities,
        universe=universe,
        faulty_all=faulty_all,
        faulty_first=faulty_first,
        baseline_scores=baseline_scores,
        lr_scores=lr_scores,
        lr_usage=lr_usage,
    )


def evaluate_bug_inputs(inputs: BugInputs) -> dict[str, list[Metrics]]:
    """Compute the four metric lists from pre-loaded inputs (pure)."""
    results: dict[str, list[Metrics]] = {
        "baselines": [],
        "baselines_first": [],
        "flexfl": [],
        "flexfl_first": [],
    }

    if inputs.universe:
        for label in BASELINE_METHODS:
            scores = inputs.baseline_scores[label]
            results["baselines"].append(
                evaluate_method(label, scores, inputs.universe, inputs.faulty_all)
            )
            results["baselines_first"].append(
                evaluate_method(label, scores, inputs.universe, inputs.faulty_first)
            )
        for label in sorted(inputs.lr_scores):
            scores = inputs.lr_scores[label]
            usage = inputs.lr_usage[label]
            results["flexfl"].append(
                evaluate_method(label, scores, inputs.universe, inputs.faulty_all, usage=usage)
            )
            results["flexfl_first"].append(
                evaluate_method(label, scores, inputs.universe, inputs.faulty_first, usage=usage)
            )

    return results


def evaluate_bug(
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
) -> dict[str, list[Metrics]]:
    """Run all four evaluations for one bug from its processed directory.

    Returns a dict with keys ``baselines``, ``baselines_first``, ``flexfl``,
    ``flexfl_first``. A list may be empty when its universe or ground-truth
    set is empty for the bug (caller writes header-only CSVs in that case).
    """
    processed_dir = get_processed_dir(project, bug_id, dataset=dataset)
    inputs = load_bug_inputs_from_processed(processed_dir, sr_model_id=sr_model_id)
    return evaluate_bug_inputs(inputs)


def evaluate_bug_from_results(
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
    sr_model_id: str = DEFAULT_SR_MODEL_ID,
) -> dict[str, list[Metrics]]:
    """Run all four evaluations for one bug from its slim results directory."""
    results_dir = get_results_bug_dir(project, bug_id, dataset=dataset)
    inputs = load_bug_inputs_from_results(results_dir, sr_model_id=sr_model_id)
    return evaluate_bug_inputs(inputs)


def write_per_bug_csvs(
    base_dir: Path,
    results: dict[str, list[Metrics]],
    *,
    subdir: str = EVALUATION_SUBDIR,
) -> dict[str, Path]:
    """Write the four per-bug metric CSVs under ``<base_dir>/<subdir>/``.

    ``subdir`` is ``Evaluation`` in a processed bug dir (historical layout)
    and ``evaluation`` (``RESULTS_EVALUATION_SUBDIR``) in a results bug dir.
    Returns name → path.
    """
    eval_dir = base_dir / subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("baselines", "baselines_first", "flexfl", "flexfl_first"):
        csv_path = eval_dir / f"{name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(PER_BUG_HEADERS)
            for metrics in results.get(name, []):
                writer.writerow(metrics.to_row())
        paths[name] = csv_path
    return paths
