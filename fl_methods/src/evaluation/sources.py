"""Per-FL-method score-map loaders.

Each loader turns a method's existing on-disk output into a
``dict[MethodEntity, float]`` keyed by canonical corpus identity. Statement-
level methods (Ochiai, SBIR) are read from the **method-level** ranking CSVs
already produced by :func:`src.common.rankings.generate_method_ranking` —
this module never re-aggregates statement scores.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.common.config import (
    FLEXFL_LR_SUBDIR,
)
from src.common.method_entity import MethodEntity
from src.evaluation.pricing import compute_response_cost

logger = logging.getLogger(__name__)

# Default Agent4SR model id whose top-20 list defines the candidate universe.
DEFAULT_SR_MODEL_ID = "llama3.1_8b"


@dataclass(frozen=True)
class Usage:
    """Aggregated LLM token usage and cost for one ``lr_result.json``."""

    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float

    @classmethod
    def zero(cls) -> Usage:
        return cls(0, 0, 0, 0.0)


def _entity_indexes(
    entities: list[MethodEntity],
) -> tuple[dict[str, MethodEntity], dict[str, MethodEntity]]:
    """Return ``(corpus_id → entity, dotted_corpus_id → entity)`` lookup pairs.

    Mirrors :mod:`src.common.rankings` so Agent4LR's dotted ``top5`` values
    (no ``$`` separator) can still resolve to a canonical entity.
    """
    entity_index: dict[str, MethodEntity] = {}
    for e in entities:
        entity_index.setdefault(e.corpus_id, e)
    dotted_index = {e.corpus_id.replace("$", "."): e for e in entities}
    return entity_index, dotted_index


def load_baseline_scores(
    rankings_dir: Path,
    method: str,
    entities: list[MethodEntity],
) -> dict[MethodEntity, float]:
    """Load Ochiai / SBIR / BoostN method-level scores.

    Reads ``<rankings_dir>/<method>.csv`` (the ``;``-delimited CSV with
    columns ``rank;signature;path;startLine;endLine;score``). The rankings
    dir is ``FlexFL/SR/rankings/`` under a processed bug dir, or
    ``rankings/`` under a results bug dir. Returns an empty dict (with a
    warning) if the file is missing or empty.
    """
    if method not in {"ochiai", "sbir", "boostn"}:
        raise ValueError(f"unknown baseline method: {method!r}")
    csv_path = rankings_dir / f"{method}.csv"
    if not csv_path.exists():
        logger.warning("baseline ranking missing: %s", csv_path)
        return {}

    entity_index, _ = _entity_indexes(entities)
    scores: dict[MethodEntity, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            sig = (row.get("signature") or "").strip()
            score_raw = (row.get("score") or "").strip()
            if not sig:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue
            entity = entity_index.get(sig)
            if entity is None:
                continue
            prev = scores.get(entity)
            scores[entity] = score if prev is None else max(prev, score)
    return scores


def load_agent4lr_scores(
    lr_result_path: Path,
    entities: list[MethodEntity],
) -> dict[MethodEntity, float]:
    """Parse an ``lr_result.json`` and assign pseudo-scores 5..1 to its top5.

    The Agent4LR runner emits ``top5`` as dotted FQNs (no ``$`` separator);
    we reinsert ``$`` via the dotted entity index to recover the canonical
    corpus identity. Unmatched entries are logged at warning level and
    omitted from the score map (they end up as "unranked" in the universe).
    """
    if not lr_result_path.exists():
        logger.warning("lr_result.json missing: %s", lr_result_path)
        return {}
    try:
        data = json.loads(lr_result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("lr_result.json malformed: %s", lr_result_path)
        return {}

    top5 = data.get("top5") or []
    if not isinstance(top5, list):
        logger.warning("lr_result.json top5 not a list: %s", lr_result_path)
        return {}

    # Sanity-check 1-based indices when present.
    indices = data.get("top5_indices")
    if isinstance(indices, list) and indices and isinstance(indices[0], int) and indices[0] < 1:
        logger.warning(
            "lr_result.json top5_indices[0]=%d is < 1; expected 1-based",
            indices[0],
        )

    return top5_to_scores(top5, entities, context=lr_result_path.parent.name)


def top5_to_scores(
    top5: Sequence[object],
    entities: list[MethodEntity],
    *,
    context: str = "",
) -> dict[MethodEntity, float]:
    """Map an Agent4LR ``top5`` list to pseudo-scores 5..1 on entities.

    Shared by the processed-tree (``lr_result.json``) and results-tree
    (``lr.json``) readers so both resolve identically. ``context`` labels
    warning messages (typically the config name).
    """
    entity_index, dotted_index = _entity_indexes(entities)
    scores: dict[MethodEntity, float] = {}
    for i, fqn in enumerate(top5):
        if not isinstance(fqn, str):
            continue
        raw = fqn.strip()
        if not raw:
            continue
        matched = entity_index.get(raw) or dotted_index.get(raw)
        if matched is None:
            logger.warning(
                "Agent4LR top5[%d]=%r not in method_signatures.csv (%s)",
                i,
                raw,
                context,
            )
            continue
        # Pseudo-score descending so assign_average_ranks yields ranks 1..5.
        score = float(len(top5) - i)
        prev = scores.get(matched)
        scores[matched] = score if prev is None else max(prev, score)
    return scores


def load_candidate_universe(
    top20_path: Path,
    entities: list[MethodEntity],
) -> set[MethodEntity]:
    """Universe = the SR top-20 candidate list fed into Agent4LR.

    Reads a ``top20/<sr_model_id>.txt`` file (one dotted FQN per line,
    written by :func:`src.common.rankings.generate_top20` — found under
    ``FlexFL/SR/rankings/`` in a processed bug dir or ``rankings/`` in a
    results bug dir) and maps each line to a canonical
    :class:`MethodEntity` via the same exact / dotted indexes used by
    :func:`load_agent4lr_scores`. Lines that don't resolve are skipped
    with a warning. Missing file returns an empty set (also logged).

    The all-mode and first-mode evaluations share this single universe —
    they differ only in the faulty-method set (``faults.csv`` vs.
    ``faults_first.csv``). ``evaluate_method`` handles
    ``faulty ∩ universe == ∅`` by emitting blank FR/AR cells, which is
    the right behaviour when the bug's first-fault set is empty (e.g.
    Jsoup-15).
    """
    if not top20_path.exists():
        logger.warning("SR top-20 candidate file missing: %s", top20_path)
        return set()

    entity_index, dotted_index = _entity_indexes(entities)
    universe: set[MethodEntity] = set()
    for i, line in enumerate(top20_path.read_text(encoding="utf-8").splitlines()):
        raw = line.strip()
        if not raw:
            continue
        matched = entity_index.get(raw) or dotted_index.get(raw)
        if matched is None:
            logger.warning(
                "SR top-20 line %d=%r not in method_signatures.csv (%s)",
                i,
                raw,
                top20_path.name,
            )
            continue
        universe.add(matched)
    return universe


def discover_agent4lr_configs(
    processed_dir: Path,
    *,
    tracker: dict | None = None,
) -> list[tuple[str, Path]]:
    """List ``(config_name, lr_result_path)`` for every finished LR run.

    Globs ``FlexFL/LR/Agent4LR/*/lr_result.json``. When a tracker is given,
    cross-checks against its ``lr`` keys and emits a debug log for any
    on-disk config that's missing from the tracker — both are kept; tracker
    omission is informational only.
    """
    lr_root = processed_dir / FLEXFL_LR_SUBDIR / "Agent4LR"
    if not lr_root.is_dir():
        return []

    results: list[tuple[str, Path]] = []
    for config_dir in sorted(lr_root.iterdir()):
        if not config_dir.is_dir():
            continue
        lr_result = config_dir / "lr_result.json"
        if not lr_result.exists():
            continue
        results.append((config_dir.name, lr_result))

    if tracker is not None:
        on_disk = {name for name, _ in results}
        tracked = set(tracker.get("lr", {}).keys())
        for orphan in sorted(on_disk - tracked):
            logger.debug("Agent4LR config %s present on disk but not in tracker", orphan)
    return results


def load_agent4lr_usage(lr_result_path: Path) -> Usage:
    """Sum LLM token usage and cost across ``response_dumps`` of one LR run.

    Each entry of ``response_dumps`` carries its own ``usage`` block plus a
    ``model`` field (which may include a dated suffix, e.g.
    ``gpt-5-mini-2025-08-07``). Cost is computed per response via
    :func:`src.evaluation.pricing.compute_response_cost`, so a single
    config that mixes models is priced correctly.

    Missing or malformed files return :meth:`Usage.zero`.
    """
    if not lr_result_path.exists():
        return Usage.zero()
    try:
        data = json.loads(lr_result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("lr_result.json malformed for usage: %s", lr_result_path)
        return Usage.zero()

    dumps = data.get("response_dumps") or []
    if not isinstance(dumps, list):
        return Usage.zero()

    total_input = 0
    total_cached = 0
    total_output = 0
    total_cost = 0.0
    for dump in dumps:
        if not isinstance(dump, dict):
            continue
        usage = dump.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        cached = 0
        details = usage.get("input_tokens_details") or {}
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        model = str(dump.get("model") or "")
        total_input += in_tok
        total_cached += cached
        total_output += out_tok
        total_cost += compute_response_cost(in_tok, cached, out_tok, model)
    return Usage(
        input_tokens=total_input,
        cached_tokens=total_cached,
        output_tokens=total_output,
        cost_usd=total_cost,
    )


def check_required_configs(
    processed_dir: Path,
    required: list[str],
) -> tuple[bool, list[str]]:
    """Validate that every named Agent4LR config produced a usable ``top5``.

    Returns ``(ok, problems)`` where ``problems`` lists configs whose
    ``lr_result.json`` is missing, malformed, or contains no usable
    candidate. A "usable" ``top5`` is a list with **at least one
    non-empty string entry** — short (< 5) but populated top5 lists are
    accepted because ``evaluate_method`` only needs ≥1 ranked candidate
    to produce a non-blank FR (the universe is the SR top-20, not the
    agent's output). ``ok`` is True iff ``problems`` is empty.
    """
    lr_root = processed_dir / FLEXFL_LR_SUBDIR / "Agent4LR"
    problems: list[str] = []
    for cfg in required:
        path = lr_root / cfg / "lr_result.json"
        if not path.exists():
            problems.append(f"{cfg} (missing)")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"{cfg} (malformed)")
            continue
        top5 = data.get("top5")
        if not isinstance(top5, list) or not any(isinstance(s, str) and s.strip() for s in top5):
            problems.append(f"{cfg} (invalid top5)")
    return (not problems, problems)
