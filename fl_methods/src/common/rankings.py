"""Generate standardised method-level ranking CSVs and combined top-N files.

This module ties together the FL method outputs (Ochiai, SBIR, BoostN) and
Agent4SR results into a unified ranking format under
``<processed>/FlexFL/SR/rankings/``.

All behavior-affecting joins key on the canonical corpus identity
(:class:`src.common.method_entity.MethodEntity`).  The CSVs emit corpus
identity in the ``signature`` column.

Outputs:
  - ``ochiai.csv``, ``sbir.csv``, ``boostn.csv`` — per-method ranking CSVs
  - ``top15.txt`` / ``top15.csv`` — combined SBIR + Ochiai + BoostN top-5
  - ``top20/<model>.txt`` / ``top20/<model>.csv`` — top15 + Agent4SR top-5
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.common.config import (
    BOOSTN_CSV,
    BOOSTN_SUBDIR,
    OCHIAI_SUBDIR,
    SBFL_STMT_SUSPS,
    SBIR_STMT_SUSPS,
    SBIR_SUBDIR,
    _model_slug,
    get_processed_dir,
    get_rankings_dir,
    get_sr_model_dir,
)
from src.common.flexfl_compat import (
    read_method_scores_csv,
    read_statement_scores_csv,
)
from src.common.method_entity import (
    MethodEntity,
    aggregate_statement_scores_to_entities,
    build_entity_line_index,
    load_method_entities,
    rank_entities,
    write_entity_ranking_csv,
)

logger = logging.getLogger(__name__)


def _entity_index_by_corpus_id(entities: list[MethodEntity]) -> dict[str, MethodEntity]:
    """Index entities by ``corpus_id`` for direct lookup.

    On overload collisions the first entity wins; ranking-side scores are
    already collapsed via the ``MethodEntity`` hash.
    """
    index: dict[str, MethodEntity] = {}
    for e in entities:
        index.setdefault(e.corpus_id, e)
    return index


def _build_param_stripped_index(
    entities: list[MethodEntity],
) -> dict[str, list[MethodEntity]]:
    """Index entities by ``pkg.Class.method`` (parameters stripped, dotted).

    Built for the third-tier fuzzy fallback in
    :func:`_fuzzy_match_corpus_id` — handles agent-emitted IDs whose
    parameter normalization (e.g. ``List<String>`` vs ``List``,
    inner-class ``.`` vs ``$``) differs from the corpus.
    """
    index: dict[str, list[MethodEntity]] = {}
    for e in entities:
        key = e.corpus_id.split("(", 1)[0].replace("$", ".")
        index.setdefault(key, []).append(e)
    return index


def _fuzzy_match_corpus_id(
    raw: str,
    *,
    entity_index: dict[str, MethodEntity],
    dotted_index: dict[str, MethodEntity],
    stripped_index: dict[str, list[MethodEntity]],
) -> MethodEntity | None:
    """Three-tier resolution for agent-emitted corpus identities.

    1. Exact ``corpus_id`` match (``pkg$Class.method(P)``).
    2. Dotted-form match (``pkg.Class.method(P)``) — agents emit this.
    3. Param-stripped match (``pkg.Class.method``) — last resort when the
       agent's parameter normalization disagrees with the corpus.  On
       overload collisions, picks deterministically: shortest
       ``corpus_id`` (i.e. fewest / shortest params), then lexicographic.

    Returns ``None`` when no tier matches.
    """
    exact = entity_index.get(raw)
    if exact is not None:
        return exact
    dotted = dotted_index.get(raw)
    if dotted is not None:
        return dotted

    stripped_key = raw.split("(", 1)[0].replace("$", ".")
    candidates = stripped_index.get(stripped_key)
    if not candidates:
        return None
    chosen = min(candidates, key=lambda e: (len(e.corpus_id), e.corpus_id))
    logger.debug(
        "Fuzzy corpus_id match: %r -> %r (%d candidate(s))",
        raw,
        chosen.corpus_id,
        len(candidates),
    )
    return chosen


def generate_method_ranking(
    method: str,
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
) -> list[tuple[MethodEntity, float, int]] | None:
    """Generate a method-level ranking CSV for a single FL method.

    Parameters
    ----------
    method:
        One of ``"ochiai"``, ``"sbir"``, ``"boostn"``.
    project, bug_id:
        Bug identifiers.

    Returns
    -------
    list or None
        Ranked list of ``(MethodEntity, score, rank)`` tuples, or ``None`` if
        required input files are missing.
    """
    processed_dir = get_processed_dir(project, bug_id, dataset=dataset)
    rankings_dir = get_rankings_dir(project, bug_id, dataset=dataset)

    try:
        entities = load_method_entities(processed_dir)
    except FileNotFoundError:
        logger.warning(
            "method_signatures.csv missing for %s-%s; skipping %s ranking",
            project,
            bug_id,
            method,
        )
        return None

    if method in ("ochiai", "sbir"):
        subdir = OCHIAI_SUBDIR if method == "ochiai" else SBIR_SUBDIR
        stmt_csv_name = SBFL_STMT_SUSPS if method == "ochiai" else SBIR_STMT_SUSPS
        stmt_csv = processed_dir / subdir / stmt_csv_name
        if not stmt_csv.exists():
            logger.warning("%s not found for %s-%s", stmt_csv, project, bug_id)
            return None

        stmt_scores = read_statement_scores_csv(stmt_csv)
        line_index = build_entity_line_index(entities)
        method_scores = aggregate_statement_scores_to_entities(stmt_scores, line_index)

    elif method == "boostn":
        boostn_csv = processed_dir / BOOSTN_SUBDIR / BOOSTN_CSV
        if not boostn_csv.exists():
            logger.warning("BoostN output not found for %s-%s", project, bug_id)
            return None

        boostn_scores = read_method_scores_csv(boostn_csv)
        entity_index = _entity_index_by_corpus_id(entities)
        method_scores = {}
        unmatched = 0
        for corpus_id, score in boostn_scores.items():
            entity = entity_index.get(corpus_id)
            if entity is None:
                unmatched += 1
                continue
            prev = method_scores.get(entity)
            method_scores[entity] = score if prev is None else max(prev, score)
        if unmatched:
            logger.warning(
                "BoostN: %d/%d corpus IDs not present in method_signatures.csv",
                unmatched,
                len(boostn_scores),
            )

    else:
        raise ValueError(f"Unknown FL method: {method!r}")

    ranked = rank_entities(method_scores)

    output_csv = rankings_dir / f"{method}.csv"
    write_entity_ranking_csv(output_csv, ranked)
    logger.info("Wrote %s ranking: %d methods -> %s", method, len(ranked), output_csv)

    return ranked


def generate_all_rankings(
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
) -> dict[str, list[tuple[MethodEntity, float, int]]]:
    """Generate ranking CSVs for all available FL methods.

    Returns a dict mapping method name to its ranked list.
    """
    results: dict[str, list[tuple[MethodEntity, float, int]]] = {}

    for method in ("ochiai", "sbir", "boostn"):
        ranked = generate_method_ranking(method, project, bug_id, dataset=dataset)
        if ranked is not None:
            results[method] = ranked

    return results


_METHOD_LABELS = {"sbir": "SBIR", "ochiai": "Ochiai", "boostn": "BoostN"}


def _write_rows_csv(output_path: Path, rows: list[tuple[str, MethodEntity]]) -> None:
    """Write rows as ``rank,method,signature,path,startLine,endLine``.

    The ``signature`` column carries the corpus identity.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "method", "signature", "path", "startLine", "endLine"])
        for idx, (method_label, entity) in enumerate(rows, start=1):
            writer.writerow(
                [
                    idx,
                    method_label,
                    entity.corpus_id,
                    entity.path,
                    entity.start_line,
                    entity.end_line,
                ]
            )


def _build_top15_rows(
    rankings: dict[str, list[tuple[MethodEntity, float, int]]],
) -> list[tuple[str, MethodEntity]]:
    """Build the 15-row top15 selection: top-5 of SBIR, Ochiai, BoostN.

    Pads with Ochiai if fewer than 15 unique corpus IDs are available.
    """
    rows: list[tuple[str, MethodEntity]] = []
    seen: set[str] = set()

    for method_name in ("sbir", "ochiai", "boostn"):
        label = _METHOD_LABELS[method_name]
        count = 0
        for entity, _score, _rank in rankings.get(method_name, []):
            if entity.corpus_id not in seen:
                rows.append((label, entity))
                seen.add(entity.corpus_id)
                count += 1
            if count >= 5:
                break

    if len(rows) < 15:
        for entity, _score, _rank in rankings.get("ochiai", []):
            if entity.corpus_id not in seen:
                rows.append(("Ochiai", entity))
                seen.add(entity.corpus_id)
            if len(rows) >= 15:
                break

    return rows


def generate_top15(
    project: str,
    bug_id: str | int,
    rankings: dict[str, list[tuple[MethodEntity, float, int]]] | None = None,
    *,
    dataset: str = "defects4j",
) -> Path:
    """Generate ``top15.txt`` and ``top15.csv`` combining top-5 from SBIR,
    Ochiai, and BoostN.

    If any method is unavailable, falls back to padding from Ochiai.

    Returns the path to the generated TXT file.
    """
    rankings_dir = get_rankings_dir(project, bug_id, dataset=dataset)
    txt_path = rankings_dir / "top15.txt"
    csv_path = rankings_dir / "top15.csv"

    if rankings is None:
        rankings = generate_all_rankings(project, bug_id, dataset=dataset)

    rows = _build_top15_rows(rankings)
    txt_path.write_text("\n".join(entity.corpus_id for _, entity in rows) + "\n", encoding="utf-8")
    _write_rows_csv(csv_path, rows)
    logger.info("Wrote top15: %d signatures -> %s (+ %s)", len(rows), txt_path, csv_path.name)
    return txt_path


def generate_top20(
    project: str,
    bug_id: str | int,
    model: str,
    *,
    dataset: str = "defects4j",
    model_id: str | None = None,
) -> Path:
    """Generate ``top20/<name>.txt`` and ``top20/<name>.csv`` following FlexFL's
    combination rule: ``SBIR[:5] + Ochiai[:5] + BoostN[:5] + Agent4SR[:5]``.

    No deduplication — duplicates across (and within) sources are preserved, so
    the list has exactly 20 lines iff every source contributes 5. Shorter lists
    are written as-is; the Agent4LR gate rejects anything that is not exactly 20.

    ``name`` is ``model_id`` when provided (the tracker-assigned slug, possibly
    with a ``__N`` suffix), otherwise ``_model_slug(model)``.

    Reads ``top5.txt`` (raw corpus IDs from the agent run) and joins each
    against ``method_signatures.csv`` for path/line metadata.
    Returns the path to the generated TXT file.
    """
    rankings_dir = get_rankings_dir(project, bug_id, dataset=dataset)
    name = model_id if model_id is not None else _model_slug(model)
    txt_path = rankings_dir / "top20" / f"{name}.txt"
    csv_path = rankings_dir / "top20" / f"{name}.csv"
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    all_rankings = generate_all_rankings(project, bug_id, dataset=dataset)
    generate_top15(project, bug_id, all_rankings, dataset=dataset)

    rows: list[tuple[str, MethodEntity]] = []
    txt_ids: list[str] = []
    for method_name in ("sbir", "ochiai", "boostn"):
        label = _METHOD_LABELS[method_name]
        for entity, _score, _rank in all_rankings.get(method_name, [])[:5]:
            rows.append((label, entity))
            txt_ids.append(entity.corpus_id)

    try:
        all_entities = load_method_entities(get_processed_dir(project, bug_id, dataset=dataset))
    except FileNotFoundError:
        all_entities = []
    entity_index = _entity_index_by_corpus_id(all_entities)
    # Fallback index for agent-emitted IDs that arrive in the dotted form
    # (no $ separator) — see normalize_method_name in agent4sr/tools.py.
    dotted_index: dict[str, MethodEntity] = {e.corpus_id.replace("$", "."): e for e in all_entities}
    stripped_index = _build_param_stripped_index(all_entities)

    model_dir = get_sr_model_dir(project, bug_id, model, dataset=dataset, model_id=model_id)
    top5_path = model_dir / "top5.txt"

    if top5_path.exists():
        agent_lines = [
            line.strip()
            for line in top5_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for raw in agent_lines[:5]:
            matched = _fuzzy_match_corpus_id(
                raw,
                entity_index=entity_index,
                dotted_index=dotted_index,
                stripped_index=stripped_index,
            )
            txt_ids.append(matched.corpus_id if matched is not None else raw)
            if matched is not None:
                rows.append(("Agent4SR", matched))
            else:
                logger.warning(
                    "Agent4SR corpus_id %r not found in method_signatures.csv "
                    "for %s-%s; omitted from top20.csv",
                    raw,
                    project,
                    bug_id,
                )
    else:
        logger.warning(
            "top5.txt not found for %s-%s model=%s",
            project,
            bug_id,
            model,
        )

    # top20/<model>.txt is consumed by a downstream step that expects the
    # dotted form (no $ separator); the CSV keeps canonical corpus identity.
    txt_path.write_text("\n".join(s.replace("$", ".") for s in txt_ids) + "\n", encoding="utf-8")
    _write_rows_csv(csv_path, rows)
    logger.info(
        "Wrote top20/%s: %d txt sigs, %d csv rows -> %s",
        name,
        len(txt_ids),
        len(rows),
        txt_path.parent,
    )
    return txt_path


def generate_top5(
    project: str,
    bug_id: str | int,
    *,
    lr_model_id: str,
    dataset: str = "defects4j",
) -> Path:
    """Generate ``top5/<lr_model_id>.txt`` and ``.csv`` from the Agent4LR run.

    Mirrors :func:`generate_top20`'s shape: ``.txt`` carries the dotted-form
    FQNs (no ``$`` separator); ``.csv`` joins against
    ``method_signatures.csv`` for path/line metadata. The source
    of truth is ``Agent4LR/<lr_model_id>/top5.txt`` — the runner writes
    that file with one resolved candidate FQN per line, ordered.

    Returns the path to the generated TXT file.
    """
    from src.common.config import get_lr_model_dir

    rankings_dir = get_rankings_dir(project, bug_id, dataset=dataset)
    txt_path = rankings_dir / "top5" / f"{lr_model_id}.txt"
    csv_path = rankings_dir / "top5" / f"{lr_model_id}.csv"
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # The LR runner writes top5.txt under FlexFL/LR/Agent4LR/<lr_model_id>/.
    # get_lr_model_dir falls back to _model_slug(model) when called without
    # an explicit lr_model_id; here we have the model_id directly.
    lr_dir = get_lr_model_dir(project, bug_id, model="", dataset=dataset, lr_model_id=lr_model_id)
    top5_src = lr_dir / "top5.txt"
    if not top5_src.exists():
        logger.warning(
            "Agent4LR top5.txt not found for %s-%s lr_model_id=%s",
            project,
            bug_id,
            lr_model_id,
        )
        txt_path.write_text("", encoding="utf-8")
        _write_rows_csv(csv_path, [])
        return txt_path

    try:
        all_entities = load_method_entities(get_processed_dir(project, bug_id, dataset=dataset))
    except FileNotFoundError:
        all_entities = []
    entity_index = _entity_index_by_corpus_id(all_entities)
    dotted_index: dict[str, MethodEntity] = {e.corpus_id.replace("$", "."): e for e in all_entities}
    stripped_index = _build_param_stripped_index(all_entities)

    rows: list[tuple[str, MethodEntity]] = []
    txt_ids: list[str] = []
    seen: set[str] = set()
    for line in top5_src.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        matched = _fuzzy_match_corpus_id(
            raw,
            entity_index=entity_index,
            dotted_index=dotted_index,
            stripped_index=stripped_index,
        )
        canonical = matched.corpus_id if matched is not None else raw
        if canonical in seen:
            continue
        seen.add(canonical)
        txt_ids.append(canonical)
        if matched is not None:
            rows.append(("Agent4LR", matched))
        else:
            logger.warning(
                "Agent4LR corpus_id %r not found in method_signatures.csv "
                "for %s-%s; omitted from top5.csv",
                raw,
                project,
                bug_id,
            )

    txt_path.write_text(
        "\n".join(s.replace("$", ".") for s in txt_ids) + ("\n" if txt_ids else ""),
        encoding="utf-8",
    )
    _write_rows_csv(csv_path, rows)
    logger.info(
        "Wrote top5/%s: %d txt sigs, %d csv rows -> %s",
        lr_model_id,
        len(txt_ids),
        len(rows),
        txt_path.parent,
    )
    return txt_path
