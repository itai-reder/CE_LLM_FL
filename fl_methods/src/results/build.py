"""Build the slim per-bug results tree (and transcripts) from processed data.

For each bug, ``build_bug_results`` distills the full processed directory
into ``results/<Benchmark>/<Project>/<BugId>/``:

* ``method_signatures.csv`` — the ordered subsequence of the full corpus
  covering the *resolution closure*: every entity the evaluation can reach
  from the SR top-20 lists, the fault signatures, and every Agent4LR
  ``top5``. Source order is preserved so the exact (first-wins) and dotted
  (last-wins) lookup indexes resolve identically to the full file.
* ``faults.csv`` / ``faults_first.csv`` — verbatim copies.
* ``rankings/{ochiai,sbir,boostn}.csv`` — filtered to closure signatures
  (rows outside the candidate universe never influence a metric).
* ``rankings/top20/<sr_model_id>.txt`` — verbatim copies.
* ``lr.json`` — every config's ``top5`` + per-response token usage +
  aggregated totals (including cost), replacing the ``lr_result.json`` tree.
* ``meta.json`` — provenance plus the processed-tree signals the BugsInPy
  exclusions report needs.

Full LLM records are copied to ``transcripts/<Benchmark>/<Project>/<BugId>/``
(gitignored).

Every build ends with an equivalence self-check: the four metric lists are
computed once from the processed dir and once from the fresh results dir and
must match exactly; on mismatch the results dir is deleted and the build
fails. Output is deterministic (no timestamps, sorted keys) so rebuilding
from unchanged processed data is byte-stable under git.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.common.config import (
    FLEXFL_SR_SUBDIR,
    get_processed_dir,
    get_results_bug_dir,
    get_transcripts_bug_dir,
)
from src.common.method_entity import MethodEntity, load_method_entities
from src.core.layout import normalize_benchmark_name
from src.evaluation.per_bug import (
    evaluate_bug_inputs,
    load_bug_inputs_from_processed,
    load_bug_inputs_from_results,
)
from src.evaluation.sources import (
    DEFAULT_SR_MODEL_ID,
    _entity_indexes,
    discover_agent4lr_configs,
    load_agent4lr_usage,
)
from src.results.read import LR_JSON_NAME, META_JSON_NAME, RANKINGS_SUBDIR

logger = logging.getLogger(__name__)

BASELINE_RANKING_NAMES = ("ochiai", "sbir", "boostn")


class ResultsBuildError(RuntimeError):
    """Raised when a built results dir fails the metric equivalence check."""


# ---------------------------------------------------------------------------
# Closure computation
# ---------------------------------------------------------------------------


def _resolve(
    raw: str,
    entity_index: dict[str, MethodEntity],
    dotted_index: dict[str, MethodEntity],
) -> MethodEntity | None:
    return entity_index.get(raw) or dotted_index.get(raw)


def _fault_signature_ids(csv_path: Path, entity_index: dict[str, MethodEntity]) -> set[str]:
    """Corpus ids of resolvable fault signatures (exact index only,
    mirroring ``per_bug._ids_to_entities``)."""
    if not csv_path.exists():
        return set()
    ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sig = (row.get("signature") or "").strip()
            if sig and sig in entity_index:
                ids.add(sig)
    return ids


def _top20_files(processed_dir: Path) -> list[Path]:
    top20_dir = processed_dir / FLEXFL_SR_SUBDIR / "rankings" / "top20"
    if not top20_dir.is_dir():
        return []
    return sorted(p for p in top20_dir.iterdir() if p.is_file() and p.suffix == ".txt")


def _compute_closure(
    processed_dir: Path,
    entities: list[MethodEntity],
    lr_configs: list[tuple[str, dict[str, Any]]],
) -> set[str]:
    """Corpus ids of every entity the evaluation can resolve for this bug."""
    entity_index, dotted_index = _entity_indexes(entities)
    closure: set[str] = set()

    for top20 in _top20_files(processed_dir):
        for line in top20.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            matched = _resolve(raw, entity_index, dotted_index)
            if matched is not None:
                closure.add(matched.corpus_id)

    for name in ("faults.csv", "faults_first.csv"):
        closure |= _fault_signature_ids(processed_dir / name, entity_index)

    for _config, data in lr_configs:
        top5 = data.get("top5") or []
        if not isinstance(top5, list):
            continue
        for fqn in top5:
            if not isinstance(fqn, str) or not fqn.strip():
                continue
            matched = _resolve(fqn.strip(), entity_index, dotted_index)
            if matched is not None:
                closure.add(matched.corpus_id)

    return closure


# ---------------------------------------------------------------------------
# Slim file writers
# ---------------------------------------------------------------------------


def _write_filtered_semicolon_csv(
    src: Path,
    dest: Path,
    key_column: str,
    keep_ids: set[str],
    *,
    first_occurrence_only: bool = False,
) -> int:
    """Copy rows of a ``;``-delimited CSV whose *key_column* is in *keep_ids*.

    Preserves source row order (required for lookup-index equivalence).
    Returns the number of data rows written.
    """
    written = 0
    seen: set[str] = set()
    with src.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin, delimiter=";")
        header = next(reader, None)
        if header is None:
            dest.write_text("", encoding="utf-8")
            return 0
        try:
            key_idx = header.index(key_column)
        except ValueError as exc:
            raise ResultsBuildError(f"{src}: no {key_column!r} column") from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, delimiter=";", lineterminator="\n")
            writer.writerow(header)
            for row in reader:
                if len(row) <= key_idx:
                    continue
                key = row[key_idx].strip()
                if key not in keep_ids:
                    continue
                if first_occurrence_only:
                    if key in seen:
                        continue
                    seen.add(key)
                writer.writerow(row)
                written += 1
    return written


def _build_lr_payload(
    lr_configs: list[tuple[str, dict[str, Any], Path]],
) -> dict[str, Any]:
    """Assemble the ``lr.json`` content from raw ``lr_result.json`` payloads."""
    configs: dict[str, Any] = {}
    for config_name, data, lr_path in lr_configs:
        top5 = data.get("top5")
        top5_indices = data.get("top5_indices")
        responses: list[dict[str, Any]] = []
        dumps = data.get("response_dumps")
        if isinstance(dumps, list):
            for dump in dumps:
                if not isinstance(dump, dict):
                    continue
                usage = dump.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                details = usage.get("input_tokens_details") or {}
                cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
                responses.append(
                    {
                        "model": str(dump.get("model") or ""),
                        "input_tokens": int(usage.get("input_tokens") or 0),
                        "cached_tokens": cached,
                        "output_tokens": int(usage.get("output_tokens") or 0),
                    }
                )
        totals = load_agent4lr_usage(lr_path)
        configs[config_name] = {
            "top5": top5 if isinstance(top5, list) else [],
            "top5_indices": top5_indices if isinstance(top5_indices, list) else [],
            "responses": responses,
            "usage_totals": {
                "input_tokens": totals.input_tokens,
                "cached_tokens": totals.cached_tokens,
                "output_tokens": totals.output_tokens,
                "cost_usd": totals.cost_usd,
            },
        }
    return {"schema_version": 1, "configs": configs}


def _has_sr_result(processed_dir: Path) -> bool:
    agent_dir = processed_dir / FLEXFL_SR_SUBDIR / "Agent4SR"
    if not agent_dir.is_dir():
        return False
    return any((d / "sr_result.json").exists() for d in agent_dir.iterdir() if d.is_dir())


def _trigger_blank(processed_dir: Path) -> bool:
    trigger = processed_dir / "trigger_test_clean.txt"
    return trigger.exists() and not trigger.read_text(encoding="utf-8").strip()


def _copy_transcripts(
    processed_dir: Path,
    transcripts_dir: Path,
    lr_configs: list[tuple[str, Path]],
) -> int:
    """Copy full LR/SR result JSONs into the transcripts tree; return count."""
    copied = 0
    for config_name, lr_path in lr_configs:
        dest = transcripts_dir / "Agent4LR" / config_name / lr_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lr_path, dest)
        copied += 1
    agent_dir = processed_dir / FLEXFL_SR_SUBDIR / "Agent4SR"
    if agent_dir.is_dir():
        for model_dir in sorted(agent_dir.iterdir()):
            sr_result = model_dir / "sr_result.json"
            if model_dir.is_dir() and sr_result.exists():
                dest = transcripts_dir / "Agent4SR" / model_dir.name / sr_result.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sr_result, dest)
                copied += 1
    return copied


# ---------------------------------------------------------------------------
# Per-bug build
# ---------------------------------------------------------------------------


def build_bug_results(
    project: str,
    bug_id: str | int,
    *,
    dataset: str = "defects4j",
    include_transcripts: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Build one bug's slim results dir (and transcripts). Returns a status dict."""
    processed_dir = get_processed_dir(project, bug_id, dataset=dataset)
    results_dir = get_results_bug_dir(project, bug_id, dataset=dataset)

    if not processed_dir.is_dir():
        return {"status": "skipped", "reason": "processed dir missing"}
    if results_dir.exists() and not force:
        return {"status": "exists"}

    try:
        entities = load_method_entities(processed_dir)
    except FileNotFoundError:
        return {"status": "skipped", "reason": "no method_signatures.csv"}

    # Raw LR payloads (read once; feed the closure, lr.json, and transcripts).
    raw_lr: list[tuple[str, dict[str, Any], Path]] = []
    for config_name, lr_path in discover_agent4lr_configs(processed_dir):
        try:
            data = json.loads(lr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping malformed lr_result.json: %s", lr_path)
            continue
        if isinstance(data, dict):
            raw_lr.append((config_name, data, lr_path))

    closure = _compute_closure(processed_dir, entities, [(n, d) for n, d, _ in raw_lr])

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Slim universe (first occurrence per corpus_id, source order kept).
        n_entities = _write_filtered_semicolon_csv(
            processed_dir / "method_signatures.csv",
            results_dir / "method_signatures.csv",
            "corpus_id",
            closure,
            first_occurrence_only=True,
        )

        # 2. Ground truth, verbatim.
        for name in ("faults.csv", "faults_first.csv"):
            src = processed_dir / name
            if src.exists():
                shutil.copy2(src, results_dir / name)

        # 3. Baseline rankings filtered to the closure; top-20 lists verbatim.
        rankings_src = processed_dir / FLEXFL_SR_SUBDIR / "rankings"
        rankings_dest = results_dir / RANKINGS_SUBDIR
        for name in BASELINE_RANKING_NAMES:
            src = rankings_src / f"{name}.csv"
            if src.exists():
                _write_filtered_semicolon_csv(
                    src, rankings_dest / f"{name}.csv", "signature", closure
                )
        for top20 in _top20_files(processed_dir):
            dest = rankings_dest / "top20" / top20.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(top20, dest)

        # 4. lr.json (only when the bug has LR runs).
        if raw_lr:
            payload = _build_lr_payload(raw_lr)
            (results_dir / LR_JSON_NAME).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        # 5. meta.json (deterministic: no timestamps).
        meta = {
            "schema_version": 1,
            "benchmark": dataset,
            "project": project,
            "bug_id": int(bug_id),
            "has_sr_result": _has_sr_result(processed_dir),
            "trigger_blank": _trigger_blank(processed_dir),
            # Canonical layout path, not the resolved one: CEFL_*_WORKSPACE may
            # point the processed tree anywhere, and meta.json must stay
            # machine-independent and byte-stable.
            "source": f"data/{normalize_benchmark_name(dataset)}/processed/{project}/{int(bug_id)}",
        }
        (results_dir / META_JSON_NAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # 6. Transcripts (gitignored tree; independent of the self-check).
        n_transcripts = 0
        if include_transcripts:
            transcripts_dir = get_transcripts_bug_dir(project, bug_id, dataset=dataset)
            n_transcripts = _copy_transcripts(
                processed_dir, transcripts_dir, [(n, p) for n, _, p in raw_lr]
            )

        # 7. Equivalence self-check, once per SR model id present.
        sr_ids = [p.stem for p in _top20_files(processed_dir)] or [DEFAULT_SR_MODEL_ID]
        for sr_id in sr_ids:
            expected = evaluate_bug_inputs(
                load_bug_inputs_from_processed(processed_dir, sr_model_id=sr_id)
            )
            actual = evaluate_bug_inputs(
                load_bug_inputs_from_results(results_dir, sr_model_id=sr_id)
            )
            if expected != actual:
                raise ResultsBuildError(
                    f"{project}-{bug_id} ({sr_id}): results-dir metrics diverge from "
                    f"processed-dir metrics; refusing to keep the slim build"
                )
    except Exception:
        shutil.rmtree(results_dir, ignore_errors=True)
        raise

    return {
        "status": "built",
        "entities": n_entities,
        "closure": len(closure),
        "lr_configs": len(raw_lr),
        "transcripts": n_transcripts,
    }
