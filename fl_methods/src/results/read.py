"""Readers for the slim per-bug results layout.

A results bug dir mirrors the file schemas of the processed tree wherever a
file is shared (``method_signatures.csv``, ``faults*.csv``, the ranking CSVs
and SR top-20 lists under ``rankings/``), so the loaders in
:mod:`src.evaluation.sources` read those directly. This module adds the two
results-only files:

* ``lr.json`` — every Agent4LR config's ``top5`` plus token usage, replacing
  the per-config ``lr_result.json`` tree (whose full transcripts live in the
  gitignored ``transcripts/`` tree instead).
* ``meta.json`` — build provenance plus the processed-tree signals the
  BugsInPy exclusions report needs (``has_sr_result``, ``trigger_blank``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.sources import Usage

logger = logging.getLogger(__name__)

LR_JSON_NAME = "lr.json"
META_JSON_NAME = "meta.json"
RANKINGS_SUBDIR = "rankings"


@dataclass(frozen=True)
class LrEntry:
    """One Agent4LR config's slim result: ranked candidates + usage."""

    top5: list[str]
    usage: Usage


def _usage_from_totals(totals: dict) -> Usage:
    return Usage(
        input_tokens=int(totals.get("input_tokens") or 0),
        cached_tokens=int(totals.get("cached_tokens") or 0),
        output_tokens=int(totals.get("output_tokens") or 0),
        cost_usd=float(totals.get("cost_usd") or 0.0),
    )


def load_lr_json(results_dir: Path) -> dict[str, LrEntry]:
    """Return ``{config_name: LrEntry}`` from ``<results_dir>/lr.json``.

    Missing file returns an empty dict (a bug with no LR runs); malformed
    content is logged and treated as empty.
    """
    path = results_dir / LR_JSON_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("lr.json malformed: %s", path)
        return {}
    configs = data.get("configs")
    if not isinstance(configs, dict):
        logger.warning("lr.json has no configs mapping: %s", path)
        return {}

    entries: dict[str, LrEntry] = {}
    for name in sorted(configs):
        entry = configs[name]
        if not isinstance(entry, dict):
            continue
        top5 = entry.get("top5")
        totals = entry.get("usage_totals")
        entries[name] = LrEntry(
            top5=[s for s in top5 if isinstance(s, str)] if isinstance(top5, list) else [],
            usage=_usage_from_totals(totals) if isinstance(totals, dict) else Usage.zero(),
        )
    return entries


def check_required_configs_results(
    results_dir: Path,
    required: list[str],
) -> tuple[bool, list[str]]:
    """Results-tree twin of :func:`src.evaluation.sources.check_required_configs`.

    Validates against ``lr.json``: each named config must be present with a
    ``top5`` containing at least one non-empty string.
    """
    entries = load_lr_json(results_dir)
    problems: list[str] = []
    for cfg in required:
        entry = entries.get(cfg)
        if entry is None:
            problems.append(f"{cfg} (missing)")
            continue
        if not any(s.strip() for s in entry.top5):
            problems.append(f"{cfg} (invalid top5)")
    return (not problems, problems)


def load_meta(results_dir: Path) -> dict | None:
    """Return parsed ``meta.json`` or ``None`` when missing/malformed."""
    path = results_dir / META_JSON_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("meta.json malformed: %s", path)
        return None
    return data if isinstance(data, dict) else None
