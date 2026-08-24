"""Combine Agent4SR results with SBIR/Ochiai/BoostN FL rankings.

Reads the SR Top-5 from the agent run output and merges with the
traditional FL method CSV outputs to produce a final candidate list.

Combination rule: ``SBIR[:5] + Ochiai[:5] + BoostN[:5] + SR_top5[:5]``.
Fallback (if SBIR/BoostN unavailable): ``Ochiai[:15]``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.agent4sr.tools import ToolContext, normalize_method_name
from src.common.config import (
    BOOSTN_CSV,
    SBFL_STMT_SUSPS,
    SBIR_STMT_SUSPS,
    _model_slug,
    get_boostn_dir,
    get_ochiai_dir,
    get_sbir_dir,
    get_sr_model_dir,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------


def _read_fl_csv(path: Path, key_column: str = "Signature") -> list[str]:
    """Read an FL ranking CSV and return method/statement IDs in rank order.

    The CSV is expected to have columns ``Signature,Suspiciousness`` (BoostN)
    or ``Statement,Suspiciousness`` (Ochiai/SBIR).
    """
    if not path.exists():
        raise FileNotFoundError(f"FL CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: list[str] = []
        for row in reader:
            val = row.get(key_column)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
        return out


# ---------------------------------------------------------------------------
# SR result loading
# ---------------------------------------------------------------------------


def _load_sr_top5(
    project: str,
    bug_id: str | int,
    model: str,
    *,
    dataset: str = "defects4j",
    model_id: str | None = None,
) -> list[str]:
    """Load the normalised Top-5 from an Agent4SR run result."""
    sr_path = (
        get_sr_model_dir(project, bug_id, model, dataset=dataset, model_id=model_id)
        / "sr_result.json"
    )
    if not sr_path.exists():
        raise FileNotFoundError(f"SR result not found: {sr_path}")
    raw = json.loads(sr_path.read_text(encoding="utf-8"))
    top5 = raw.get("top5")
    if not isinstance(top5, list):
        return []
    return [str(x) for x in top5 if isinstance(x, str) and x.strip()]


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def combine_candidates_for_bug(
    project: str,
    bug_id: str | int,
    model: str,
    *,
    dataset: str = "defects4j",
    model_id: str | None = None,
) -> list[str]:
    """Combine FL results + SR Top-5 into a single candidate list.

    Rule: ``SBIR[:5] + Ochiai[:5] + BoostN[:5] + SR[:5]``.
    Fallback: ``Ochiai[:15]`` if SBIR or BoostN is unavailable.
    """
    bug_id_str = str(bug_id)
    suspicious: list[str] = []

    try:
        sbir_dir = get_sbir_dir(project, bug_id, dataset=dataset)
        ochiai_dir = get_ochiai_dir(project, bug_id, dataset=dataset)
        boostn_dir = get_boostn_dir(project, bug_id, dataset=dataset)

        # SBIR and Ochiai use "Statement" column; BoostN uses "Signature"
        sbir_path = sbir_dir / SBIR_STMT_SUSPS
        ochiai_path = ochiai_dir / SBFL_STMT_SUSPS
        boostn_path = boostn_dir / BOOSTN_CSV

        suspicious.extend(_read_fl_csv(sbir_path, key_column="Statement")[:5])
        suspicious.extend(_read_fl_csv(ochiai_path, key_column="Statement")[:5])
        suspicious.extend(_read_fl_csv(boostn_path, key_column="Signature")[:5])
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Falling back to Ochiai-only for %s-%s: %s", project, bug_id, exc)
        ochiai_dir = get_ochiai_dir(project, bug_id, dataset=dataset)
        ochiai_path = ochiai_dir / SBFL_STMT_SUSPS
        try:
            suspicious = _read_fl_csv(ochiai_path, key_column="Statement")[: 5 * 3]
        except FileNotFoundError:
            logger.error("No Ochiai output found for %s-%s", project, bug_id)
            suspicious = []

    # Add SR Top-5
    try:
        sr_top5 = _load_sr_top5(project, bug_id, model, dataset=dataset, model_id=model_id)
        ctx = ToolContext(project=project, bug_id=bug_id_str, dataset=dataset)
        for m in sr_top5[:5]:
            suspicious.append(normalize_method_name(ctx=ctx, method_name=m))
    except FileNotFoundError as exc:
        logger.warning("No SR result for %s-%s model=%s: %s", project, bug_id, model, exc)

    return suspicious


def write_candidates(
    project: str,
    bug_id: str | int,
    model: str = "llama3.1:8b",
    *,
    dataset: str = "defects4j",
    model_id: str | None = None,
) -> Path:
    """Combine and write the candidate list to a text file.

    Returns the path to the written file.
    """
    cand = combine_candidates_for_bug(project, bug_id, model, dataset=dataset, model_id=model_id)
    name = model_id if model_id is not None else _model_slug(model)
    out_dir = (
        get_sr_model_dir(project, bug_id, model, dataset=dataset, model_id=model_id)
        / "candidates"
        / f"{name}_All"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.txt"
    out_path.write_text("\n".join(cand) + "\n", encoding="utf-8")
    logger.info("Candidates written to %s (%d entries)", out_path, len(cand))
    return out_path
