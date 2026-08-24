"""Ochiai SBFL (Spectrum-Based Fault Localization).

Parses GZoltar's Ochiai ranking output and produces normalised
statement-level suspiciousness scores in JSON + CSV format.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import cast

from src.common.config import (
    OCHIAI_RANKING_FILE,
    SBFL_JSON,
    SBFL_STMT_SUSPS,
    get_ochiai_dir,
    get_ochiai_ranking_dir,
)
from src.common.coverage import _parse_spectra_id
from src.core.layout import normalize_benchmark_name

LOGGER = logging.getLogger(__name__)


class SBFL:
    """Parse GZoltar Ochiai ranking and write normalised outputs."""

    @staticmethod
    def _to_statement_id(raw_name: str) -> str:
        """Convert GZoltar name ``pkg$Class#method(params):line`` to ``pkg.Class#line``."""
        class_part = raw_name.split("#", 1)[0].replace("$", ".")
        line_number = raw_name.rsplit(":", 1)[1]
        return f"{class_part}#{line_number}"

    @classmethod
    def _reduce_statement_id(cls, raw_name: str, *, is_bugsinpy: bool) -> str | None:
        """Reduce a spectra row name to the canonical ``<class_fqn>#<line>`` statement id.

        Defects4J (GZoltar) ids are ``pkg$Class#method(params):line``. BugsInPy
        (FauxPy) ids carry no ``#`` (``<module>$<qualname>(params):line``), so they
        are reduced via :func:`coverage._parse_spectra_id` — the same Python-shape
        logic used to parse the coverage spectra — to the class-bearing form the
        statement→method aggregation joins on. Returns ``None`` for a row
        that does not parse.
        """
        if is_bugsinpy:
            parsed = _parse_spectra_id(raw_name)
            if parsed is None:
                return None
            class_fqn, line = parsed
            return f"{class_fqn}#{line}"
        if "#" not in raw_name or ":" not in raw_name:
            return None
        return cls._to_statement_id(raw_name)

    @classmethod
    def parse_ochiai_csv(cls, file_path: Path, *, dataset: str = "defects4j") -> dict[str, float]:
        """Parse a semicolon-delimited Ochiai ranking CSV (GZoltar or FauxPy)."""
        if not file_path.exists():
            raise FileNotFoundError(f"Ochiai ranking file not found: {file_path}")

        is_bugsinpy = normalize_benchmark_name(dataset) == "BIP"
        scores: dict[str, float] = {}
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=";")
            header = next(reader, None)
            if header is None:
                return scores

            for row in reader:
                if len(row) < 2:
                    continue

                raw_name = row[0].strip()
                raw_score = row[1].strip()
                if not raw_name or not raw_score:
                    continue

                stmt_id = cls._reduce_statement_id(raw_name, is_bugsinpy=is_bugsinpy)
                if stmt_id is None:
                    LOGGER.warning("Skipping malformed statement name: %s", raw_name)
                    continue

                try:
                    score = float(raw_score)
                except ValueError:
                    LOGGER.warning("Skipping malformed suspiciousness value: %s", raw_score)
                    continue

                scores[stmt_id] = score

        return scores

    @staticmethod
    def _write_json(scores: dict[str, float], output_path: Path) -> None:
        payload = {"sbfl_scores": scores}
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _write_stmt_susps(scores: dict[str, float], output_path: Path) -> None:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Statement", "Suspiciousness"])
            for stmt_id, score in ranked:
                writer.writerow([stmt_id, score])

    @classmethod
    def load_ranking(cls, file_path: Path | str) -> dict[str, float]:
        """Load a previously-written statement ranking CSV (comma-delimited)."""
        ranking_path = Path(file_path)
        if not ranking_path.exists():
            raise FileNotFoundError(f"Ranking file not found: {ranking_path}")

        scores: dict[str, float] = {}
        with ranking_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return scores

            for row in reader:
                if len(row) < 2:
                    continue
                stmt_id = row[0].strip()
                raw_score = row[1].strip()
                if not stmt_id or not raw_score:
                    continue
                try:
                    scores[stmt_id] = float(raw_score)
                except ValueError:
                    LOGGER.warning("Skipping malformed score in ranking file: %s", raw_score)
        return scores

    def process_project(
        self,
        project: str,
        bug_id: str | int,
        *,
        dataset: str = "defects4j",
    ) -> dict[str, float]:
        """Parse Ochiai ranking, write JSON + CSV outputs to Ochiai/ subdir."""
        out_dir = get_ochiai_dir(project, bug_id, dataset=dataset)
        ranking_dir = get_ochiai_ranking_dir(project, bug_id, dataset=dataset)
        ranking_file = ranking_dir / OCHIAI_RANKING_FILE
        json_output = out_dir / SBFL_JSON
        stmt_susps_output = out_dir / SBFL_STMT_SUSPS
        LOGGER.info("Loading Ochiai ranking from %s", ranking_file)
        scores = self.parse_ochiai_csv(ranking_file, dataset=dataset)
        self._write_json(scores, json_output)
        self._write_stmt_susps(scores, stmt_susps_output)
        return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Parse existing GZoltar Ochiai ranking")
    _ = parser.add_argument("--project", required=True, help="Defects4J project name")
    _ = parser.add_argument("--bug_id", required=True, help="Defects4J bug id")
    args = parser.parse_args()

    project = cast(str, args.project)
    bug_id = cast(str, args.bug_id)
    SBFL().process_project(project=project, bug_id=bug_id)
