"""Blues statement-level IR scoring for SBIR.

Uses shared preprocessing and benchmark-aware statement extraction
(:func:`src.common.source_corpus.iter_statement_corpus`, Java for Defects4J /
Python for BugsInPy), then applies a 6-configuration Blues-style ensemble over
one BM25 run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import (
    BLUES_JSON,
    BLUES_STMT_SUSPS,
    get_processed_dir,
    get_sbir_dir,
    get_src_dir,
    get_stopwords_file,
)
from src.common.source_corpus import iter_statement_corpus
from src.common.text_processor import load_stopwords, preprocess

LOGGER = logging.getLogger(__name__)


def _build_bm25(corpus_tokens: list[list[str]]) -> Any:
    """Lazy-import rank_bm25 and build BM25Okapi index."""
    from importlib import import_module

    bm25_module = import_module("rank_bm25")
    return bm25_module.BM25Okapi(corpus_tokens)


@dataclass(frozen=True)
class BluesConfig:
    """One of the six Blues scoring configurations."""

    name: str
    m: int | None  # number of top statements per class (None = all)
    scoring: str  # "high" or "wted"


CONFIGS: tuple[BluesConfig, ...] = (
    BluesConfig(name="m1_high", m=1, scoring="high"),
    BluesConfig(name="m25_high", m=25, scoring="high"),
    BluesConfig(name="m50_high", m=50, scoring="high"),
    BluesConfig(name="m100_high", m=100, scoring="high"),
    BluesConfig(name="mall_high", m=None, scoring="high"),
    BluesConfig(name="mall_wted", m=None, scoring="wted"),
)


class Blues:
    """Compute Blues suspiciousness scores at statement granularity."""

    def __init__(self) -> None:
        self.stopwords = load_stopwords()

    @staticmethod
    def _read_bug_report(processed_dir: Path) -> str:
        """Read and concatenate bug report title + description."""
        bug_report_path = processed_dir / "bug_report.json"
        if not bug_report_path.exists():
            raise FileNotFoundError(f"Bug report not found: {bug_report_path}")

        with bug_report_path.open("r", encoding="utf-8") as handle:
            bug_report = json.load(handle)

        title = str(bug_report.get("title", ""))
        description = str(bug_report.get("description", ""))
        return f"{title} {description}".strip()

    def _build_statement_corpus(
        self, src_dir: Path, dataset: str
    ) -> tuple[list[dict[str, str]], list[list[str]]]:
        """Build the statement-level BM25 corpus from the source tree (Java or Python)."""
        statements = iter_statement_corpus(src_dir, dataset)
        LOGGER.info("Found %d source statements under %s", len(statements), src_dir)

        documents: list[dict[str, str]] = []
        corpus_tokens: list[list[str]] = []

        for statement in statements:
            tokens = preprocess(statement.content, stopwords=self.stopwords)
            if not tokens:
                continue
            documents.append(
                {
                    "stmt_id": statement.stmt_id,
                    "class_fqn": statement.class_fqn,
                }
            )
            corpus_tokens.append(tokens)

        LOGGER.info("Indexed %d preprocessed statements", len(corpus_tokens))
        return documents, corpus_tokens

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """Min-max normalise scores to [0, 1]."""
        if not scores:
            return []
        max_score = max(scores)
        if max_score <= 0.0:
            return [0.0 for _ in scores]
        return [score / max_score for score in scores]

    @staticmethod
    def _group_by_class(
        documents: list[dict[str, str]],
        normalized_scores: list[float],
    ) -> dict[str, list[tuple[str, float]]]:
        """Group (stmt_id, score) pairs by class FQN, sorted descending."""
        by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for doc, score in zip(documents, normalized_scores, strict=True):
            by_class[doc["class_fqn"]].append((doc["stmt_id"], score))

        for class_fqn in by_class:
            by_class[class_fqn].sort(key=lambda item: item[1], reverse=True)
        return by_class

    @staticmethod
    def _config_scores(
        by_class: dict[str, list[tuple[str, float]]],
        config: BluesConfig,
    ) -> dict[str, float]:
        """Compute scores for a single Blues configuration."""
        scores: dict[str, float] = {}

        for _, ranked_statements in by_class.items():
            limited = ranked_statements if config.m is None else ranked_statements[: config.m]
            if not limited:
                continue

            if config.scoring == "high":
                max_score = limited[0][1]
                for stmt_id, _ in limited:
                    scores[stmt_id] = max(scores.get(stmt_id, 0.0), max_score)
                continue

            if config.scoring == "wted":
                weighted_sum = 0.0
                for rank, (_, score) in enumerate(limited):
                    weighted_sum += score / (rank + 1)
                for stmt_id, _ in limited:
                    scores[stmt_id] = max(scores.get(stmt_id, 0.0), weighted_sum)
                continue

            raise ValueError(f"Unsupported scoring mode: {config.scoring}")

        return scores

    @staticmethod
    def _consensus_max(
        stmt_ids: list[str],
        config_score_maps: list[dict[str, float]],
    ) -> dict[str, float]:
        """Take the max score across all configurations for each statement."""
        consensus = {stmt_id: 0.0 for stmt_id in stmt_ids}
        for stmt_id in stmt_ids:
            consensus[stmt_id] = max(
                (score_map.get(stmt_id, 0.0) for score_map in config_score_maps),
                default=0.0,
            )
        return consensus

    @staticmethod
    def _apply_top_k_filter(scores: dict[str, float], top_k: int = 100) -> dict[str, float]:
        """Zero-out all but the top-k scoring statements."""
        ranked_stmt_ids = sorted(scores, key=lambda stmt_id: scores[stmt_id], reverse=True)
        keep = set(ranked_stmt_ids[:top_k])
        return {stmt_id: (score if stmt_id in keep else 0.0) for stmt_id, score in scores.items()}

    @staticmethod
    def _write_outputs(scores: dict[str, float], out_dir: Path) -> None:
        """Write Blues scores as JSON and ranked CSV."""
        json_output = out_dir / BLUES_JSON
        csv_output = out_dir / BLUES_STMT_SUSPS

        with json_output.open("w", encoding="utf-8") as handle:
            json.dump({"blues_scores": scores}, handle, indent=2, sort_keys=True)

        ranked_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        with csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Statement", "Suspiciousness"])
            for stmt_id, score in ranked_items:
                writer.writerow([stmt_id, score])

        LOGGER.info("Wrote Blues JSON: %s", json_output)
        LOGGER.info("Wrote Blues statement ranking: %s", csv_output)

    def process_project(
        self,
        project: str,
        bug_id: str | int,
        *,
        dataset: str = "defects4j",
    ) -> dict[str, float]:
        """Run Blues scoring, write JSON + CSV outputs, return scores dict."""
        self.stopwords = load_stopwords(get_stopwords_file(dataset))
        processed_dir = get_processed_dir(project, bug_id, dataset=dataset)
        src_dir = get_src_dir(project, bug_id, dataset=dataset)

        LOGGER.info("Running Blues for %s-%s", project, bug_id)
        LOGGER.info("Processed dir: %s", processed_dir)
        LOGGER.info("Source dir: %s", src_dir)

        query_text = self._read_bug_report(processed_dir)
        query_tokens = preprocess(query_text, stopwords=self.stopwords)
        if not query_tokens:
            raise ValueError("Bug report query is empty after preprocessing")

        documents, corpus_tokens = self._build_statement_corpus(src_dir, dataset)
        if not corpus_tokens:
            raise ValueError(f"No statements indexed from source directory: {src_dir}")

        bm25 = _build_bm25(corpus_tokens)
        raw_scores = bm25.get_scores(query_tokens)
        normalized_scores = self._normalize([float(score) for score in raw_scores])

        by_class = self._group_by_class(documents, normalized_scores)
        config_score_maps = [self._config_scores(by_class, config) for config in CONFIGS]

        stmt_ids = [doc["stmt_id"] for doc in documents]
        consensus_scores = self._consensus_max(stmt_ids, config_score_maps)
        filtered_scores = self._apply_top_k_filter(consensus_scores, top_k=100)

        sbir_dir = get_sbir_dir(project, bug_id, dataset=dataset)
        self._write_outputs(filtered_scores, sbir_dir)
        LOGGER.info("Finished Blues with %d statements", len(filtered_scores))
        return filtered_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Compute Blues statement suspiciousness")
    parser.add_argument("--project", required=True, help="Defects4J project name")
    parser.add_argument("--bug_id", required=True, help="Defects4J bug id")
    args = parser.parse_args()

    Blues().process_project(project=args.project, bug_id=args.bug_id)
