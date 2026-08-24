"""BoostN method-level IR scoring for fault localization.

Uses a specialised BM25 variant with custom IDF and adaptive k1
parameter. Methods shorter than 11 characters are filtered to 0.0.

Key differences from Blues (statement-level):
  - Adaptive k1: 0.0 for <= 3000 methods, 1.0 for > 3000
  - b = 0.3 (Blues uses default 0.75)
  - Custom IDF with epsilon correction for negative values
  - Short-method filter (content < 11 chars -> score 0.0)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.common import config
from src.common.source_corpus import iter_method_corpus
from src.common.text_processor import load_stopwords, preprocess

logger = logging.getLogger(__name__)


class BM25Variant(BM25Okapi):
    """BoostN BM25 variant with custom IDF and optional IDF-only scoring."""

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.0,
        b: float = 0.3,
        epsilon: float = 0.25,
    ) -> None:
        # Must call super first, then override k1/b (parent sets defaults)
        super().__init__(corpus)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

    def _calc_idf(self, nd: dict[str, int]) -> None:
        """Custom IDF: log(1 + (N - n + 0.5) / (n + 0.5)), with epsilon floor."""
        idf_sum = 0.0
        negative_idfs: list[str] = []
        for word, freq in nd.items():
            idf = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf) if len(self.idf) > 0 else 0

        if self.average_idf < 0:
            self.average_idf = self.epsilon

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, document: Iterable[str], index: int) -> float:
        """Score a query against a single document. Uses IDF-only when k1=0."""
        score = 0.0
        doc_len = self.doc_len[index]
        for word in document:
            if word not in self.doc_freqs[index]:
                continue

            if self.k1 == 0.0:
                score += self.idf[word]
            else:
                tf = self.doc_freqs[index][word]
                numerator = self.idf[word] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
        return score

    def get_scores(self, query: list[str]) -> np.ndarray:
        """Return scores for all documents.

        In IDF-only mode (k1 == 0), we avoid the parent vectorized BM25 expression,
        which can emit NaN due to 0/0 terms in some corpora.
        """
        if self.k1 == 0.0:
            scores = np.zeros(self.corpus_size, dtype=float)
            for index in range(self.corpus_size):
                scores[index] = self.get_score(query, index)
            return scores

        scores = super().get_scores(query)
        cleaned: np.ndarray = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned


class BoostN:
    """BoostN processor for one project bug instance (Defects4J or BugsInPy)."""

    def __init__(self, stopwords: frozenset[str] | None = None) -> None:
        self._stopwords_override = stopwords
        self.stopwords = (
            stopwords if stopwords is not None else load_stopwords(config.STOPWORDS_FILE)
        )

    def process_project(
        self,
        project: str,
        bug_id: str | int,
        *,
        dataset: str = "defects4j",
    ) -> dict[str, float]:
        """Run BoostN scoring and write JSON+CSV outputs for one bug."""
        if self._stopwords_override is None:
            self.stopwords = load_stopwords(config.get_stopwords_file(dataset))
        processed_dir = config.get_processed_dir(project, bug_id, dataset=dataset)
        src_dir = config.get_src_dir(project, bug_id, dataset=dataset)
        bug_report_path = processed_dir / "bug_report.json"

        if not bug_report_path.exists():
            raise FileNotFoundError(f"Missing bug report: {bug_report_path}")
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source directory: {src_dir}")

        with open(bug_report_path, encoding="utf-8") as file_obj:
            bug_report = json.load(file_obj)

        query_text = f"{bug_report.get('title', '')} {bug_report.get('description', '')}".replace(
            "\n", " "
        )
        query_tokens = preprocess(query_text, self.stopwords)

        method_pairs = iter_method_corpus(src_dir, dataset)
        logger.info(
            "%s-%s: extracted %d methods from %s", project, bug_id, len(method_pairs), src_dir
        )

        results: dict[str, float] = {}
        if method_pairs:
            corpus = [preprocess(m.content, self.stopwords) for m, _ in method_pairs]
            # Adaptive k1: IDF-only for small corpora, full BM25 for large
            k1 = 1.0 if len(corpus) > 3000 else 0.0
            bm25 = BM25Variant(corpus, k1=k1, b=0.3)

            scores = bm25.get_scores(query_tokens)
            max_score = max(scores) if len(scores) > 0 else 0.0
            score_divisor = max_score if max_score > 0 else 1.0

            for (method_info, corpus_id), score in zip(method_pairs, scores, strict=True):
                value = 0.0 if len(method_info.content) < 11 else float(score / score_divisor)
                # Overload collisions in the corpus identity collapse via max.
                prev = results.get(corpus_id)
                results[corpus_id] = value if prev is None else max(prev, value)

        out_dir = config.get_boostn_dir(project, bug_id, dataset=dataset)
        self._write_outputs(out_dir, results)
        logger.info(
            "%s-%s: wrote %s and %s to %s",
            project,
            bug_id,
            config.BOOSTN_JSON,
            config.BOOSTN_CSV,
            out_dir,
        )
        return results

    @staticmethod
    def _write_outputs(out_dir: Path, results: dict[str, float]) -> None:
        """Write BoostN scores as JSON and ranked CSV."""
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / config.BOOSTN_JSON
        csv_path = out_dir / config.BOOSTN_CSV

        with open(json_path, "w", encoding="utf-8") as file_obj:
            json.dump({"boostn_scores": results}, file_obj, indent=2)

        sorted_items = sorted(results.items(), key=lambda item: item[1], reverse=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["Signature", "Suspiciousness"])
            for signature, score in sorted_items:
                writer.writerow([signature, score])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run BoostN method-level fault localization.")
    parser.add_argument("--project", required=True, help="Defects4J project name, e.g., Lang")
    parser.add_argument("--bug_id", required=True, help="Defects4J bug id, e.g., 1")
    args = parser.parse_args()

    boostn = BoostN()
    boostn.process_project(args.project, args.bug_id)
