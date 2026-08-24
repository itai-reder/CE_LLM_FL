"""RAFL -- Rank Aggregation for Fault Localization.

Aggregates SBFL (Ochiai) and Blues statement rankings using Borda count,
with optional Cross-Entropy (CE) Monte Carlo optimisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from random import Random

from src.common.config import (
    BLUES_STMT_SUSPS,
    SBFL_STMT_SUSPS,
    SBIR_JSON,
    SBIR_STMT_SUSPS,
    get_ochiai_dir,
    get_sbir_dir,
)

LOGGER = logging.getLogger(__name__)


class CLIArgs(argparse.Namespace):
    """Typed namespace for CLI argument parsing."""

    project: str = ""
    bug_id: str = ""
    use_ce: bool = False
    ce_max_iter: int | None = None
    ce_pop_size: int | None = None
    sbfl_weight: float = 0.5


@dataclass(frozen=True)
class CERunConfig:
    """Hyperparameters for CE Monte Carlo rank optimisation."""

    samples_per_iteration: int = 10_000
    pop_size: int = 100
    rho: float = 0.01
    convergence_window: int = 7
    max_iterations: int = 1_000
    crossover_prob: float = 0.4
    mutation_prob: float = 0.01
    seed: int = 42
    smoothing: float = 0.7


class RAFL:
    """Rank aggregation for SBIR using Borda or optional CE Monte Carlo."""

    def __init__(self, ce_config: CERunConfig | None = None) -> None:
        self.ce_config: CERunConfig = ce_config or CERunConfig()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_scores(file_path: Path) -> dict[str, float]:
        """Load a statement ranking CSV (comma-delimited)."""
        if not file_path.exists():
            raise FileNotFoundError(f"Ranking file not found: {file_path}")

        scores: dict[str, float] = {}
        with file_path.open("r", encoding="utf-8", newline="") as handle:
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
                    score = float(raw_score)
                except ValueError:
                    LOGGER.warning("Skipping malformed suspiciousness value: %s", raw_score)
                    continue
                scores[stmt_id] = score
        return scores

    # ------------------------------------------------------------------
    # Ranking utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_to_ranks(scores: dict[str, float]) -> dict[str, int]:
        """Convert scores to 1-indexed ranks, assigning the same rank for ties."""
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ranks: dict[str, int] = {}
        previous_score: float | None = None
        current_rank = 0

        for index, (stmt_id, score) in enumerate(ranked, start=1):
            if previous_score is None or score != previous_score:
                current_rank = index
                previous_score = score
            ranks[stmt_id] = current_rank
        return ranks

    @staticmethod
    def _compute_borda_ranking(
        sbfl_ranks: dict[str, int],
        blues_ranks: dict[str, int],
        *,
        sbfl_weight: float = 0.5,
    ) -> tuple[list[str], dict[str, float]]:
        """Compute a Borda-count aggregated ranking from two input rankings."""
        if not 0.0 <= sbfl_weight <= 1.0:
            raise ValueError(f"sbfl_weight must be in [0, 1], got {sbfl_weight}")

        blues_weight = 1.0 - sbfl_weight
        n1 = len(sbfl_ranks)
        n2 = len(blues_ranks)
        all_statements = sorted(set(sbfl_ranks) | set(blues_ranks))

        borda_scores: dict[str, float] = {}
        for stmt_id in all_statements:
            rank_sbfl = sbfl_ranks.get(stmt_id, n1 + 1)
            rank_blues = blues_ranks.get(stmt_id, n2 + 1)
            score_sbfl = n1 + 1 - rank_sbfl
            score_blues = n2 + 1 - rank_blues
            borda_scores[stmt_id] = sbfl_weight * score_sbfl + blues_weight * score_blues

        ranking = sorted(all_statements, key=lambda stmt: (-borda_scores[stmt], stmt))
        return ranking, borda_scores

    @staticmethod
    def _ranking_to_position_map(ranking: list[str]) -> dict[str, int]:
        """Convert an ordered ranking list to a {stmt_id: 1-based position} map."""
        return {stmt_id: idx for idx, stmt_id in enumerate(ranking, start=1)}

    @staticmethod
    def _spearman_footrule_distance(
        candidate_ranking: list[str],
        input_ranks: dict[str, int],
    ) -> int:
        """Compute Spearman footrule distance between candidate and source ranking."""
        candidate_positions = RAFL._ranking_to_position_map(candidate_ranking)
        default_rank = len(input_ranks) + 1

        distance = 0
        for stmt_id, candidate_pos in candidate_positions.items():
            source_rank = input_ranks.get(stmt_id, default_rank)
            distance += abs(candidate_pos - source_rank)
        return distance

    def _objective(
        self,
        candidate_ranking: list[str],
        input_rankings: list[dict[str, int]],
    ) -> int:
        """Sum of Spearman footrule distances across all input rankings."""
        return sum(
            self._spearman_footrule_distance(candidate_ranking, source_ranks)
            for source_ranks in input_rankings
        )

    # ------------------------------------------------------------------
    # CE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _induce_scope_ranks(source_ranks: dict[str, int], scope: list[str]) -> dict[str, int]:
        """Induce rankings for a subset of statements from source rankings."""
        present = [stmt_id for stmt_id in scope if stmt_id in source_ranks]
        present.sort(key=lambda stmt_id: (source_ranks[stmt_id], stmt_id))

        induced: dict[str, int] = {}
        previous_source_rank: int | None = None
        current_rank = 0

        for index, stmt_id in enumerate(present, start=1):
            source_rank = source_ranks[stmt_id]
            if previous_source_rank is None or source_rank != previous_source_rank:
                current_rank = index
                previous_source_rank = source_rank
            induced[stmt_id] = current_rank

        worst_rank = len(present) + 1
        for stmt_id in sorted(s for s in scope if s not in source_ranks):
            induced[stmt_id] = worst_rank

        return induced

    @staticmethod
    def _build_initial_probability_matrix(
        elements: list[str],
        initial_ranking: list[str],
    ) -> dict[str, list[float]]:
        """Build position probability matrix seeded from the initial ranking."""
        n = len(elements)
        matrix = {stmt_id: [1.0 / n] * n for stmt_id in elements}

        if n == 1:
            matrix[elements[0]][0] = 1.0
            return matrix

        boost = 0.4
        for pos, stmt_id in enumerate(initial_ranking):
            base = (1.0 - boost) / n
            row = [base] * n
            row[pos] += boost
            matrix[stmt_id] = row
        return matrix

    @staticmethod
    def _select_weighted_without_replacement(
        rng: Random,
        available: list[str],
        weights: list[float],
    ) -> str:
        """Weighted random selection without replacement."""
        total = sum(weights)
        if total <= 0.0:
            return available[rng.randrange(len(available))]

        threshold = rng.random() * total
        partial = 0.0
        for idx, weight in enumerate(weights):
            partial += weight
            if partial >= threshold:
                return available[idx]
        return available[-1]

    def _sample_permutation(
        self,
        rng: Random,
        elements: list[str],
        matrix: dict[str, list[float]],
        pivot: list[str],
        crossover_prob: float,
        mutation_prob: float,
    ) -> list[str]:
        """Sample a permutation from the probability matrix with crossover/mutation."""
        n = len(elements)
        available = elements.copy()
        sampled: list[str] = []

        for pos in range(n):
            weights = [matrix[stmt_id][pos] for stmt_id in available]
            chosen = self._select_weighted_without_replacement(rng, available, weights)
            sampled.append(chosen)
            available.remove(chosen)

        if n >= 2 and rng.random() < crossover_prob:
            a = rng.randrange(n)
            b = rng.randrange(n)
            left, right = (a, b) if a <= b else (b, a)
            segment = pivot[left : right + 1]
            segment_set = set(segment)
            remainder = [stmt for stmt in sampled if stmt not in segment_set]
            sampled = remainder[:left] + segment + remainder[left:]

        if n >= 2 and mutation_prob > 0.0:
            swaps = max(1, int(n * mutation_prob))
            for _ in range(swaps):
                i = rng.randrange(n)
                j = rng.randrange(n)
                sampled[i], sampled[j] = sampled[j], sampled[i]

        return sampled

    def _update_probability_matrix(
        self,
        old_matrix: dict[str, list[float]],
        elite_samples: list[list[str]],
        smoothing: float,
    ) -> dict[str, list[float]]:
        """Update the probability matrix from elite samples with smoothing."""
        n = len(elite_samples[0])
        elite_count = len(elite_samples)

        frequencies = {stmt_id: [0.0] * n for stmt_id in old_matrix}
        for perm in elite_samples:
            for pos, stmt_id in enumerate(perm):
                frequencies[stmt_id][pos] += 1.0 / elite_count

        updated: dict[str, list[float]] = {}
        for stmt_id, old_row in old_matrix.items():
            row: list[float] = []
            for pos in range(n):
                row.append((1.0 - smoothing) * old_row[pos] + smoothing * frequencies[stmt_id][pos])

            row_sum = sum(row)
            row = [1.0 / n] * n if row_sum <= 0.0 else [v / row_sum for v in row]
            updated[stmt_id] = row

        return updated

    def _run_ce(
        self,
        initial_ranking: list[str],
        input_rankings: list[dict[str, int]],
        config: CERunConfig,
    ) -> list[str]:
        """Run Cross-Entropy optimisation to find a consensus ranking."""
        rng = Random(config.seed)
        elements = initial_ranking.copy()
        matrix = self._build_initial_probability_matrix(elements, initial_ranking)

        best = initial_ranking.copy()
        best_cost = self._objective(best, input_rankings)
        stale_iterations = 0

        population_size = max(1, min(config.pop_size, config.samples_per_iteration))
        elite_size = max(1, int(population_size * config.rho))

        LOGGER.info(
            "Starting CE optimization with seed=%d, samples=%d, pop=%d, elite=%d, max_iter=%d",
            config.seed,
            config.samples_per_iteration,
            population_size,
            elite_size,
            config.max_iterations,
        )

        for iteration in range(1, config.max_iterations + 1):
            sampled: list[tuple[int, list[str]]] = []
            for _ in range(config.samples_per_iteration):
                candidate = self._sample_permutation(
                    rng=rng,
                    elements=elements,
                    matrix=matrix,
                    pivot=initial_ranking,
                    crossover_prob=config.crossover_prob,
                    mutation_prob=config.mutation_prob,
                )
                sampled.append((self._objective(candidate, input_rankings), candidate))

            sampled.sort(key=lambda item: item[0])
            population = sampled[:population_size]
            elites = population[:elite_size]
            elite_best_cost, elite_best_perm = elites[0]

            if elite_best_cost < best_cost:
                best = elite_best_perm
                best_cost = elite_best_cost
                stale_iterations = 0
            else:
                stale_iterations += 1

            matrix = self._update_probability_matrix(
                old_matrix=matrix,
                elite_samples=[perm for _, perm in elites],
                smoothing=config.smoothing,
            )

            if iteration % 5 == 0 or iteration == 1:
                LOGGER.info(
                    "CE iteration %d: current_best_cost=%d, stale=%d",
                    iteration,
                    best_cost,
                    stale_iterations,
                )

            if stale_iterations >= config.convergence_window:
                LOGGER.info(
                    "CE convergence reached after %d iterations (window=%d)",
                    iteration,
                    config.convergence_window,
                )
                break

        LOGGER.info("CE completed with best cost=%d", best_cost)
        return best

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _ranking_to_scores(ranking: list[str]) -> dict[str, float]:
        """Convert an ordered ranking to normalised [0, 1] scores."""
        total = len(ranking)
        if total == 0:
            return {}

        scores: dict[str, float] = {}
        for idx, stmt_id in enumerate(ranking, start=1):
            scores[stmt_id] = (total - idx + 1) / total
        return scores

    @staticmethod
    def _write_outputs(out_dir: Path, scores: dict[str, float]) -> None:
        """Write SBIR final ranking as CSV + JSON."""
        csv_path = out_dir / SBIR_STMT_SUSPS
        json_path = out_dir / SBIR_JSON

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Statement", "Suspiciousness"])
            for stmt_id, score in ranked:
                writer.writerow([stmt_id, score])

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump({"sbir_scores": dict(ranked)}, handle, indent=2)

        LOGGER.info("Wrote SBIR CSV to %s", csv_path)
        LOGGER.info("Wrote SBIR JSON to %s", json_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_project(
        self,
        project: str,
        bug_id: str | int,
        use_ce: bool = False,
        ce_max_iter: int | None = None,
        ce_pop_size: int | None = None,
        sbfl_weight: float = 0.5,
        *,
        dataset: str = "defects4j",
    ) -> dict[str, float]:
        """Aggregate SBFL + Blues rankings into a final SBIR ranking."""
        ochiai_dir = get_ochiai_dir(project, bug_id, dataset=dataset)
        sbir_dir = get_sbir_dir(project, bug_id, dataset=dataset)

        sbfl_path = ochiai_dir / SBFL_STMT_SUSPS
        blues_path = sbir_dir / BLUES_STMT_SUSPS
        LOGGER.info("Loading SBFL ranking from %s", sbfl_path)
        sbfl_scores = self._load_scores(sbfl_path)
        LOGGER.info("Loading Blues ranking from %s", blues_path)
        blues_scores = self._load_scores(blues_path)

        sbfl_ranks = self._scores_to_ranks(sbfl_scores)
        blues_ranks = self._scores_to_ranks(blues_scores)

        borda_ranking, _ = self._compute_borda_ranking(
            sbfl_ranks,
            blues_ranks,
            sbfl_weight=sbfl_weight,
        )
        final_ranking = borda_ranking

        LOGGER.info(
            "Computed weighted Borda baseline for %d statements (SBFL=%d, Blues=%d, sbfl_weight=%.2f)",
            len(borda_ranking),
            len(sbfl_ranks),
            len(blues_ranks),
            sbfl_weight,
        )

        if use_ce and borda_ranking:
            ce_config = CERunConfig(
                samples_per_iteration=self.ce_config.samples_per_iteration,
                pop_size=(ce_pop_size if ce_pop_size is not None else self.ce_config.pop_size),
                rho=self.ce_config.rho,
                convergence_window=self.ce_config.convergence_window,
                max_iterations=(
                    ce_max_iter if ce_max_iter is not None else self.ce_config.max_iterations
                ),
                crossover_prob=self.ce_config.crossover_prob,
                mutation_prob=self.ce_config.mutation_prob,
                seed=self.ce_config.seed,
                smoothing=self.ce_config.smoothing,
            )

            ce_scope = borda_ranking
            tail: list[str] = []
            if len(borda_ranking) > 500:
                ce_scope = borda_ranking[:500]
                tail = borda_ranking[500:]
                LOGGER.warning(
                    "CE approximation enabled: %d statements detected, "
                    "optimizing top-500 only; remaining %d stay in Borda order",
                    len(borda_ranking),
                    len(tail),
                )

            ce_inputs = [
                self._induce_scope_ranks(sbfl_ranks, ce_scope),
                self._induce_scope_ranks(blues_ranks, ce_scope),
            ]

            optimized = self._run_ce(
                initial_ranking=ce_scope,
                input_rankings=ce_inputs,
                config=ce_config,
            )
            final_ranking = optimized + tail
            LOGGER.info("Using CE-optimized ranking")
        elif use_ce:
            LOGGER.info("Skipping CE because ranking is empty")
        else:
            LOGGER.info("Using Borda baseline ranking (CE disabled)")

        final_scores = self._ranking_to_scores(final_ranking)
        self._write_outputs(sbir_dir, final_scores)
        return final_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate SBFL and Blues rankings into final SBIR ranking",
    )
    parser.add_argument("--project", required=True, help="Defects4J project name")
    parser.add_argument("--bug_id", required=True, help="Defects4J bug id")
    parser.add_argument("--use_ce", action="store_true", help="Enable CE Monte Carlo optimization")
    parser.add_argument(
        "--ce_max_iter", type=int, default=None, help="Override CE maximum iterations"
    )
    parser.add_argument(
        "--ce_pop_size",
        type=int,
        default=None,
        help="Override CE population size (research default is 100)",
    )
    parser.add_argument(
        "--sbfl_weight",
        type=float,
        default=0.5,
        help="SBFL weight in weighted Borda aggregation (Blues weight is 1-sbfl_weight).",
    )
    args: CLIArgs = parser.parse_args(namespace=CLIArgs())

    RAFL().process_project(
        project=args.project,
        bug_id=args.bug_id,
        use_ce=args.use_ce,
        ce_max_iter=args.ce_max_iter,
        ce_pop_size=args.ce_pop_size,
        sbfl_weight=args.sbfl_weight,
    )
