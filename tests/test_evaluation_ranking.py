"""Tests for src.evaluation.ranking."""

from __future__ import annotations

import pytest

from src.evaluation.ranking import (
    append_unranked_with_uniform_rank,
    assign_average_ranks,
    score_universe,
)


class TestAssignAverageRanks:
    def test_distinct_scores(self) -> None:
        ranks = assign_average_ranks([("a", 3.0), ("b", 1.0), ("c", 2.0)])
        assert ranks == {"a": 1.0, "c": 2.0, "b": 3.0}

    def test_four_way_tie_at_positions_2_to_5(self) -> None:
        # Top item alone at 1; next 4 share rank 3.5; last alone at 6.
        ranks = assign_average_ranks(
            [
                ("top", 10.0),
                ("t1", 5.0),
                ("t2", 5.0),
                ("t3", 5.0),
                ("t4", 5.0),
                ("last", 1.0),
            ]
        )
        assert ranks["top"] == 1.0
        for key in ("t1", "t2", "t3", "t4"):
            assert ranks[key] == 3.5
        assert ranks["last"] == 6.0

    def test_all_tied(self) -> None:
        ranks = assign_average_ranks([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        # mean of 1,2,3 = 2
        assert ranks == {"a": 2.0, "b": 2.0, "c": 2.0}

    def test_empty(self) -> None:
        assert assign_average_ranks([]) == {}

    def test_tiebreak_stable_order(self) -> None:
        # Tied scores resolved by tiebreak; ranks are identical but the
        # ordering through the loop is deterministic.
        ranks = assign_average_ranks(
            [("zeta", 1.0), ("alpha", 1.0)],
            tiebreak=lambda k: k,
        )
        assert ranks == {"alpha": 1.5, "zeta": 1.5}


class TestAppendUnrankedWithUniformRank:
    def test_5_ranked_of_200_unranked_mean_103(self) -> None:
        ranked = {f"m{i}": float(i) for i in range(1, 6)}
        universe = list(ranked.keys()) + [f"u{i}" for i in range(195)]
        result = append_unranked_with_uniform_rank(ranked, universe)
        # Positions 6..200 → mean = (6+200)/2 = 103
        unranked_values = {result[f"u{i}"] for i in range(195)}
        assert unranked_values == {103.0}
        # Ranked members untouched
        for i in range(1, 6):
            assert result[f"m{i}"] == float(i)

    def test_universe_equals_ranked(self) -> None:
        ranked = {"a": 1.0, "b": 2.0}
        result = append_unranked_with_uniform_rank(ranked, ["a", "b"])
        assert result == ranked
        assert result is not ranked  # shallow copy

    def test_input_not_mutated(self) -> None:
        ranked = {"a": 1.0}
        append_unranked_with_uniform_rank(ranked, ["a", "b", "c"])
        assert ranked == {"a": 1.0}


class TestScoreUniverse:
    def test_combined_filter_rank_and_fill(self) -> None:
        # Faulty methods scored; universe has 1 extra unranked candidate.
        scored = {"a": 5.0, "b": 5.0, "c": 1.0}
        universe = ["a", "b", "c", "d"]
        ranks = score_universe(scored, universe)
        # a, b tied at positions 1,2 → rank 1.5; c at position 3; d uniform at 4.
        assert ranks["a"] == 1.5
        assert ranks["b"] == 1.5
        assert ranks["c"] == 3.0
        assert ranks["d"] == 4.0

    def test_scored_outside_universe_filtered_out(self) -> None:
        scored = {"a": 5.0, "z": 99.0}  # z not in universe
        ranks = score_universe(scored, ["a", "b"])
        assert ranks == {"a": 1.0, "b": 2.0}


class TestRanksAreOneBased:
    """Sanity: rank 1 means first place; large universe sizes are floats."""

    def test_smallest_rank_is_one(self) -> None:
        ranks = assign_average_ranks([("only", 1.0)])
        assert ranks == {"only": 1.0}

    def test_no_zero_ranks(self) -> None:
        ranks = assign_average_ranks([(f"k{i}", float(i)) for i in range(10)])
        assert min(ranks.values()) == pytest.approx(1.0)
