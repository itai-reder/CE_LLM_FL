"""Tests for src.sbir.rafl module."""

from __future__ import annotations

from pathlib import Path

from src.sbir.rafl import RAFL


class TestScoresToRanks:
    """Tests for converting scores to 1-indexed ranks."""

    def test_distinct_scores(self) -> None:
        scores = {"A#1": 0.9, "B#1": 0.5, "C#1": 0.1}
        ranks = RAFL._scores_to_ranks(scores)
        assert ranks["A#1"] == 1
        assert ranks["B#1"] == 2
        assert ranks["C#1"] == 3

    def test_tied_scores(self) -> None:
        scores = {"A#1": 0.9, "B#1": 0.9, "C#1": 0.5}
        ranks = RAFL._scores_to_ranks(scores)
        # Both A#1 and B#1 should share rank 1
        assert ranks["A#1"] == 1
        assert ranks["B#1"] == 1
        assert ranks["C#1"] == 3

    def test_empty(self) -> None:
        assert RAFL._scores_to_ranks({}) == {}

    def test_single_element(self) -> None:
        ranks = RAFL._scores_to_ranks({"A#1": 1.0})
        assert ranks["A#1"] == 1


class TestComputeBordaRanking:
    """Tests for Borda-count aggregation."""

    def test_simple_borda(self) -> None:
        sbfl_ranks = {"A#1": 1, "B#1": 2, "C#1": 3}
        blues_ranks = {"A#1": 3, "B#1": 1, "C#1": 2}
        ranking, borda_scores = RAFL._compute_borda_ranking(sbfl_ranks, blues_ranks)
        # default weights: sbfl=0.5, blues=0.5, with n1=n2=3
        # A#1: 0.5*(3) + 0.5*(1) = 2.0
        # B#1: 0.5*(2) + 0.5*(3) = 2.5
        # C#1: 0.5*(1) + 0.5*(2) = 1.5
        assert abs(borda_scores["A#1"] - 2.0) < 1e-9
        assert abs(borda_scores["B#1"] - 2.5) < 1e-9
        assert abs(borda_scores["C#1"] - 1.5) < 1e-9
        # B#1 should be first (highest borda score)
        assert ranking[0] == "B#1"
        assert ranking[1] == "A#1"
        assert ranking[2] == "C#1"

    def test_disjoint_sets(self) -> None:
        sbfl_ranks = {"A#1": 1}
        blues_ranks = {"B#1": 1}
        ranking, borda_scores = RAFL._compute_borda_ranking(sbfl_ranks, blues_ranks)
        assert len(ranking) == 2
        # default weights 0.5/0.5 keep the two statements tied at 0.5
        # Both have borda=1, so order is alphabetical (tiebreaker)
        assert abs(borda_scores["A#1"] - 0.5) < 1e-9
        assert abs(borda_scores["B#1"] - 0.5) < 1e-9

    def test_weighted_borda_prefers_sbfl_when_weight_is_one(self) -> None:
        sbfl_ranks = {"A#1": 1, "B#1": 2}
        blues_ranks = {"A#1": 2, "B#1": 1}
        ranking, _ = RAFL._compute_borda_ranking(
            sbfl_ranks,
            blues_ranks,
            sbfl_weight=1.0,
        )
        assert ranking == ["A#1", "B#1"]

    def test_weighted_borda_prefers_blues_when_weight_is_zero(self) -> None:
        sbfl_ranks = {"A#1": 1, "B#1": 2}
        blues_ranks = {"A#1": 2, "B#1": 1}
        ranking, _ = RAFL._compute_borda_ranking(
            sbfl_ranks,
            blues_ranks,
            sbfl_weight=0.0,
        )
        assert ranking == ["B#1", "A#1"]


class TestRankingToScores:
    """Tests for converting ranking to normalised scores."""

    def test_simple_scores(self) -> None:
        ranking = ["A#1", "B#1", "C#1"]
        scores = RAFL._ranking_to_scores(ranking)
        assert scores["A#1"] == 1.0  # (3-1+1)/3
        assert abs(scores["B#1"] - 2 / 3) < 1e-9  # (3-2+1)/3
        assert abs(scores["C#1"] - 1 / 3) < 1e-9  # (3-3+1)/3

    def test_empty(self) -> None:
        assert RAFL._ranking_to_scores([]) == {}

    def test_single(self) -> None:
        scores = RAFL._ranking_to_scores(["A#1"])
        assert scores["A#1"] == 1.0


class TestLoadScores:
    """Tests for loading ranking CSV files."""

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ranking.csv"
        csv_file.write_text("Statement,Suspiciousness\nA#1,0.9\nB#1,0.5\nC#1,0.1\n")
        scores = RAFL._load_scores(csv_file)
        assert len(scores) == 3
        assert scores["A#1"] == 0.9

    def test_missing_file_raises(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            RAFL._load_scores(Path("/tmp/nonexistent_rafl.csv"))


class TestSpearmanFootrule:
    """Tests for the Spearman footrule distance."""

    def test_identical_ranking(self) -> None:
        candidate = ["A", "B", "C"]
        input_ranks = {"A": 1, "B": 2, "C": 3}
        assert RAFL._spearman_footrule_distance(candidate, input_ranks) == 0

    def test_reversed_ranking(self) -> None:
        candidate = ["C", "B", "A"]
        input_ranks = {"A": 1, "B": 2, "C": 3}
        # C at pos 1 vs rank 3: |1-3|=2
        # B at pos 2 vs rank 2: |2-2|=0
        # A at pos 3 vs rank 1: |3-1|=2
        assert RAFL._spearman_footrule_distance(candidate, input_ranks) == 4


class TestRankingToPositionMap:
    """Tests for converting a ranking list to position map."""

    def test_basic(self) -> None:
        pos = RAFL._ranking_to_position_map(["A", "B", "C"])
        assert pos == {"A": 1, "B": 2, "C": 3}

    def test_empty(self) -> None:
        assert RAFL._ranking_to_position_map([]) == {}


class TestInduceScopeRanks:
    """Tests for inducing rankings within a scope."""

    def test_subset(self) -> None:
        source_ranks = {"A": 1, "B": 2, "C": 3, "D": 4}
        scope = ["B", "D"]
        induced = RAFL._induce_scope_ranks(source_ranks, scope)
        assert induced["B"] == 1
        assert induced["D"] == 2

    def test_missing_from_source(self) -> None:
        source_ranks = {"A": 1}
        scope = ["A", "Z"]
        induced = RAFL._induce_scope_ranks(source_ranks, scope)
        assert induced["A"] == 1
        # Z is not in source_ranks, gets worst_rank = len(present) + 1 = 2
        assert induced["Z"] == 2
