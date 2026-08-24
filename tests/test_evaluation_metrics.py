"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    first_rank,
    mean_rank,
    top_k,
    wasted_effort,
)


class TestFirstRank:
    def test_returns_lowest_rank_of_faulty(self) -> None:
        ranks = {"a": 3.0, "b": 1.0, "c": 5.0}
        assert first_rank(ranks, {"b", "c"}) == 1.0

    def test_none_when_no_faulty_in_universe(self) -> None:
        assert first_rank({"a": 1.0}, {"missing"}) is None


class TestMeanRank:
    def test_arithmetic_mean(self) -> None:
        ranks = {"a": 2.0, "b": 4.0, "c": 6.0}
        assert mean_rank(ranks, {"a", "b", "c"}) == pytest.approx(4.0)

    def test_partial_membership(self) -> None:
        # Only b and c are faulty; ignore a's rank.
        ranks = {"a": 10.0, "b": 2.0, "c": 4.0}
        assert mean_rank(ranks, {"b", "c"}) == pytest.approx(3.0)

    def test_none_when_no_hit(self) -> None:
        assert mean_rank({"a": 1.0}, set()) is None


class TestTopK:
    def test_hit(self) -> None:
        assert top_k(3.0, 5) == 1

    def test_miss(self) -> None:
        assert top_k(6.0, 5) == 0

    def test_boundary_inclusive(self) -> None:
        assert top_k(5.0, 5) == 1

    def test_tied_average_at_boundary(self) -> None:
        # Mean rank 4.5 from positions 4 and 5 still counts as top-5.
        assert top_k(4.5, 5) == 1

    def test_none_rank_returns_zero(self) -> None:
        assert top_k(None, 5) == 0


class TestWastedEffort:
    def test_user_example_20_healthy_faults_at_1_3_10(self) -> None:
        # User spec: ranks 1,3,10; |F|=3; n_healthy=20.
        # WE = (10 - 3) / 20 = 0.35
        ranks = {f"h{i}": float(i) for i in range(1, 24)}  # 23 candidates
        # Ranks 1, 3, 10 → keys h1, h3, h10
        faulty = {"h1", "h3", "h10"}
        # universe_size = 23 (3 faulty + 20 healthy)
        assert wasted_effort(ranks, faulty, 23) == pytest.approx(0.35)

    def test_single_fault_at_top(self) -> None:
        # Fault at rank 1; |F|=1; n_healthy=N-1. WE = (1-1)/(N-1) = 0.
        ranks = {f"m{i}": float(i) for i in range(1, 6)}
        assert wasted_effort(ranks, {"m1"}, 5) == pytest.approx(0.0)

    def test_single_fault_at_end(self) -> None:
        # Fault at rank N; |F|=1; n_healthy=N-1. WE = (N-1)/(N-1) = 1.
        ranks = {f"m{i}": float(i) for i in range(1, 6)}
        assert wasted_effort(ranks, {"m5"}, 5) == pytest.approx(1.0)

    def test_no_faulty_returns_none(self) -> None:
        ranks = {"a": 1.0, "b": 2.0}
        assert wasted_effort(ranks, set(), 2) is None

    def test_universe_entirely_faulty_returns_none(self) -> None:
        ranks = {"a": 1.0, "b": 2.0}
        assert wasted_effort(ranks, {"a", "b"}, 2) is None

    def test_negative_we_when_faults_cluster_at_rank_one(self) -> None:
        # All 2 faults tied at rank 1 (avg of positions 1,2); |F|=2;
        # n_healthy = 5-2 = 3. WE = (1 - 2) / 3 = -1/3. Negative is a
        # faithful signal under tied-rank semantics — see docstring.
        ranks = {"f1": 1.0, "f2": 1.0, "h1": 3.0, "h2": 4.0, "h3": 5.0}
        # Faults both have rank 1.0 (mean of 1,2).
        # In a real flow they'd actually be 1.5; force-set for clarity.
        ranks["f1"] = 1.0
        ranks["f2"] = 1.0
        result = wasted_effort(ranks, {"f1", "f2"}, 5)
        assert result is not None
        assert result < 0
