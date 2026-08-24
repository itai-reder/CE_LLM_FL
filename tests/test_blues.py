"""Tests for src.sbir.blues module."""

from __future__ import annotations

from src.sbir.blues import Blues, BluesConfig


class TestNormalize:
    """Tests for score normalisation."""

    def test_basic_normalize(self) -> None:
        scores = [0.0, 0.5, 1.0, 0.25]
        result = Blues._normalize(scores)
        assert result == [0.0, 0.5, 1.0, 0.25]

    def test_all_zero(self) -> None:
        scores = [0.0, 0.0, 0.0]
        result = Blues._normalize(scores)
        assert result == [0.0, 0.0, 0.0]

    def test_empty(self) -> None:
        assert Blues._normalize([]) == []

    def test_scales_to_one(self) -> None:
        scores = [2.0, 4.0, 6.0]
        result = Blues._normalize(scores)
        assert result[2] == 1.0
        assert abs(result[0] - 2.0 / 6.0) < 1e-9


class TestGroupByClass:
    """Tests for grouping statements by class FQN."""

    def test_groups_correctly(self) -> None:
        docs = [
            {"stmt_id": "A#1", "class_fqn": "A"},
            {"stmt_id": "A#2", "class_fqn": "A"},
            {"stmt_id": "B#1", "class_fqn": "B"},
        ]
        scores = [0.8, 0.2, 0.5]
        by_class = Blues._group_by_class(docs, scores)
        assert len(by_class) == 2
        assert len(by_class["A"]) == 2
        assert len(by_class["B"]) == 1
        # A should be sorted descending by score
        assert by_class["A"][0][0] == "A#1"
        assert by_class["A"][1][0] == "A#2"


class TestConfigScores:
    """Tests for per-configuration scoring."""

    def test_high_scoring(self) -> None:
        by_class = {
            "A": [("A#1", 0.9), ("A#2", 0.5), ("A#3", 0.1)],
            "B": [("B#1", 0.7)],
        }
        config = BluesConfig(name="m1_high", m=1, scoring="high")
        scores = Blues._config_scores(by_class, config)
        # m=1: only top statement per class, scored with max_score
        assert scores["A#1"] == 0.9
        assert "A#2" not in scores or scores.get("A#2", 0.0) == 0.0
        assert scores["B#1"] == 0.7

    def test_wted_scoring(self) -> None:
        by_class = {
            "A": [("A#1", 0.8), ("A#2", 0.4)],
        }
        config = BluesConfig(name="mall_wted", m=None, scoring="wted")
        scores = Blues._config_scores(by_class, config)
        # weighted_sum = 0.8/1 + 0.4/2 = 0.8 + 0.2 = 1.0
        expected_sum = 0.8 / 1 + 0.4 / 2
        assert abs(scores["A#1"] - expected_sum) < 1e-9
        assert abs(scores["A#2"] - expected_sum) < 1e-9

    def test_m_limits_statements(self) -> None:
        by_class = {
            "A": [("A#1", 0.9), ("A#2", 0.5), ("A#3", 0.1)],
        }
        config = BluesConfig(name="m2_high", m=2, scoring="high")
        scores = Blues._config_scores(by_class, config)
        assert "A#1" in scores
        assert "A#2" in scores
        assert "A#3" not in scores


class TestConsensusMax:
    """Tests for max-consensus across configurations."""

    def test_takes_max(self) -> None:
        stmt_ids = ["A#1", "A#2", "B#1"]
        maps = [
            {"A#1": 0.3, "A#2": 0.9, "B#1": 0.1},
            {"A#1": 0.7, "A#2": 0.4, "B#1": 0.5},
        ]
        result = Blues._consensus_max(stmt_ids, maps)
        assert result["A#1"] == 0.7
        assert result["A#2"] == 0.9
        assert result["B#1"] == 0.5


class TestTopKFilter:
    """Tests for top-k filtering."""

    def test_keeps_top_k(self) -> None:
        scores = {"A#1": 0.9, "A#2": 0.5, "B#1": 0.3, "C#1": 0.1}
        result = Blues._apply_top_k_filter(scores, top_k=2)
        assert result["A#1"] == 0.9
        assert result["A#2"] == 0.5
        assert result["B#1"] == 0.0
        assert result["C#1"] == 0.0

    def test_top_k_larger_than_set(self) -> None:
        scores = {"A#1": 0.9, "A#2": 0.5}
        result = Blues._apply_top_k_filter(scores, top_k=100)
        assert result == scores
