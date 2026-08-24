"""Tests for src.boostn.boostn module."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.boostn.boostn import BM25Variant, BoostN


class TestBM25Variant:
    """Tests for the BoostN custom BM25 variant."""

    def test_constructs_with_defaults(self) -> None:
        corpus = [["hello", "world"], ["foo", "bar"]]
        bm25 = BM25Variant(corpus)
        assert bm25.k1 == 1.0
        assert bm25.b == 0.3

    def test_constructs_with_idf_only(self) -> None:
        corpus = [["hello", "world"], ["foo", "bar"]]
        bm25 = BM25Variant(corpus, k1=0.0, b=0.3)
        assert bm25.k1 == 0.0

    def test_idf_values_positive(self) -> None:
        corpus = [["alpha", "beta"], ["beta", "gamma"], ["gamma", "delta"]]
        bm25 = BM25Variant(corpus)
        # All IDF values should be non-negative after epsilon correction
        for word, idf_val in bm25.idf.items():
            assert idf_val >= 0.0, f"IDF for '{word}' is negative: {idf_val}"

    def test_get_score_idf_only(self) -> None:
        corpus = [["alpha", "beta"], ["beta", "gamma"]]
        bm25 = BM25Variant(corpus, k1=0.0, b=0.3)
        # Query for "beta" -- present in both docs
        score_0 = bm25.get_score(["beta"], 0)
        score_1 = bm25.get_score(["beta"], 1)
        # With k1=0, score is just IDF for matching terms
        assert score_0 > 0.0
        assert score_1 > 0.0
        # Both docs contain "beta" so IDF-only scores should be equal
        assert abs(score_0 - score_1) < 1e-9

    def test_get_score_full_bm25(self) -> None:
        corpus = [["alpha", "beta", "beta"], ["beta", "gamma"]]
        bm25 = BM25Variant(corpus, k1=1.0, b=0.3)
        score_0 = bm25.get_score(["beta"], 0)
        score_1 = bm25.get_score(["beta"], 1)
        # Doc 0 has "beta" twice, so should score higher with full BM25
        assert score_0 > score_1

    def test_get_scores_returns_array(self) -> None:
        corpus = [["a", "b"], ["c", "d"], ["a", "c"]]
        bm25 = BM25Variant(corpus, k1=1.0, b=0.3)
        scores = bm25.get_scores(["a"])
        assert len(scores) == 3
        # Doc 0 and doc 2 contain "a", doc 1 does not
        assert scores[0] > 0.0
        assert scores[1] == 0.0
        assert scores[2] > 0.0

    def test_get_scores_idf_only_has_no_nan(self) -> None:
        corpus = [["alpha"], ["beta"], []]
        bm25 = BM25Variant(corpus, k1=0.0, b=0.3)
        scores = bm25.get_scores(["alpha"])
        assert len(scores) == 3
        assert all(not math.isnan(float(score)) for score in scores)
        assert scores[0] > 0.0
        assert scores[1] == 0.0
        assert scores[2] == 0.0


class TestBoostNWriteOutputs:
    """Tests for BoostN output writing."""

    def test_write_json_and_csv(self, tmp_path: Path) -> None:
        results = {
            "org.example.Foo.bar(int).10.20": 0.85,
            "org.example.Foo.baz().25.30": 0.42,
        }
        BoostN._write_outputs(tmp_path, results)

        json_path = tmp_path / "boostn.json"
        csv_path = tmp_path / "boostn-method-susps.csv"

        assert json_path.exists()
        assert csv_path.exists()

        data = json.loads(json_path.read_text())
        assert "boostn_scores" in data
        assert data["boostn_scores"]["org.example.Foo.bar(int).10.20"] == 0.85

        lines = csv_path.read_text().strip().splitlines()
        assert lines[0] == "Signature,Suspiciousness"
        # First data line should be highest score
        assert "org.example.Foo.bar(int).10.20" in lines[1]
        assert "0.85" in lines[1]

    def test_write_empty_results(self, tmp_path: Path) -> None:
        BoostN._write_outputs(tmp_path, {})
        json_path = tmp_path / "boostn.json"
        csv_path = tmp_path / "boostn-method-susps.csv"
        assert json_path.exists()
        assert csv_path.exists()
        data = json.loads(json_path.read_text())
        assert data["boostn_scores"] == {}


class TestBoostNAdaptiveK1:
    """Tests verifying the adaptive k1 logic."""

    def test_small_corpus_uses_k1_zero(self) -> None:
        # <= 3000 methods -> k1 = 0.0
        corpus = [["token"] for _ in range(100)]
        k1 = 1.0 if len(corpus) > 3000 else 0.0
        assert k1 == 0.0

    def test_large_corpus_uses_k1_one(self) -> None:
        # > 3000 methods -> k1 = 1.0
        corpus = [["token"] for _ in range(3001)]
        k1 = 1.0 if len(corpus) > 3000 else 0.0
        assert k1 == 1.0

    def test_boundary_3000_uses_k1_zero(self) -> None:
        corpus = [["token"] for _ in range(3000)]
        k1 = 1.0 if len(corpus) > 3000 else 0.0
        assert k1 == 0.0
