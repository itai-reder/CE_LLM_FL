"""Tests for src.evaluation.per_bug."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.common.method_entity import MethodEntity
from src.evaluation.per_bug import (
    PER_BUG_HEADERS,
    Metrics,
    evaluate_method,
    write_per_bug_csvs,
)
from src.evaluation.sources import Usage


def _entities() -> list[MethodEntity]:
    return [
        MethodEntity(
            corpus_id=f"pkg$Cls.m{i}()",
            class_fqn_dotted="pkg.Cls",
            path="pkg/Cls.java",
            start_line=i * 10,
            end_line=i * 10 + 5,
        )
        for i in range(1, 11)
    ]


# ---------------------------------------------------------------------------
# evaluate_method — algorithmic behaviour
# ---------------------------------------------------------------------------


class TestEvaluateMethod:
    def test_perfect_ranking_first_rank_one(self) -> None:
        ents = _entities()
        universe = set(ents)
        faulty = {ents[0]}  # m1
        scored = {ents[0]: 1.0}  # only m1 ranked, others uniform
        metrics = evaluate_method("Ochiai", scored, universe, faulty)
        assert metrics.FR == 1.0
        assert metrics.AR == 1.0
        assert metrics.Top1 == 1
        assert metrics.WE == 0.0

    def test_empty_universe_returns_zeros(self) -> None:
        metrics = evaluate_method("Foo", {}, set(), {_entities()[0]})
        assert metrics.FR is None
        assert metrics.AR is None
        assert metrics.WE is None
        assert metrics.Top1 == metrics.Top5 == 0

    def test_no_faulty_in_universe(self) -> None:
        ents = _entities()
        universe = set(ents[:3])
        # faulty member is outside the universe
        faulty = {ents[5]}
        metrics = evaluate_method("Foo", {ents[0]: 1.0}, universe, faulty)
        assert metrics.FR is None
        assert metrics.AR is None
        assert metrics.WE is None
        assert metrics.Top1 == 0


# ---------------------------------------------------------------------------
# Metrics.to_row — CSV cell rendering
# ---------------------------------------------------------------------------


class TestMetricsToRow:
    def test_blank_cells_for_none(self) -> None:
        m = Metrics("M", None, None, 0, 0, 0, 0, 0, None)
        # Tail token columns default to 0/0/0 and a blank cost cell.
        assert m.to_row() == ["M", "", "", "0", "0", "0", "0", "0", "", "0", "0", "0", ""]

    def test_numeric_cells_rendered(self) -> None:
        m = Metrics("M", 1.0, 2.5, 1, 1, 1, 1, 1, 0.04)
        row = m.to_row()
        assert row[0] == "M"
        assert row[1] == "1.0"
        assert row[2] == "2.5"
        assert row[3:8] == ["1", "1", "1", "1", "1"]
        assert row[8] == "0.04"
        # Token defaults appear in the new tail.
        assert row[9:12] == ["0", "0", "0"]
        assert row[12] == ""

    def test_token_and_cost_columns(self) -> None:
        m = Metrics(
            "Agent4LR-X",
            1.0,
            1.0,
            1,
            1,
            1,
            1,
            1,
            0.0,
            InputTokens=12_345,
            CachedTokens=678,
            OutputTokens=910,
            CostUSD=0.012345,
        )
        row = m.to_row()
        assert row[9:13] == ["12345", "678", "910", "0.012345"]


class TestEvaluateMethodTokens:
    def test_usage_passes_through_to_metrics(self) -> None:
        ents = _entities()
        usage = Usage(input_tokens=1000, cached_tokens=200, output_tokens=50, cost_usd=0.001)
        metrics = evaluate_method("Agent4LR-Foo", {ents[0]: 1.0}, set(ents), {ents[0]}, usage=usage)
        assert metrics.InputTokens == 1000
        assert metrics.CachedTokens == 200
        assert metrics.OutputTokens == 50
        assert metrics.CostUSD == pytest.approx(0.001)

    def test_no_usage_keeps_defaults(self) -> None:
        ents = _entities()
        metrics = evaluate_method("Ochiai", {ents[0]: 1.0}, set(ents), {ents[0]})
        assert metrics.InputTokens == 0
        assert metrics.CachedTokens == 0
        assert metrics.OutputTokens == 0
        assert metrics.CostUSD is None

    def test_usage_preserved_on_empty_universe(self) -> None:
        usage = Usage(input_tokens=10, cached_tokens=0, output_tokens=5, cost_usd=0.0001)
        metrics = evaluate_method("Agent4LR-Foo", {}, set(), set(), usage=usage)
        assert metrics.FR is None
        assert metrics.InputTokens == 10
        assert metrics.CostUSD == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# write_per_bug_csvs
# ---------------------------------------------------------------------------


class TestWritePerBugCsvs:
    def test_writes_all_four_files_with_headers(self, tmp_path: Path) -> None:
        m = Metrics("Ochiai", 1.0, 1.0, 1, 1, 1, 1, 1, 0.0)
        results = {
            "baselines": [m],
            "baselines_first": [],
            "flexfl": [],
            "flexfl_first": [],
        }
        paths = write_per_bug_csvs(tmp_path, results)
        assert set(paths.keys()) == {
            "baselines",
            "baselines_first",
            "flexfl",
            "flexfl_first",
        }
        for name, path in paths.items():
            assert path.exists()
            rows = list(csv.reader(path.open(encoding="utf-8")))
            assert rows[0] == list(PER_BUG_HEADERS)
            if name == "baselines":
                assert len(rows) == 2
                assert rows[1][0] == "Ochiai"
            else:
                assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Real Lang/1 — end-to-end (skipped if data missing)
# ---------------------------------------------------------------------------


from src.common.config import get_processed_dir  # noqa: E402

REAL_PROCESSED = get_processed_dir("Lang", 1)
REAL_TOP20 = REAL_PROCESSED / "FlexFL" / "SR" / "rankings" / "top20" / "llama3.1_8b.txt"


@pytest.mark.skipif(
    not (REAL_PROCESSED / "method_signatures.csv").exists()
    or not (REAL_PROCESSED / "FlexFL" / "SR" / "rankings" / "ochiai.csv").exists()
    or not REAL_TOP20.exists(),
    reason="Lang/1 baseline rankings or SR top-20 not available",
)
class TestRealLang1:
    def test_evaluate_bug_baselines_hit_top1(self) -> None:
        from src.evaluation.per_bug import evaluate_bug

        results = evaluate_bug("Lang", 1)
        assert results["baselines"], "expected at least one baseline row"
        # Lang/1 has createNumber as the faulty method, ranked #1 by all baselines.
        baseline_methods = {m.Method: m for m in results["baselines"]}
        for name in ("Ochiai", "SBIR", "BoostN"):
            assert name in baseline_methods
            assert baseline_methods[name].FR == 1.0
            assert baseline_methods[name].Top1 == 1


# ---------------------------------------------------------------------------
# Real Jsoup/15 — pin the universe-from-top20 fix
# ---------------------------------------------------------------------------


REAL_JSOUP15 = get_processed_dir("Jsoup", 15)
JSOUP15_TOP20 = REAL_JSOUP15 / "FlexFL" / "SR" / "rankings" / "top20" / "llama3.1_8b.txt"


@pytest.mark.skipif(
    not (REAL_JSOUP15 / "method_signatures.csv").exists() or not JSOUP15_TOP20.exists(),
    reason="Jsoup/15 method_signatures.csv or SR top-20 not available",
)
class TestRealJsoup15:
    """Jsoup/15 is the canonical case for the SR-top20 universe fix.

    GZoltar coverage didn't capture the faulty method, but the SR top-20 did.
    Under the old (coverage) universe, FR/AR were blank for every row.
    Under the new (top-20) universe, ``flexfl`` / ``baselines`` rows are
    non-blank (faulty method is in the candidates), and the ``*_first``
    rows stay blank because ``faults_first.csv`` is empty for this bug.
    """

    def test_baselines_get_non_blank_fr_under_top20_universe(self) -> None:
        from src.evaluation.per_bug import evaluate_bug

        results = evaluate_bug("Jsoup", 15)
        # All four slots should have rows now (one universe, two faulty sets).
        for slot in ("baselines", "baselines_first", "flexfl", "flexfl_first"):
            assert results[slot], f"{slot}: expected rows under top-20 universe"

        baseline_all = {m.Method: m for m in results["baselines"]}
        for name in ("Ochiai", "SBIR", "BoostN"):
            assert name in baseline_all
            assert baseline_all[name].FR is not None, (
                f"baselines[{name}].FR should be populated for Jsoup/15"
            )

    def test_first_fault_universe_blank_when_faults_first_empty(self) -> None:
        from src.evaluation.per_bug import evaluate_bug

        results = evaluate_bug("Jsoup", 15)
        # faults_first.csv is empty for Jsoup/15 → faulty_first ∩ universe = ∅
        # → FR/AR/WE should be None for every row in *_first.
        for m in results["baselines_first"]:
            assert m.FR is None and m.AR is None
        for m in results["flexfl_first"]:
            assert m.FR is None and m.AR is None
