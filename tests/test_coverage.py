"""Tests for src.common.coverage."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.common.config import get_processed_dir
from src.common.coverage import (
    _parse_spectra_id,
    compute_universe,
    covered_columns,
    covered_method_entities,
    parse_failing_test_names,
    parse_trigger_test_names,
    read_test_row_indices,
)
from src.common.method_entity import MethodEntity

# ---------------------------------------------------------------------------
# parse_trigger_test_names
# ---------------------------------------------------------------------------


class TestParseTriggerTestNames:
    def test_preserves_order_and_converts_separator(self, tmp_path: Path) -> None:
        path = tmp_path / "trigger_tests"
        path.write_text("--- org.example.Foo::testA\nstuff\n--- org.example.Bar::testB\nstuff\n")
        assert parse_trigger_test_names(path) == [
            "org.example.Foo#testA",
            "org.example.Bar#testB",
        ]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert parse_trigger_test_names(tmp_path / "missing") == []

    def test_skips_malformed_headers(self, tmp_path: Path) -> None:
        path = tmp_path / "trigger_tests"
        path.write_text("--- noseparator\n--- pkg.Cls::ok\n")
        assert parse_trigger_test_names(path) == ["pkg.Cls#ok"]


# ---------------------------------------------------------------------------
# read_test_row_indices
# ---------------------------------------------------------------------------


class TestReadTestRowIndices:
    def test_returns_row_indices_in_order(self, tmp_path: Path) -> None:
        tests_csv = tmp_path / "tests.csv"
        tests_csv.write_text(
            textwrap.dedent(
                """\
                name,outcome,runtime,stacktrace
                a.A#x,PASS,1,
                b.B#y,FAIL,2,boom
                c.C#z,PASS,3,
                """
            )
        )
        assert read_test_row_indices(tests_csv, ["c.C#z", "a.A#x"]) == [2, 0]

    def test_raises_on_missing_name(self, tmp_path: Path) -> None:
        tests_csv = tmp_path / "tests.csv"
        tests_csv.write_text("name,outcome,runtime,stacktrace\na.A#x,PASS,1,\n")
        with pytest.raises(KeyError):
            read_test_row_indices(tests_csv, ["nope#nada"])


# ---------------------------------------------------------------------------
# covered_columns
# ---------------------------------------------------------------------------


class TestCoveredColumns:
    def test_strips_outcome_marker_and_returns_ones(self, tmp_path: Path) -> None:
        matrix = tmp_path / "matrix.txt"
        matrix.write_text("1 0 1 0 +\n0 1 1 1 -\n")
        assert covered_columns(matrix, 0) == {0, 2}
        assert covered_columns(matrix, 1) == {1, 2, 3}

    def test_raises_on_missing_row(self, tmp_path: Path) -> None:
        matrix = tmp_path / "matrix.txt"
        matrix.write_text("1 0 +\n")
        with pytest.raises(IndexError):
            covered_columns(matrix, 5)


# ---------------------------------------------------------------------------
# _parse_spectra_id
# ---------------------------------------------------------------------------


class TestParseSpectraId:
    def test_simple(self) -> None:
        assert _parse_spectra_id("pkg.sub$Cls#method(int):42") == ("pkg.sub.Cls", 42)

    def test_with_qualified_params(self) -> None:
        assert _parse_spectra_id("a.b$C#m(java.lang.String,int):10") == ("a.b.C", 10)

    def test_inner_class(self) -> None:
        assert _parse_spectra_id("a.b$Outer$Inner#m():3") == ("a.b.Outer.Inner", 3)

    def test_init_and_clinit(self) -> None:
        assert _parse_spectra_id("a$Cls#<clinit>():5") == ("a.Cls", 5)
        assert _parse_spectra_id("a$Cls#Cls():7") == ("a.Cls", 7)

    def test_returns_none_on_malformed(self) -> None:
        assert _parse_spectra_id("garbage") is None
        assert _parse_spectra_id("a$Cls:m:notanumber") is None

    def test_python_module_level_function(self) -> None:
        # FauxPy/Python shape: <module>$<func>(params):line (no '#').
        assert _parse_spectra_id("youtube_dl.utils$parse_duration(s):1828") == (
            "youtube_dl.utils",
            1828,
        )

    def test_python_class_method(self) -> None:
        assert _parse_spectra_id(
            "youtube_dl.extractor.common$InfoExtractor._parse_mpd_formats(self,mpd_doc):1753"
        ) == ("youtube_dl.extractor.common.InfoExtractor", 1753)

    def test_python_nested_class(self) -> None:
        assert _parse_spectra_id("pkg.mod$Outer.Inner.deep(cls,x,y):12") == (
            "pkg.mod.Outer.Inner",
            12,
        )


# ---------------------------------------------------------------------------
# covered_method_entities
# ---------------------------------------------------------------------------


def _make_entities() -> list[MethodEntity]:
    return [
        MethodEntity(
            corpus_id="a.b$Cls.foo(int)",
            class_fqn_dotted="a.b.Cls",
            path="a/b/Cls.java",
            start_line=10,
            end_line=20,
        ),
        MethodEntity(
            corpus_id="a.b$Cls.bar(String)",
            class_fqn_dotted="a.b.Cls",
            path="a/b/Cls.java",
            start_line=25,
            end_line=40,
        ),
    ]


class TestCoveredMethodEntities:
    def test_maps_columns_to_entities_via_line_index(self, tmp_path: Path) -> None:
        entities = _make_entities()
        spectra = tmp_path / "spectra.csv"
        spectra.write_text(
            textwrap.dedent(
                """\
                name
                a.b$Cls#foo(int):12
                a.b$Cls#bar(java.lang.String):30
                a.b$Cls#bar(java.lang.String):31
                a.b$Other#thing():99
                """
            )
        )
        # Cover column 0 (foo line 12) and column 2 (bar line 31).
        result = covered_method_entities(spectra, {0, 2}, entities)
        ids = {e.corpus_id for e in result}
        assert ids == {"a.b$Cls.foo(int)", "a.b$Cls.bar(String)"}

    def test_empty_column_set_returns_empty(self, tmp_path: Path) -> None:
        spectra = tmp_path / "spectra.csv"
        spectra.write_text("name\na$C#m():1\n")
        assert covered_method_entities(spectra, set(), _make_entities()) == set()


# ---------------------------------------------------------------------------
# compute_universe — uses the real Lang/1 fixture data
# ---------------------------------------------------------------------------


REAL_PROCESSED = get_processed_dir("Lang", 1)


@pytest.mark.skipif(
    not (REAL_PROCESSED / "trigger_tests").exists()
    or not (REAL_PROCESSED / "method_signatures.csv").exists(),
    reason="Lang/1 processed data not available",
)
class TestComputeUniverseLang1:
    def _entities(self) -> list[MethodEntity]:
        from src.common.method_entity import load_method_entities

        return load_method_entities(REAL_PROCESSED)

    def test_first_universe_contains_create_number(self) -> None:
        entities = self._entities()
        universe = compute_universe(REAL_PROCESSED, entities, first_only=True)
        ids = {e.corpus_id for e in universe}
        assert "org.apache.commons.lang3.math$NumberUtils.createNumber(String)" in ids
        # The faulty method must be in the first-trigger-test universe.
        assert universe, "expected non-empty first-trigger universe for Lang/1"

    def test_default_universe_superset_of_first(self) -> None:
        entities = self._entities()
        first = compute_universe(REAL_PROCESSED, entities, first_only=True)
        all_ = compute_universe(REAL_PROCESSED, entities, first_only=False)
        assert first.issubset(all_)


# ---------------------------------------------------------------------------
# BugsInPy: parse_failing_test_names + compute_universe (reads failing_tests.txt)
# ---------------------------------------------------------------------------


class TestParseFailingTestNames:
    def test_converts_separator(self, tmp_path: Path) -> None:
        p = tmp_path / "failing_tests.txt"
        p.write_text("a.b$C::m\nx.y::f\n")
        assert parse_failing_test_names(p) == ["a.b$C#m", "x.y#f"]

    def test_missing_returns_empty(self, tmp_path: Path) -> None:
        assert parse_failing_test_names(tmp_path / "nope") == []


def test_compute_universe_bugsinpy_reads_failing_tests(tmp_path: Path) -> None:
    """BugsInPy universe: names from failing_tests.txt, coverage from FauxPy/coverage."""
    cov = tmp_path / "FauxPy" / "coverage"
    cov.mkdir(parents=True)
    (tmp_path / "failing_tests.txt").write_text("mod$Cls::test_a\n")
    (cov / "tests.csv").write_text(
        "name,outcome,runtime,stacktrace\nmod$Cls#test_a,FAIL,0,\nmod$Cls#test_b,PASS,0,\n"
    )
    # row 0 (test_a, FAIL) covers col 0; row 1 (test_b, PASS) covers col 1.
    (cov / "matrix.txt").write_text("1 0 -\n0 1 +\n")
    (cov / "spectra.csv").write_text("name\nmod$foo(x):10\nmod$bar(y):20\n")

    entities = [
        MethodEntity("mod$foo(x)", "mod", "mod.py", 10, 15),
        MethodEntity("mod$bar(y)", "mod", "mod.py", 20, 25),
    ]
    universe = compute_universe(tmp_path, entities, first_only=True, dataset="bugsinpy")
    assert {e.corpus_id for e in universe} == {"mod$foo(x)"}
    # No dependence on the deprecated extensionless trigger_tests file.
    assert not (tmp_path / "trigger_tests").exists()
