"""Tests for src.common.flexfl_compat module."""

from __future__ import annotations

import csv
from pathlib import Path

from src.common.flexfl_compat import (
    build_line_to_method_rows,
    convert_boostn_scores,
    convert_boostn_scores_corpus_id,
    convert_statement_scores_with_map,
    parse_method_id_to_flex_row,
    write_flex_method_csv,
)


def test_parse_method_id_to_flex_row() -> None:
    method_id = "org.example.Foo.doWork(int,String).42.51"
    row = parse_method_id_to_flex_row(method_id)

    assert row is not None
    assert row.file == "org.example.Foo"
    assert row.signature == "doWork(int,String)"
    assert row.start_line == 42
    assert row.end_line == 51


def test_convert_boostn_scores_orders_descending() -> None:
    scores = {
        "org.example.Foo.a().10.12": 0.2,
        "org.example.Foo.b().20.25": 0.8,
    }

    ranked = convert_boostn_scores(scores)

    assert len(ranked) == 2
    assert ranked[0][0].signature == "b()"
    assert ranked[0][1] == 0.8
    assert ranked[1][0].signature == "a()"
    assert ranked[1][1] == 0.2


def test_statement_scores_aggregate_to_methods(fixtures_dir: Path) -> None:
    line_map = build_line_to_method_rows(fixtures_dir)
    statement_scores = {
        ("org.example.SampleClass", 9): 0.3,
        ("org.example.SampleClass", 10): 0.9,
        ("org.example.SampleClass", 17): 0.7,
    }

    ranked = convert_statement_scores_with_map(statement_scores, line_map)
    score_by_key = {row.method_key: score for row, score in ranked}

    assert score_by_key["org.example.SampleClass.add(int,int)"] == 0.9
    assert score_by_key["org.example.SampleClass.format(String,Object)"] == 0.7


def test_convert_boostn_scores_corpus_id_joins_against_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "method_signatures.csv"
    csv_path.write_text(
        "corpus_id;path;startLine;endLine\n"
        "org.example$Foo.a();src/Foo.java;10;12\n"
        "org.example$Foo.b(int);src/Foo.java;20;25\n",
        encoding="utf-8",
    )
    scores = {
        "org.example$Foo.a()": 0.2,
        "org.example$Foo.b(int)": 0.8,
        "org.example$Foo.missing()": 0.5,  # not in csv -> dropped with warning
    }

    ranked = convert_boostn_scores_corpus_id(scores, tmp_path)

    assert len(ranked) == 2
    assert ranked[0][0].file == "org.example.Foo"
    assert ranked[0][0].signature == "b(int)"
    assert ranked[0][0].start_line == 20
    assert ranked[0][0].end_line == 25
    assert ranked[0][1] == 0.8
    assert ranked[1][0].signature == "a()"
    assert ranked[1][1] == 0.2


def test_write_flex_method_csv_headers(tmp_path: Path) -> None:
    ranked = convert_boostn_scores({"org.example.Foo.a().10.12": 1.0})

    without_score = tmp_path / "without_score.csv"
    with_score = tmp_path / "with_score.csv"
    write_flex_method_csv(without_score, ranked, include_suspiciousness=False)
    write_flex_method_csv(with_score, ranked, include_suspiciousness=True)

    with without_score.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["File", "Signature", "StartLine", "EndLine"]

    with with_score.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["File", "Signature", "StartLine", "EndLine", "Suspiciousness"]
