"""Tests for src.sbir.sbfl module."""

from __future__ import annotations

from pathlib import Path

from src.sbir.sbfl import SBFL


class TestToStatementId:
    """Tests for GZoltar name -> statement ID conversion."""

    def test_simple_conversion(self) -> None:
        raw = "org.example$SampleClass#method(int):42"
        assert SBFL._to_statement_id(raw) == "org.example.SampleClass#42"

    def test_nested_class(self) -> None:
        raw = "org.example$Outer$Inner#foo():10"
        assert SBFL._to_statement_id(raw) == "org.example.Outer.Inner#10"

    def test_no_dollar_sign(self) -> None:
        raw = "SomeClass#bar():5"
        assert SBFL._to_statement_id(raw) == "SomeClass#5"


class TestParseOchiaiCsv:
    """Tests for parsing GZoltar semicolon-delimited Ochiai ranking files."""

    def test_parses_valid_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ochiai.ranking.csv"
        csv_file.write_text(
            "name;suspiciousness_value\n"
            "org.example$Foo#bar():10;0.75\n"
            "org.example$Foo#bar():20;0.50\n"
            "org.example$Baz#qux():5;0.25\n"
        )
        scores = SBFL.parse_ochiai_csv(csv_file)
        assert len(scores) == 3
        assert scores["org.example.Foo#10"] == 0.75
        assert scores["org.example.Foo#20"] == 0.50
        assert scores["org.example.Baz#5"] == 0.25

    def test_skips_malformed_rows(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ochiai.ranking.csv"
        csv_file.write_text(
            "name;suspiciousness_value\n"
            "org.example$Foo#bar():10;0.75\n"
            "malformed_no_hash_no_colon;0.50\n"
            "org.example$Baz#qux():5;not_a_number\n"
        )
        scores = SBFL.parse_ochiai_csv(csv_file)
        assert len(scores) == 1
        assert scores["org.example.Foo#10"] == 0.75

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ochiai.ranking.csv"
        csv_file.write_text("name;suspiciousness_value\n")
        scores = SBFL.parse_ochiai_csv(csv_file)
        assert scores == {}

    def test_header_only_no_newline(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ochiai.ranking.csv"
        csv_file.write_text("")
        scores = SBFL.parse_ochiai_csv(csv_file)
        assert scores == {}

    def test_missing_file_raises(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            SBFL.parse_ochiai_csv(Path("/tmp/nonexistent_ochiai.csv"))


class TestWriteOutputs:
    """Tests for JSON and CSV output writing."""

    def test_write_json(self, tmp_path: Path) -> None:
        import json

        scores = {"org.example.Foo#10": 0.75, "org.example.Bar#20": 0.50}
        output = tmp_path / "sbfl.json"
        SBFL._write_json(scores, output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "sbfl_scores" in data
        assert data["sbfl_scores"]["org.example.Foo#10"] == 0.75

    def test_write_stmt_susps(self, tmp_path: Path) -> None:
        scores = {"org.example.Foo#10": 0.75, "org.example.Bar#20": 0.50}
        output = tmp_path / "stmt-susps.txt"
        SBFL._write_stmt_susps(scores, output)
        assert output.exists()
        lines = output.read_text().strip().splitlines()
        assert lines[0] == "Statement,Suspiciousness"
        # First data line should be highest score
        assert "org.example.Foo#10" in lines[1]
        assert "0.75" in lines[1]


class TestLoadRanking:
    """Tests for loading our own comma-delimited ranking CSV."""

    def test_load_valid_ranking(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "ranking.csv"
        csv_file.write_text(
            "Statement,Suspiciousness\norg.example.Foo#10,0.75\norg.example.Bar#20,0.50\n"
        )
        scores = SBFL.load_ranking(csv_file)
        assert len(scores) == 2
        assert scores["org.example.Foo#10"] == 0.75
        assert scores["org.example.Bar#20"] == 0.50

    def test_missing_file_raises(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            SBFL.load_ranking("/tmp/nonexistent_ranking.csv")
