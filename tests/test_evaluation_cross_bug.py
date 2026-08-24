"""Tests for src.evaluation.cross_bug."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.evaluation.cross_bug import (
    LONG_HEADERS,
    SUMMARY_HEADERS,
    append_to_long_csv,
    write_summary_csv,
)
from src.evaluation.per_bug import Metrics


def _read(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


# ---------------------------------------------------------------------------
# append_to_long_csv — idempotency
# ---------------------------------------------------------------------------


class TestAppendToLongCsv:
    def test_appends_for_new_bug(self, tmp_path: Path) -> None:
        out = tmp_path / "baselines.csv"
        append_to_long_csv(
            out,
            "Lang",
            1,
            [Metrics("Ochiai", 1.0, 1.0, 1, 1, 1, 1, 1, 0.0)],
        )
        append_to_long_csv(
            out,
            "Lang",
            2,
            [Metrics("Ochiai", 5.0, 7.5, 0, 0, 1, 1, 1, 0.1)],
        )
        rows = _read(out)
        assert rows[0] == list(LONG_HEADERS)
        assert len(rows) == 3
        assert rows[1][:3] == ["Lang", "1", "Ochiai"]
        assert rows[2][:3] == ["Lang", "2", "Ochiai"]

    def test_replaces_prior_rows_for_same_bug(self, tmp_path: Path) -> None:
        out = tmp_path / "baselines.csv"
        append_to_long_csv(
            out,
            "Lang",
            1,
            [
                Metrics("Ochiai", 1.0, 1.0, 1, 1, 1, 1, 1, 0.0),
                Metrics("SBIR", 1.0, 1.0, 1, 1, 1, 1, 1, 0.0),
            ],
        )
        # Re-run with different metrics for Lang/1
        append_to_long_csv(
            out,
            "Lang",
            1,
            [Metrics("Ochiai", 5.0, 5.0, 0, 0, 0, 0, 1, 0.2)],
        )
        rows = _read(out)
        # Only the new Ochiai row remains for Lang/1 (SBIR was dropped).
        assert len(rows) == 2
        # Token-tail defaults: 0,0,0,blank for baseline rows.
        assert rows[1] == [
            "Lang",
            "1",
            "Ochiai",
            "5.0",
            "5.0",
            "0",
            "0",
            "0",
            "0",
            "1",
            "0.2",
            "0",
            "0",
            "0",
            "",
        ]

    def test_blank_cells_for_none_metrics(self, tmp_path: Path) -> None:
        out = tmp_path / "x.csv"
        append_to_long_csv(
            out,
            "Foo",
            1,
            [Metrics("M", None, None, 0, 0, 0, 0, 0, None)],
        )
        rows = _read(out)
        assert rows[1] == [
            "Foo",
            "1",
            "M",
            "",
            "",
            "0",
            "0",
            "0",
            "0",
            "0",
            "",
            "0",
            "0",
            "0",
            "",
        ]

    def test_token_columns_emitted_for_agent_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "flexfl.csv"
        append_to_long_csv(
            out,
            "Lang",
            1,
            [
                Metrics(
                    "Agent4LR-X",
                    1.0,
                    1.0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.0,
                    InputTokens=1234,
                    CachedTokens=56,
                    OutputTokens=78,
                    CostUSD=0.0009,
                )
            ],
        )
        rows = _read(out)
        assert rows[1][-4:] == ["1234", "56", "78", "0.000900"]


# ---------------------------------------------------------------------------
# write_summary_csv — aggregation rules
# ---------------------------------------------------------------------------


class TestWriteSummaryCsv:
    def _build_long_csv(self, tmp_path: Path, rows: list[list[str]]) -> Path:
        path = tmp_path / "long.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(LONG_HEADERS)
            for row in rows:
                writer.writerow(row)
        return path

    def test_means_across_bugs_for_one_method(self, tmp_path: Path) -> None:
        long = self._build_long_csv(
            tmp_path,
            [
                # baseline rows have zero token columns and a blank cost cell
                [
                    "Lang",
                    "1",
                    "Ochiai",
                    "1.0",
                    "1.0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "0.0",
                    "0",
                    "0",
                    "0",
                    "",
                ],
                [
                    "Lang",
                    "2",
                    "Ochiai",
                    "5.0",
                    "7.0",
                    "0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "0.2",
                    "0",
                    "0",
                    "0",
                    "",
                ],
            ],
        )
        summary = tmp_path / "summary.csv"
        write_summary_csv(long, summary)
        rows = _read(summary)
        assert rows[0] == list(SUMMARY_HEADERS)
        ochiai = next(r for r in rows[1:] if r[0] == "Ochiai")
        assert ochiai[1] == "2"  # bugs
        assert float(ochiai[2]) == 3.0  # MFR = (1+5)/2
        assert float(ochiai[3]) == 4.0  # MAR = (1+7)/2
        assert float(ochiai[4]) == 0.5  # Top1Rate = (1+0)/2
        assert float(ochiai[5]) == 1.0  # Top2Rate = (1+1)/2
        assert float(ochiai[9]) == 0.1  # MeanWE = (0+0.2)/2
        # Token / cost aggregates: all zero for baseline rows.
        assert ochiai[10:14] == ["0", "0", "0", "0.000000"]
        assert ochiai[14] == ""  # MeanCostUSD blank when no cost cells

    def test_drops_rows_with_blank_fr(self, tmp_path: Path) -> None:
        long = self._build_long_csv(
            tmp_path,
            [
                # Lang/2 had no faulty in universe → blank FR; should be dropped.
                [
                    "Lang",
                    "1",
                    "Ochiai",
                    "2.0",
                    "2.0",
                    "0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "0.1",
                    "0",
                    "0",
                    "0",
                    "",
                ],
                ["Lang", "2", "Ochiai", "", "", "0", "0", "0", "0", "0", "", "0", "0", "0", ""],
            ],
        )
        summary = tmp_path / "summary.csv"
        write_summary_csv(long, summary)
        ochiai = next(r for r in _read(summary)[1:] if r[0] == "Ochiai")
        assert ochiai[1] == "1"  # only Lang/1 counts
        assert float(ochiai[2]) == 2.0

    def test_empty_method_group_writes_blank_row(self, tmp_path: Path) -> None:
        long = self._build_long_csv(
            tmp_path,
            [["Lang", "1", "Foo", "", "", "0", "0", "0", "0", "0", "", "0", "0", "0", ""]],
        )
        summary = tmp_path / "summary.csv"
        write_summary_csv(long, summary)
        foo = next(r for r in _read(summary)[1:] if r[0] == "Foo")
        assert foo[1] == "0"
        # MFR, MAR, TopKRate*5, MeanWE all blank; then 0/0/0/0.000000/blank tail.
        assert foo[2:] == ["", "", "", "", "", "", "", "", "0", "0", "0", "0.000000", ""]

    def test_token_aggregation_for_agent_rows(self, tmp_path: Path) -> None:
        long = self._build_long_csv(
            tmp_path,
            [
                [
                    "Lang",
                    "1",
                    "Agent4LR-X",
                    "1.0",
                    "1.0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "0.0",
                    "1000",
                    "200",
                    "50",
                    "0.001000",
                ],
                [
                    "Lang",
                    "2",
                    "Agent4LR-X",
                    "2.0",
                    "2.0",
                    "0",
                    "1",
                    "1",
                    "1",
                    "1",
                    "0.1",
                    "500",
                    "0",
                    "25",
                    "0.000125",
                ],
            ],
        )
        summary = tmp_path / "summary.csv"
        write_summary_csv(long, summary)
        row = next(r for r in _read(summary)[1:] if r[0] == "Agent4LR-X")
        assert row[10] == "1500"  # TotalInputTokens
        assert row[11] == "200"  # TotalCachedTokens
        assert row[12] == "75"  # TotalOutputTokens
        assert float(row[13]) == pytest.approx(0.001125)  # TotalCostUSD
        assert float(row[14]) == pytest.approx(0.0005625, abs=1e-6)  # MeanCostUSD
