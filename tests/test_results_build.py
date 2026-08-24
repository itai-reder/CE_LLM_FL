"""Tests for src.results.build — slimming the processed tree into results/.

The fixture bug carries a small corpus where only part of the universe is
reachable from the SR top-20, the fault signatures and the Agent4LR top-5s,
so the closure filter is observable. The builder's own equivalence
self-check (processed-dir metrics == results-dir metrics) guards correctness;
these tests pin the slimming behaviour and the failure path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.results import build as build_mod
from src.results.build import ResultsBuildError, build_bug_results
from src.results.read import load_lr_json, load_meta

SR_MODEL = "llama3.1_8b"

# Reachable: a (top-20 + fault), b (top-20, duplicated row), c (LR top5).
# Unreachable: d, plus the noise entities.
_A = "pkg$Cls.a(String)"
_B = "pkg$Cls.b()"
_C = "pkg$Cls.c()"
_D = "pkg$Other.d()"


def _dotted(corpus_id: str) -> str:
    return corpus_id.replace("$", ".")


def _write_semicolon_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _make_processed_dir(root: Path, *, lr_configs: tuple[str, ...] = ("M1R1",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    entity_rows = [
        [_A, "pkg/Cls.java", "10", "20"],
        [_B, "pkg/Cls.java", "30", "40"],
        [_B, "pkg/Cls.java", "300", "400"],  # overload: only the first row survives
        [_C, "pkg/Cls.java", "50", "60"],
        [_D, "pkg/Other.java", "70", "80"],
    ]
    entity_rows += [
        [f"pkg$Noise.n{i}()", "pkg/Noise.java", str(i * 10), str(i * 10 + 5)] for i in range(1, 21)
    ]
    _write_semicolon_csv(
        root / "method_signatures.csv", ["corpus_id", "path", "startLine", "endLine"], entity_rows
    )

    for name in ("faults.csv", "faults_first.csv"):
        (root / name).write_text(f"path,line,signature\npkg/Cls.java,12,{_A}\n", encoding="utf-8")

    sr_dir = root / "FlexFL" / "SR"
    rankings = sr_dir / "rankings"
    for name, order in (
        ("ochiai", [_A, _B, _D]),
        ("sbir", [_B, _A]),
        ("boostn", [_A, "pkg$Noise.n1()"]),
    ):
        _write_semicolon_csv(
            rankings / f"{name}.csv",
            ["rank", "signature", "path", "startLine", "endLine", "score"],
            [
                [str(i + 1), sig, "pkg/Cls.java", "10", "20", f"{1.0 - i * 0.1:.1f}"]
                for i, sig in enumerate(order)
            ],
        )

    top20 = rankings / "top20" / f"{SR_MODEL}.txt"
    top20.parent.mkdir(parents=True, exist_ok=True)
    top20.write_text(f"{_dotted(_A)}\n{_dotted(_B)}\n", encoding="utf-8")

    agent_sr = sr_dir / "Agent4SR" / SR_MODEL
    agent_sr.mkdir(parents=True, exist_ok=True)
    (agent_sr / "sr_result.json").write_text('{"top20": []}', encoding="utf-8")

    for config in lr_configs:
        config_dir = root / "FlexFL" / "LR" / "Agent4LR" / config
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "lr_result.json").write_text(
            json.dumps(
                {
                    "top5": [_dotted(_C), _dotted(_A)],
                    "top5_indices": [2, 0],
                    "response_dumps": [
                        {
                            "model": "gpt-5-mini-2025-08-07",
                            "usage": {
                                "input_tokens": 1000,
                                "output_tokens": 40,
                                "input_tokens_details": {"cached_tokens": 256},
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build the fixture bug and hand back (status, results_dir, transcripts_dir)."""
    processed = _make_processed_dir(tmp_path / "processed")
    results_dir = tmp_path / "results" / "Proj" / "1"
    transcripts_dir = tmp_path / "transcripts" / "Proj" / "1"
    monkeypatch.setattr(build_mod, "get_processed_dir", lambda p, b, dataset: processed)
    monkeypatch.setattr(build_mod, "get_results_bug_dir", lambda p, b, dataset: results_dir)
    monkeypatch.setattr(build_mod, "get_transcripts_bug_dir", lambda p, b, dataset: transcripts_dir)
    status = build_bug_results("Proj", 1)
    return status, results_dir, transcripts_dir, processed


def _corpus_ids(csv_path: Path) -> list[str]:
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return [row["corpus_id"] for row in csv.DictReader(fh, delimiter=";")]


# ---------------------------------------------------------------------------
# Slimming
# ---------------------------------------------------------------------------


class TestClosureSlimming:
    def test_build_succeeds_and_reports_counts(self, built) -> None:
        status, _results_dir, _t, _p = built
        assert status["status"] == "built"
        assert status["closure"] == 3  # a, b, c
        assert status["entities"] == 3  # deduped rows written
        assert status["lr_configs"] == 1

    def test_universe_keeps_only_the_closure_in_source_order(self, built) -> None:
        _s, results_dir, _t, _p = built
        assert _corpus_ids(results_dir / "method_signatures.csv") == [_A, _B, _C]

    def test_duplicate_corpus_id_keeps_first_row(self, built) -> None:
        _s, results_dir, _t, _p = built
        with (results_dir / "method_signatures.csv").open(encoding="utf-8", newline="") as fh:
            rows = {row["corpus_id"]: row for row in csv.DictReader(fh, delimiter=";")}
        assert rows[_B]["startLine"] == "30"  # not the 300 overload row

    def test_rankings_filtered_to_closure(self, built) -> None:
        _s, results_dir, _t, _p = built
        rankings = results_dir / "rankings"
        with (rankings / "ochiai.csv").open(encoding="utf-8", newline="") as fh:
            sigs = [row["signature"] for row in csv.DictReader(fh, delimiter=";")]
        assert sigs == [_A, _B]  # _D dropped, order preserved
        with (rankings / "boostn.csv").open(encoding="utf-8", newline="") as fh:
            assert [row["signature"] for row in csv.DictReader(fh, delimiter=";")] == [_A]

    def test_top20_and_faults_copied_verbatim(self, built) -> None:
        _s, results_dir, _t, processed = built
        top20 = results_dir / "rankings" / "top20" / f"{SR_MODEL}.txt"
        assert (
            top20.read_bytes()
            == (processed / "FlexFL" / "SR" / "rankings" / "top20" / f"{SR_MODEL}.txt").read_bytes()
        )
        for name in ("faults.csv", "faults_first.csv"):
            assert (results_dir / name).read_bytes() == (processed / name).read_bytes()

    def test_lr_json_carries_top5_and_usage(self, built) -> None:
        _s, results_dir, _t, _p = built
        entries = load_lr_json(results_dir)
        assert list(entries) == ["M1R1"]
        entry = entries["M1R1"]
        assert entry.top5 == [_dotted(_C), _dotted(_A)]
        assert entry.usage.input_tokens == 1000
        assert entry.usage.cached_tokens == 256
        assert entry.usage.output_tokens == 40
        assert entry.usage.cost_usd > 0.0  # priced from the gpt-5-mini response

    def test_meta_records_signals(self, built) -> None:
        _s, results_dir, _t, _p = built
        meta = load_meta(results_dir)
        assert meta is not None
        assert meta["project"] == "Proj"
        assert meta["bug_id"] == 1
        assert meta["source"] == "data/D4J/processed/Proj/1"  # layout path, not the tmp dir
        assert meta["has_sr_result"] is True
        assert meta["trigger_blank"] is False
        assert "built_at" not in meta and "timestamp" not in meta  # deterministic output

    def test_transcripts_copied_out_of_band(self, built) -> None:
        _s, results_dir, transcripts_dir, _p = built
        assert (transcripts_dir / "Agent4LR" / "M1R1" / "lr_result.json").exists()
        assert (transcripts_dir / "Agent4SR" / SR_MODEL / "sr_result.json").exists()
        assert not (results_dir / "FlexFL").exists()  # transcripts stay out of results/


# ---------------------------------------------------------------------------
# Rebuild semantics and determinism
# ---------------------------------------------------------------------------


class TestRebuild:
    def test_second_build_without_force_is_a_noop(self, built) -> None:
        assert build_bug_results("Proj", 1)["status"] == "exists"

    def test_force_rebuild_is_byte_stable(self, built) -> None:
        _s, results_dir, _t, _p = built
        before = {
            p.relative_to(results_dir): p.read_bytes()
            for p in sorted(results_dir.rglob("*"))
            if p.is_file()
        }
        assert build_bug_results("Proj", 1, force=True)["status"] == "built"
        after = {
            p.relative_to(results_dir): p.read_bytes()
            for p in sorted(results_dir.rglob("*"))
            if p.is_file()
        }
        assert before == after

    def test_missing_processed_dir_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(build_mod, "get_processed_dir", lambda p, b, dataset: tmp_path / "nope")
        monkeypatch.setattr(build_mod, "get_results_bug_dir", lambda p, b, dataset: tmp_path / "r")
        status = build_bug_results("Proj", 1)
        assert status["status"] == "skipped"
        assert not (tmp_path / "r").exists()

    def test_missing_corpus_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "processed"
        empty.mkdir()
        monkeypatch.setattr(build_mod, "get_processed_dir", lambda p, b, dataset: empty)
        monkeypatch.setattr(build_mod, "get_results_bug_dir", lambda p, b, dataset: tmp_path / "r")
        status = build_bug_results("Proj", 1)
        assert status["status"] == "skipped"
        assert status["reason"] == "no method_signatures.csv"


# ---------------------------------------------------------------------------
# Equivalence self-check
# ---------------------------------------------------------------------------


class TestEquivalenceSelfCheck:
    def test_over_slimmed_build_is_rejected_and_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processed = _make_processed_dir(tmp_path / "processed")
        results_dir = tmp_path / "results" / "Proj" / "1"
        monkeypatch.setattr(build_mod, "get_processed_dir", lambda p, b, dataset: processed)
        monkeypatch.setattr(build_mod, "get_results_bug_dir", lambda p, b, dataset: results_dir)
        monkeypatch.setattr(
            build_mod, "get_transcripts_bug_dir", lambda p, b, dataset: tmp_path / "transcripts"
        )
        # Drop the fault entity from the closure: metrics must then diverge.
        monkeypatch.setattr(
            build_mod, "_compute_closure", lambda processed_dir, entities, lr: {_B, _C}
        )

        with pytest.raises(ResultsBuildError, match="diverge"):
            build_bug_results("Proj", 1)
        assert not results_dir.exists()
