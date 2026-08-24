"""Tests for BugsInPy evaluation support.

Covers: --benchmark/--dataset normalization in run_evaluation.py, the BIP
no-universe skip (skip-before-write + long-CSV purge), the D4J empty-universe
regression path, sr_model_id threading, the exclusions bucket report, and the
dataset-aware update_tracker backfill.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
import run_evaluation
import update_tracker

from src.evaluation import exclusions
from src.evaluation.cross_bug import SLOTS, append_to_long_csv
from src.evaluation.per_bug import Metrics, evaluate_bug_from_results
from src.evaluation.sources import DEFAULT_SR_MODEL_ID

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_CORPUS_ID = "pkg.mod$Cls.method(self)"
_DOTTED_ID = _CORPUS_ID.replace("$", ".")


def _make_bug_dir(
    root: Path,
    *,
    corpus_ids: list[str] | None = None,
    fault_sigs: list[str] | None = None,
    fault_rows_without_sig: int = 0,
    top20: list[str] | None = None,
    top20_name: str = DEFAULT_SR_MODEL_ID,
    sr_result: bool = False,
    lr_result: bool = False,
    trigger: str | None = None,
) -> Path:
    """Build a minimal results-shaped bug dir under *root*.

    Mirrors what ``run_build_results.py`` emits: ``method_signatures.csv`` /
    ``faults*.csv`` at the root, the SR top-20 under ``rankings/top20/``,
    the Agent4LR configs in ``lr.json``, and the processed-tree signals
    (``has_sr_result``, ``trigger_blank``) in ``meta.json``.
    """
    root.mkdir(parents=True, exist_ok=True)
    corpus_ids = corpus_ids if corpus_ids is not None else [_CORPUS_ID]
    lines = ["corpus_id;path;startLine;endLine"]
    lines += [f"{cid};pkg/mod.py;10;20" for cid in corpus_ids]
    (root / "method_signatures.csv").write_text("\n".join(lines) + "\n")

    fault_lines = ["path,line,signature"]
    for sig in fault_sigs or []:
        fault_lines.append(f"pkg/mod.py,12,{sig}")
    for _ in range(fault_rows_without_sig):
        fault_lines.append("pkg/mod.py,3,")
    (root / "faults.csv").write_text("\n".join(fault_lines) + "\n")
    (root / "faults_first.csv").write_text("\n".join(fault_lines) + "\n")

    if top20 is not None:
        top20_dir = root / "rankings" / "top20"
        top20_dir.mkdir(parents=True)
        (top20_dir / f"{top20_name}.txt").write_text("\n".join(top20) + "\n")
    if lr_result:
        payload = {
            "schema_version": 1,
            "configs": {
                "cfg": {
                    "top5": [_DOTTED_ID],
                    "top5_indices": [1],
                    "responses": [],
                    "usage_totals": {
                        "input_tokens": 0,
                        "cached_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    },
                }
            },
        }
        (root / "lr.json").write_text(json.dumps(payload))
    meta = {
        "schema_version": 1,
        "has_sr_result": bool(sr_result),
        "trigger_blank": trigger is not None and not trigger.strip(),
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def _patch_eval_paths(monkeypatch: pytest.MonkeyPatch, bug_dir: Path, out_dir: Path) -> None:
    """Point run_evaluation + per_bug at a tmp results bug dir and long-CSV dir."""
    monkeypatch.setattr(run_evaluation, "get_results_bug_dir", lambda p, b, dataset: bug_dir)
    monkeypatch.setattr("src.evaluation.per_bug.get_results_bug_dir", lambda p, b, dataset: bug_dir)
    paths = {slot: (out_dir / f"{slot}.csv", out_dir / f"{slot}_summary.csv") for slot in SLOTS}
    monkeypatch.setattr(run_evaluation, "slot_paths", lambda dataset: paths)


def _metrics(method: str = "Ochiai") -> Metrics:
    return Metrics(method, 1.0, 1.0, 1, 1, 1, 1, 1, 0.0)


def _long_rows(path: Path) -> list[list[str]]:
    with path.open() as fh:
        return list(csv.reader(fh))[1:]


# ---------------------------------------------------------------------------
# --benchmark / --dataset normalization
# ---------------------------------------------------------------------------


class TestNormalizeBenchmarkArgs:
    def test_benchmark_authoritative(self) -> None:
        args = argparse.Namespace(benchmark="bugsinpy", dataset="defects4j")
        run_evaluation._normalize_benchmark_args(args)
        assert args.dataset == "bugsinpy"

    def test_dataset_alias_honored_when_benchmark_default(self) -> None:
        args = argparse.Namespace(benchmark="defects4j", dataset="bugsinpy")
        run_evaluation._normalize_benchmark_args(args)
        assert args.dataset == "bugsinpy"

    def test_benchmark_wins_over_dataset(self) -> None:
        args = argparse.Namespace(benchmark="bugsinpy", dataset="somethingelse")
        run_evaluation._normalize_benchmark_args(args)
        assert args.dataset == "bugsinpy"

    def test_unknown_key_raises(self) -> None:
        args = argparse.Namespace(benchmark="defects4j", dataset="nope")
        with pytest.raises(SystemExit):
            run_evaluation._normalize_benchmark_args(args)

    def test_default_stays_defects4j(self) -> None:
        args = argparse.Namespace(benchmark="defects4j", dataset="defects4j")
        run_evaluation._normalize_benchmark_args(args)
        assert args.dataset == "defects4j"


# ---------------------------------------------------------------------------
# process_bug — BIP skip semantics vs D4J regression
# ---------------------------------------------------------------------------


class TestProcessBugSkip:
    def test_bip_no_universe_skips_before_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bug_dir = _make_bug_dir(tmp_path / "bug", fault_sigs=[_CORPUS_ID])  # no top-20
        out_dir = tmp_path / "_evaluation"
        _patch_eval_paths(monkeypatch, bug_dir, out_dir)

        result = run_evaluation.process_bug("Proj", 1, dataset="bugsinpy")

        assert result["status"] == "skipped"
        assert result["skip_code"] == "no_universe"
        assert "top20" in result["reason"]
        assert not (bug_dir / "evaluation").exists()

    def test_bip_skip_purges_only_this_bugs_long_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bug_dir = _make_bug_dir(tmp_path / "bug", fault_sigs=[_CORPUS_ID])
        out_dir = tmp_path / "_evaluation"
        _patch_eval_paths(monkeypatch, bug_dir, out_dir)
        out_dir.mkdir()
        for slot in SLOTS:
            append_to_long_csv(out_dir / f"{slot}.csv", "Proj", 1, [_metrics()])
            append_to_long_csv(out_dir / f"{slot}.csv", "Other", 2, [_metrics()])

        result = run_evaluation.process_bug("Proj", 1, dataset="bugsinpy")

        assert result["status"] == "skipped"
        for slot in SLOTS:
            rows = _long_rows(out_dir / f"{slot}.csv")
            assert [(r[0], r[1]) for r in rows] == [("Other", "2")]

    def test_d4j_empty_universe_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bug_dir = _make_bug_dir(tmp_path / "bug", fault_sigs=[_CORPUS_ID])  # no top-20
        out_dir = tmp_path / "_evaluation"
        _patch_eval_paths(monkeypatch, bug_dir, out_dir)
        out_dir.mkdir()
        for slot in SLOTS:
            append_to_long_csv(out_dir / f"{slot}.csv", "Proj", 1, [_metrics()])

        result = run_evaluation.process_bug("Proj", 1, dataset="defects4j")

        # Historical behavior: status ok, header-only per-bug CSVs written,
        # prior long rows replaced by nothing (empty append).
        assert result["status"] == "ok"
        assert result["counts"] == {slot: 0 for slot in SLOTS}
        per_bug = bug_dir / "evaluation" / "baselines.csv"
        assert per_bug.exists()
        assert len(per_bug.read_text().splitlines()) == 1
        for slot in SLOTS:
            assert _long_rows(out_dir / f"{slot}.csv") == []

    def test_bip_with_universe_evaluates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bug_dir = _make_bug_dir(tmp_path / "bug", fault_sigs=[_CORPUS_ID], top20=[_DOTTED_ID])
        out_dir = tmp_path / "_evaluation"
        _patch_eval_paths(monkeypatch, bug_dir, out_dir)

        result = run_evaluation.process_bug("Proj", 1, dataset="bugsinpy")

        assert result["status"] == "ok"
        assert result["counts"]["baselines"] == 3
        assert (bug_dir / "evaluation" / "baselines.csv").exists()


# ---------------------------------------------------------------------------
# sr_model_id threading
# ---------------------------------------------------------------------------


class TestSrModelIdThreading:
    def test_evaluate_bug_resolves_custom_top20(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bug_dir = _make_bug_dir(
            tmp_path / "bug",
            fault_sigs=[_CORPUS_ID],
            top20=[_DOTTED_ID],
            top20_name="custom_model",
        )
        monkeypatch.setattr(
            "src.evaluation.per_bug.get_results_bug_dir", lambda p, b, dataset: bug_dir
        )

        default = evaluate_bug_from_results("Proj", 1, dataset="bugsinpy")
        custom = evaluate_bug_from_results(
            "Proj", 1, dataset="bugsinpy", sr_model_id="custom_model"
        )

        assert not any(default.values())
        assert len(custom["baselines"]) == 3


# ---------------------------------------------------------------------------
# Exclusions report
# ---------------------------------------------------------------------------


def _audit_row(status: str, **extra: str) -> dict[str, str]:
    row = {"status": status, "error_msg": "", "unreached_sigs": "1"}
    row.update(extra)
    return row


def _classified(processed_dir: Path, audit_row: dict[str, str] | None) -> tuple[str, str, str]:
    verdict = exclusions.classify_exclusion(processed_dir, audit_row)
    assert verdict is not None
    return verdict


class TestClassifyExclusion:
    def test_extraction_incomplete_from_audit(self, tmp_path: Path) -> None:
        verdict = _classified(tmp_path, _audit_row("real_failure", error_msg="boom"))
        assert verdict == ("extraction_incomplete", "all", "boom")

    def test_extraction_incomplete_when_absent_from_audit(self, tmp_path: Path) -> None:
        bucket, scope, _ = _classified(tmp_path, None)
        assert (bucket, scope) == ("extraction_incomplete", "all")

    def test_fl_unreachable(self, tmp_path: Path) -> None:
        bucket, scope, detail = _classified(
            tmp_path, _audit_row("fl_unreachable", unreached_sigs="3")
        )
        assert bucket == "fl_unreachable"
        assert scope == "all"
        assert "n=3" in detail

    def test_method_unlocalizable(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(tmp_path / "b", fault_sigs=[], fault_rows_without_sig=2)
        bucket, scope, _ = _classified(bug, _audit_row("ok"))
        assert (bucket, scope) == ("method_unlocalizable", "aggregates")

    def test_sr_not_run_blank_trigger(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(tmp_path / "b", fault_sigs=[_CORPUS_ID], trigger="")
        bucket, scope, detail = _classified(bug, _audit_row("ok"))
        assert (bucket, scope) == ("sr_not_run", "all")
        assert "blank trigger" in detail

    def test_sr_not_run_plain(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(tmp_path / "b", fault_sigs=[_CORPUS_ID], trigger="test body")
        bucket, _, detail = _classified(bug, _audit_row("ok"))
        assert bucket == "sr_not_run"
        assert "blank trigger" not in detail

    def test_lr_readiness_skipped_no_top20(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(tmp_path / "b", fault_sigs=[_CORPUS_ID], sr_result=True)
        bucket, scope, _ = _classified(bug, _audit_row("ok"))
        assert (bucket, scope) == ("lr_readiness_skipped", "all")

    def test_lr_measurement_skipped_fault_not_in_top20(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(
            tmp_path / "b",
            fault_sigs=[_CORPUS_ID],
            sr_result=True,
            top20=["pkg.other.Cls.m(self)"],
        )
        bucket, scope, _ = _classified(bug, _audit_row("ok"))
        assert (bucket, scope) == ("lr_measurement_skipped", "flexfl")

    def test_lr_not_run_fault_in_top20(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(
            tmp_path / "b", fault_sigs=[_CORPUS_ID], sr_result=True, top20=[_DOTTED_ID]
        )
        bucket, scope, _ = _classified(bug, _audit_row("ok"))
        assert (bucket, scope) == ("lr_not_run", "flexfl")

    def test_fully_evaluable_returns_none(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(
            tmp_path / "b",
            fault_sigs=[_CORPUS_ID],
            sr_result=True,
            top20=[_DOTTED_ID],
            lr_result=True,
        )
        assert exclusions.classify_exclusion(bug, _audit_row("ok")) is None

    def test_precedence_audit_beats_sr_state(self, tmp_path: Path) -> None:
        # real_failure bug that also has no sr_result → extraction_incomplete wins
        bucket, _, _ = _classified(tmp_path / "missing", _audit_row("real_failure"))
        assert bucket == "extraction_incomplete"

    def test_precedence_unlocalizable_beats_missing_top20(self, tmp_path: Path) -> None:
        bug = _make_bug_dir(tmp_path / "b", fault_sigs=[], fault_rows_without_sig=1, sr_result=True)
        bucket, _, _ = _classified(bug, _audit_row("ok"))
        assert bucket == "method_unlocalizable"


class _FakeAdapter:
    def __init__(self, cases: dict[str, list[int]]) -> None:
        self._cases = cases

    def list_projects(self) -> list[str]:
        return sorted(self._cases)

    def list_cases(self, project: str) -> list[int]:
        return self._cases[project]


class TestWriteExclusionsReport:
    def test_report_written_sorted_and_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bugs_root = tmp_path / "results"
        # ProjB/1: evaluable; ProjA/1: no top-20 (excluded)
        evaluable = _make_bug_dir(
            bugs_root / "ProjB" / "1",
            fault_sigs=[_CORPUS_ID],
            sr_result=True,
            top20=[_DOTTED_ID],
            lr_result=True,
        )
        excluded = _make_bug_dir(bugs_root / "ProjA" / "1", fault_sigs=[_CORPUS_ID], sr_result=True)
        assert evaluable and excluded  # built

        audit_csv = tmp_path / "audit.csv"
        audit_csv.write_text(
            "project,bug,status,error_msg,unreached_sigs\nProjA,1,ok,,0\nProjB,1,ok,,0\n"
        )
        monkeypatch.setattr(
            exclusions,
            "get_benchmark_adapter",
            lambda dataset: _FakeAdapter({"ProjA": [1], "ProjB": [1]}),
        )
        monkeypatch.setattr(exclusions, "_audit_csv_path", lambda dataset: audit_csv)
        monkeypatch.setattr(
            exclusions,
            "get_results_bug_dir",
            lambda p, b, dataset: bugs_root / p / str(b),
        )

        out = tmp_path / "exclusions.csv"
        exclusions.write_exclusions_report(out_path=out)
        exclusions.write_exclusions_report(out_path=out)  # idempotent rewrite

        with out.open() as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == list(exclusions.EXCLUSION_HEADERS)
        assert len(rows) == 2  # only the excluded bug
        assert rows[1][:4] == ["ProjA", "1", "lr_readiness_skipped", "all"]


# ---------------------------------------------------------------------------
# update_tracker — dataset-aware backfill
# ---------------------------------------------------------------------------


def _make_bip_extraction_dir(root: Path) -> Path:
    """Minimal BIP tree exercising the dispatched _infer_extraction checks."""
    root.mkdir(parents=True)
    (root / "failing_tests.txt").write_text("pkg.mod$Cls::test_x\n")
    cov = root / "FauxPy" / "coverage"
    cov.mkdir(parents=True)
    (cov / "spectra.csv").write_text("name\npkg.mod$Cls.m(self):12\n")
    (cov / "matrix.txt").write_text("1 +\n")
    (cov / "tests.csv").write_text("name,outcome\nt,PASS\n")
    reports = root / "FauxPy" / "reports"
    reports.mkdir(parents=True)
    (reports / "ochiai.ranking.csv").write_text("name;suspiciousness_value\nx;1.0\n")
    (root / "trigger_test_clean.txt").write_text("")  # blank = non-mirrorable, still ran
    return root


class TestInferExtractionDispatch:
    def test_bip_paths_and_conditionals(self, tmp_path: Path) -> None:
        bug_dir = _make_bip_extraction_dir(tmp_path / "bip")
        tracker = update_tracker._empty_tracker()
        update_tracker._infer_extraction(bug_dir, tracker, dataset="bugsinpy")
        completed = tracker["extraction"]["completed"]
        assert "gzoltar" in completed  # FauxPy coverage/reports resolved
        assert "failing_tests" in completed  # no raw-file requirement
        assert "trigger_test_processed" in completed  # blank counts for BIP
        assert "properties" not in completed  # D4J-only check skipped

    def test_d4j_blank_trigger_not_completed(self, tmp_path: Path) -> None:
        bug_dir = tmp_path / "d4j"
        bug_dir.mkdir()
        (bug_dir / "trigger_test_clean.txt").write_text("")
        tracker = update_tracker._empty_tracker()
        update_tracker._infer_extraction(bug_dir, tracker, dataset="defects4j")
        assert "trigger_test_processed" not in tracker["extraction"]["completed"]

    def test_d4j_sfl_paths_still_resolve(self, tmp_path: Path) -> None:
        bug_dir = tmp_path / "d4j"
        sfl = bug_dir / "sfl" / "sfl" / "txt"
        sfl.mkdir(parents=True)
        for name in ("spectra.csv", "matrix.txt", "tests.csv", "ochiai.ranking.csv"):
            (sfl / name).write_text("content\n")
        tracker = update_tracker._empty_tracker()
        update_tracker._infer_extraction(bug_dir, tracker, dataset="defects4j")
        assert "gzoltar" in tracker["extraction"]["completed"]


class TestUpdateSingleBugDataset:
    def test_save_tracker_receives_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bug_dir = _make_bip_extraction_dir(tmp_path / "bip")
        captured: dict[str, str] = {}

        def _fake_save(tracker: dict, project: str, bug_id: int, *, dataset: str) -> None:
            captured["dataset"] = dataset

        monkeypatch.setattr(update_tracker, "save_tracker", _fake_save)
        update_tracker.update_single_bug(bug_dir, "Proj", 1, dataset="bugsinpy", no_backup=True)
        assert captured["dataset"] == "bugsinpy"
