"""Tests for src.results.read — the slim results-tree readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.results.read import (
    LR_JSON_NAME,
    META_JSON_NAME,
    LrEntry,
    check_required_configs_results,
    load_lr_json,
    load_meta,
)


def _write_lr(results_dir: Path, configs: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / LR_JSON_NAME).write_text(
        json.dumps({"schema_version": 1, "configs": configs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config(top5: list[object], **usage: float) -> dict:
    totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    totals.update(usage)
    return {"top5": top5, "top5_indices": [], "responses": [], "usage_totals": totals}


# ---------------------------------------------------------------------------
# load_lr_json
# ---------------------------------------------------------------------------


class TestLoadLrJson:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert load_lr_json(tmp_path) == {}

    def test_malformed_json_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / LR_JSON_NAME).write_text("{not json", encoding="utf-8")
        assert load_lr_json(tmp_path) == {}

    def test_missing_configs_mapping_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / LR_JSON_NAME).write_text('{"schema_version": 1}', encoding="utf-8")
        assert load_lr_json(tmp_path) == {}

    def test_top5_and_usage_round_trip(self, tmp_path: Path) -> None:
        _write_lr(
            tmp_path,
            {
                "M1R1-M1R1-M1R1": _config(
                    ["pkg.Cls.a()", "pkg.Cls.b()"],
                    input_tokens=1200,
                    cached_tokens=300,
                    output_tokens=45,
                    cost_usd=0.00123,
                )
            },
        )
        entries = load_lr_json(tmp_path)
        entry = entries["M1R1-M1R1-M1R1"]
        assert isinstance(entry, LrEntry)
        assert entry.top5 == ["pkg.Cls.a()", "pkg.Cls.b()"]
        assert entry.usage.input_tokens == 1200
        assert entry.usage.cached_tokens == 300
        assert entry.usage.output_tokens == 45
        assert entry.usage.cost_usd == pytest.approx(0.00123)

    def test_configs_returned_in_sorted_order(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {name: _config([f"pkg.Cls.{name}()"]) for name in ("zz", "aa", "mm")})
        assert list(load_lr_json(tmp_path)) == ["aa", "mm", "zz"]

    def test_absent_usage_totals_defaults_to_zero(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {"cfg": {"top5": ["pkg.Cls.a()"]}})
        usage = load_lr_json(tmp_path)["cfg"].usage
        assert (usage.input_tokens, usage.cached_tokens, usage.output_tokens) == (0, 0, 0)
        assert usage.cost_usd == pytest.approx(0.0)

    def test_non_string_top5_entries_dropped(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {"cfg": _config(["pkg.Cls.a()", None, 7, "pkg.Cls.b()"])})
        assert load_lr_json(tmp_path)["cfg"].top5 == ["pkg.Cls.a()", "pkg.Cls.b()"]

    def test_non_dict_entry_skipped(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {"good": _config(["pkg.Cls.a()"]), "bad": "oops"})
        assert list(load_lr_json(tmp_path)) == ["good"]


# ---------------------------------------------------------------------------
# check_required_configs_results
# ---------------------------------------------------------------------------


class TestCheckRequiredConfigsResults:
    def test_all_present(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {"a": _config(["pkg.Cls.a()"]), "b": _config(["pkg.Cls.b()"])})
        assert check_required_configs_results(tmp_path, ["a", "b"]) == (True, [])

    def test_missing_config_reported(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {"a": _config(["pkg.Cls.a()"])})
        ok, problems = check_required_configs_results(tmp_path, ["a", "b"])
        assert not ok
        assert problems == ["b (missing)"]

    def test_blank_top5_is_invalid(self, tmp_path: Path) -> None:
        _write_lr(tmp_path, {"a": _config(["", "  "])})
        ok, problems = check_required_configs_results(tmp_path, ["a"])
        assert not ok
        assert problems == ["a (invalid top5)"]

    def test_no_lr_json_reports_every_config_missing(self, tmp_path: Path) -> None:
        ok, problems = check_required_configs_results(tmp_path, ["a", "b"])
        assert not ok
        assert problems == ["a (missing)", "b (missing)"]


# ---------------------------------------------------------------------------
# load_meta
# ---------------------------------------------------------------------------


class TestLoadMeta:
    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_meta(tmp_path) is None

    def test_malformed_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / META_JSON_NAME).write_text("{", encoding="utf-8")
        assert load_meta(tmp_path) is None

    def test_non_object_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / META_JSON_NAME).write_text("[1, 2]", encoding="utf-8")
        assert load_meta(tmp_path) is None

    def test_signals_round_trip(self, tmp_path: Path) -> None:
        (tmp_path / META_JSON_NAME).write_text(
            json.dumps({"has_sr_result": True, "trigger_blank": False, "bug_id": 3}),
            encoding="utf-8",
        )
        meta = load_meta(tmp_path)
        assert meta is not None
        assert meta["has_sr_result"] is True
        assert meta["trigger_blank"] is False
        assert meta["bug_id"] == 3
