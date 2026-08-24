"""Tests for src.evaluation.sources."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.common.method_entity import MethodEntity
from src.evaluation.sources import (
    Usage,
    check_required_configs,
    discover_agent4lr_configs,
    load_agent4lr_scores,
    load_agent4lr_usage,
    load_baseline_scores,
    load_candidate_universe,
)


def _entities() -> list[MethodEntity]:
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


# ---------------------------------------------------------------------------
# load_baseline_scores
# ---------------------------------------------------------------------------


class TestLoadBaselineScores:
    def _setup_csv(self, tmp_path: Path, method: str, body: str) -> Path:
        # load_baseline_scores takes the rankings dir itself (FlexFL/SR/rankings/
        # in a processed bug dir, rankings/ in a results bug dir).
        (tmp_path / f"{method}.csv").write_text(body, encoding="utf-8")
        return tmp_path

    def test_parses_semicolon_delimited_ranking_csv(self, tmp_path: Path) -> None:
        body = textwrap.dedent(
            """\
            rank;signature;path;startLine;endLine;score
            1;a.b$Cls.foo(int);a/b/Cls.java;10;20;0.9
            2;a.b$Cls.bar(String);a/b/Cls.java;25;40;0.5
            """
        )
        self._setup_csv(tmp_path, "ochiai", body)
        scores = load_baseline_scores(tmp_path, "ochiai", _entities())
        ids = {e.corpus_id: v for e, v in scores.items()}
        assert ids == {"a.b$Cls.foo(int)": 0.9, "a.b$Cls.bar(String)": 0.5}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_baseline_scores(tmp_path, "boostn", _entities()) == {}

    def test_unknown_method_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_baseline_scores(tmp_path, "nope", _entities())


# ---------------------------------------------------------------------------
# load_agent4lr_scores
# ---------------------------------------------------------------------------


class TestLoadAgent4LRScores:
    def _write_lr_result(self, path: Path, top5: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "top5": top5,
                    "top5_indices": [1, 2, 3, 4, 5],
                    "config_name": "M1R1-M1R1-M1R1",
                    "schema_version": 1,
                }
            )
        )

    def test_reinserts_dollar_via_dotted_index(self, tmp_path: Path) -> None:
        lr = tmp_path / "lr_result.json"
        # Top1 is the dotted form (no $); should still match.
        self._write_lr_result(lr, ["a.b.Cls.foo(int)", "a.b.Cls.bar(String)"])
        scores = load_agent4lr_scores(lr, _entities())
        ids = {e.corpus_id for e in scores}
        assert ids == {"a.b$Cls.foo(int)", "a.b$Cls.bar(String)"}

    def test_pseudo_score_descending(self, tmp_path: Path) -> None:
        lr = tmp_path / "lr_result.json"
        self._write_lr_result(lr, ["a.b.Cls.foo(int)", "a.b.Cls.bar(String)"])
        scores = load_agent4lr_scores(lr, _entities())
        # foo (index 0) should outrank bar (index 1)
        items = {e.corpus_id: v for e, v in scores.items()}
        assert items["a.b$Cls.foo(int)"] > items["a.b$Cls.bar(String)"]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_agent4lr_scores(tmp_path / "nope.json", _entities()) == {}

    def test_unmatched_fqn_logged_and_skipped(self, tmp_path: Path) -> None:
        lr = tmp_path / "lr_result.json"
        self._write_lr_result(lr, ["nope.nada.Mystery.method()"])
        scores = load_agent4lr_scores(lr, _entities())
        assert scores == {}


# ---------------------------------------------------------------------------
# discover_agent4lr_configs
# ---------------------------------------------------------------------------


class TestDiscoverAgent4LRConfigs:
    def test_globs_only_dirs_with_lr_result(self, tmp_path: Path) -> None:
        base = tmp_path / "FlexFL" / "LR" / "Agent4LR"
        (base / "configA").mkdir(parents=True)
        (base / "configA" / "lr_result.json").write_text("{}")
        (base / "configB").mkdir()
        # configB has no lr_result.json — should be skipped
        (base / "configC").mkdir()
        (base / "configC" / "lr_result.json").write_text("{}")

        result = discover_agent4lr_configs(tmp_path)
        names = [name for name, _ in result]
        assert names == ["configA", "configC"]

    def test_returns_empty_when_lr_root_missing(self, tmp_path: Path) -> None:
        assert discover_agent4lr_configs(tmp_path) == []


# ---------------------------------------------------------------------------
# load_agent4lr_usage
# ---------------------------------------------------------------------------


class TestLoadAgent4LRUsage:
    @staticmethod
    def _write(path: Path, dumps: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"response_dumps": dumps}))

    def test_sums_tokens_across_dumps(self, tmp_path: Path) -> None:
        lr = tmp_path / "lr_result.json"
        self._write(
            lr,
            [
                {
                    "model": "gpt-5-mini-2025-08-07",
                    "usage": {
                        "input_tokens": 1000,
                        "input_tokens_details": {"cached_tokens": 200},
                        "output_tokens": 50,
                    },
                },
                {
                    "model": "gpt-5-mini-2025-08-07",
                    "usage": {
                        "input_tokens": 500,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 25,
                    },
                },
            ],
        )
        u = load_agent4lr_usage(lr)
        assert u.input_tokens == 1500
        assert u.cached_tokens == 200
        assert u.output_tokens == 75
        # Cost: ((1000-200)*0.25 + 200*0.025 + 50*2.0) / 1e6
        #     + ((500-0)*0.25 + 0*0.025 + 25*2.0) / 1e6
        expected = ((800 * 0.25 + 200 * 0.025 + 50 * 2.0) + (500 * 0.25 + 25 * 2.0)) / 1e6
        assert u.cost_usd == pytest.approx(expected)

    def test_unknown_model_contributes_zero_cost(self, tmp_path: Path) -> None:
        lr = tmp_path / "lr_result.json"
        self._write(
            lr,
            [
                {
                    "model": "qwen3.5:9b",
                    "usage": {
                        "input_tokens": 1000,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 200,
                    },
                }
            ],
        )
        u = load_agent4lr_usage(lr)
        assert u.input_tokens == 1000
        assert u.output_tokens == 200
        assert u.cost_usd == 0.0

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        u = load_agent4lr_usage(tmp_path / "missing.json")
        assert u == Usage.zero()

    def test_malformed_json_returns_zero(self, tmp_path: Path) -> None:
        lr = tmp_path / "lr_result.json"
        lr.write_text("not json {{{")
        u = load_agent4lr_usage(lr)
        assert u == Usage.zero()


# ---------------------------------------------------------------------------
# check_required_configs
# ---------------------------------------------------------------------------


class TestCheckRequiredConfigs:
    @staticmethod
    def _write(base: Path, cfg: str, top5: list | str | None) -> None:
        path = base / "FlexFL" / "LR" / "Agent4LR" / cfg / "lr_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if top5 == "__malformed__":
            path.write_text("{{not json")
        else:
            payload = {} if top5 is None else {"top5": top5}
            path.write_text(json.dumps(payload))

    def test_all_present_and_valid(self, tmp_path: Path) -> None:
        good = ["a", "b", "c", "d", "e"]
        self._write(tmp_path, "C1", good)
        self._write(tmp_path, "C2", good)
        ok, problems = check_required_configs(tmp_path, ["C1", "C2"])
        assert ok and problems == []

    def test_flags_missing(self, tmp_path: Path) -> None:
        self._write(tmp_path, "C1", ["a", "b", "c", "d", "e"])
        ok, problems = check_required_configs(tmp_path, ["C1", "C2"])
        assert not ok
        assert any("C2" in p and "missing" in p for p in problems)

    def test_accepts_short_top5(self, tmp_path: Path) -> None:
        # Relaxed gate: <5 entries is fine as long as ≥1 is non-empty.
        self._write(tmp_path, "C1", ["a", "b", "c"])  # only 3
        ok, problems = check_required_configs(tmp_path, ["C1"])
        assert ok and problems == []

    def test_accepts_top5_with_some_empty_entries(self, tmp_path: Path) -> None:
        # Mixed empty + populated entries: still has usable candidates.
        self._write(tmp_path, "C1", ["a", "b", "", "d", "e"])
        ok, problems = check_required_configs(tmp_path, ["C1"])
        assert ok and problems == []

    def test_flags_all_empty_top5(self, tmp_path: Path) -> None:
        # No usable candidates at all → invalid.
        self._write(tmp_path, "C1", ["", "", ""])
        ok, problems = check_required_configs(tmp_path, ["C1"])
        assert not ok
        assert any("C1" in p and "invalid" in p for p in problems)

    def test_flags_empty_list_top5(self, tmp_path: Path) -> None:
        self._write(tmp_path, "C1", [])
        ok, problems = check_required_configs(tmp_path, ["C1"])
        assert not ok
        assert any("C1" in p and "invalid" in p for p in problems)

    def test_flags_malformed_json(self, tmp_path: Path) -> None:
        self._write(tmp_path, "C1", "__malformed__")
        ok, problems = check_required_configs(tmp_path, ["C1"])
        assert not ok
        assert any("malformed" in p for p in problems)


# ---------------------------------------------------------------------------
# load_candidate_universe
# ---------------------------------------------------------------------------


class TestLoadCandidateUniverse:
    def _write_top20(self, base: Path, lines: list[str], model: str = "llama3.1_8b") -> Path:
        top20_dir = base / "top20"
        top20_dir.mkdir(parents=True, exist_ok=True)
        path = top20_dir / f"{model}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_resolves_dotted_lines_via_dotted_index(self, tmp_path: Path) -> None:
        path = self._write_top20(tmp_path, ["a.b.Cls.foo(int)", "a.b.Cls.bar(String)"])
        universe = load_candidate_universe(path, _entities())
        ids = {e.corpus_id for e in universe}
        assert ids == {"a.b$Cls.foo(int)", "a.b$Cls.bar(String)"}

    def test_skips_unresolvable_lines(self, tmp_path: Path) -> None:
        path = self._write_top20(
            tmp_path,
            ["a.b.Cls.foo(int)", "nope.nada.Mystery.method()", ""],
        )
        universe = load_candidate_universe(path, _entities())
        ids = {e.corpus_id for e in universe}
        assert ids == {"a.b$Cls.foo(int)"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "top20" / "llama3.1_8b.txt"
        assert load_candidate_universe(missing, _entities()) == set()

    def test_custom_sr_model_id(self, tmp_path: Path) -> None:
        path = self._write_top20(tmp_path, ["a.b.Cls.foo(int)"], model="other_model")
        universe = load_candidate_universe(path, _entities())
        assert {e.corpus_id for e in universe} == {"a.b$Cls.foo(int)"}


# ---------------------------------------------------------------------------
# Real Lang/1 lr_result.json
# ---------------------------------------------------------------------------


from src.common.config import get_processed_dir  # noqa: E402

REAL_PROCESSED = get_processed_dir("Lang", 1)
REAL_LR = REAL_PROCESSED / "FlexFL" / "LR" / "Agent4LR" / "M1R1-M1R1-M1R1" / "lr_result.json"


@pytest.mark.skipif(
    not REAL_LR.exists() or not (REAL_PROCESSED / "method_signatures.csv").exists(),
    reason="Lang/1 LR data not available",
)
class TestRealLang1:
    def test_lang1_lr_resolves_create_number(self) -> None:
        from src.common.method_entity import load_method_entities

        entities = load_method_entities(REAL_PROCESSED)
        scores = load_agent4lr_scores(REAL_LR, entities)
        ids = {e.corpus_id for e in scores}
        assert "org.apache.commons.lang3.math$NumberUtils.createNumber(String)" in ids
