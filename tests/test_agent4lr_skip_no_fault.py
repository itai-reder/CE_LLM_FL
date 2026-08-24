"""Tests for the LR skip-on-no-fault-in-top-20 gate.

Covers:

* :func:`_load_fault_signatures` — reads ``faults.csv`` signatures.
* :func:`_any_fault_in_candidates` — corpus/dotted normalisation and edge cases.
* :func:`run_agent4lr_for_bug` — integration: returns ``None`` and writes
  nothing when no fault appears in the top-20 candidate list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.agent4lr.agent import (
    _any_fault_in_candidates,
    _load_fault_signatures,
    candidate_list_skip_reason,
    run_agent4lr_for_bug,
)
from src.agent4lr.agents import AgentSpec
from src.agent4lr.io import LRBugInputs

# ---------------------------------------------------------------------------
# _load_fault_signatures
# ---------------------------------------------------------------------------


def test_load_fault_signatures_missing_csv_returns_empty(tmp_path: Path) -> None:
    assert _load_fault_signatures(tmp_path) == set()


def test_load_fault_signatures_reads_signature_column(tmp_path: Path) -> None:
    (tmp_path / "faults.csv").write_text(
        "path,line,signature\na/b/C.java,10,a.b$C.foo(int)\na/b/C.java,20,a.b$C.bar(String)\n",
        encoding="utf-8",
    )
    assert _load_fault_signatures(tmp_path) == {
        "a.b$C.foo(int)",
        "a.b$C.bar(String)",
    }


def test_load_fault_signatures_skips_blank_signatures(tmp_path: Path) -> None:
    (tmp_path / "faults.csv").write_text(
        "path,line,signature\na/b/C.java,10,a.b$C.foo(int)\na/b/C.java,20,\na/b/C.java,30,   \n",
        encoding="utf-8",
    )
    assert _load_fault_signatures(tmp_path) == {"a.b$C.foo(int)"}


def test_load_fault_signatures_empty_csv_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "faults.csv").write_text("path,line,signature\n", encoding="utf-8")
    assert _load_fault_signatures(tmp_path) == set()


# ---------------------------------------------------------------------------
# _any_fault_in_candidates
# ---------------------------------------------------------------------------


def test_any_fault_in_candidates_no_faults() -> None:
    assert _any_fault_in_candidates(["a.b.C.foo()"], set()) is False


def test_any_fault_in_candidates_no_candidates() -> None:
    assert _any_fault_in_candidates([], {"a.b$C.foo()"}) is False


def test_any_fault_in_candidates_dotted_match() -> None:
    # fault corpus-id $C → dotted .C must match the candidate.
    candidates = ["a.b.C.foo(int)"]
    faults = {"a.b$C.foo(int)"}
    assert _any_fault_in_candidates(candidates, faults) is True


def test_any_fault_in_candidates_no_match() -> None:
    candidates = ["x.y.Z.bar()"]
    faults = {"a.b$C.foo(int)"}
    assert _any_fault_in_candidates(candidates, faults) is False


def test_any_fault_in_candidates_corpus_form_also_matches() -> None:
    # If both sides happen to be in corpus form, that's still a match.
    candidates = ["a.b$C.foo(int)"]
    faults = {"a.b$C.foo(int)"}
    assert _any_fault_in_candidates(candidates, faults) is True


def test_any_fault_in_candidates_partial_overlap_matches() -> None:
    # Many candidates, many faults; one overlapping fault is enough.
    candidates = ["x.y.Z.bar()", "a.b.C.foo(int)", "p.q.R.baz()"]
    faults = {"a.b$C.foo(int)", "u.v$W.zap()"}
    assert _any_fault_in_candidates(candidates, faults) is True


# ---------------------------------------------------------------------------
# Integration: run_agent4lr_for_bug skip behavior
# ---------------------------------------------------------------------------


@pytest.fixture
def lr_config() -> tuple[AgentSpec, ...]:
    return (
        AgentSpec(role="planner", provider="openai", model="gpt-5-mini"),
        AgentSpec(role="tool_caller", provider="openai", model="gpt-5-mini", iterations=10),
        AgentSpec(role="finisher", provider="openai", model="gpt-5-mini"),
    )


def _make_inputs(candidates: list[str]) -> LRBugInputs:
    return LRBugInputs(
        project="Lang",
        bug_id="1",
        sr_model_id="llama3.1_8b",
        candidates=candidates,
        bug_report_title="t",
        bug_report_description="d",
        trigger_test_clean="trigger",
    )


def test_run_agent4lr_skips_when_no_fault_in_top20(
    tmp_path: Path, lr_config: tuple[AgentSpec, ...]
) -> None:
    """Integration: candidates don't intersect faults → return None, no LLM, no files."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # faults.csv has fault X, but candidates only mention Y.
    (processed_dir / "faults.csv").write_text(
        "path,line,signature\na/b/X.java,1,a.b$X.foo()\n", encoding="utf-8"
    )

    inputs = _make_inputs(["completely.different.Y.bar()"] * 20)

    with (
        patch("src.agent4lr.agent.load_lr_bug_inputs", return_value=inputs),
        patch("src.agent4lr.agent.get_lr_model_dir", return_value=model_dir),
        patch("src.agent4lr.agent.get_processed_dir", return_value=processed_dir),
        patch("src.agent4lr.agent.build_provider") as build_prov,
    ):
        result = run_agent4lr_for_bug(
            project="Lang",
            bug_id="1",
            config_name="test",
            config=lr_config,
            sr_model_id="llama3.1_8b",
            lr_model_id="test-cfg",
        )

    assert result is None
    build_prov.assert_not_called()
    assert not (model_dir / "lr_result.json").exists()
    assert not (model_dir / "top5.txt").exists()


def test_run_agent4lr_skips_when_no_faults_at_all(
    tmp_path: Path, lr_config: tuple[AgentSpec, ...]
) -> None:
    """Integration: faults.csv missing → return None, no LLM."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    inputs = _make_inputs(["a.b.X.foo()"] * 20)

    with (
        patch("src.agent4lr.agent.load_lr_bug_inputs", return_value=inputs),
        patch("src.agent4lr.agent.get_lr_model_dir", return_value=model_dir),
        patch("src.agent4lr.agent.get_processed_dir", return_value=processed_dir),
        patch("src.agent4lr.agent.build_provider") as build_prov,
    ):
        result = run_agent4lr_for_bug(
            project="Lang",
            bug_id="1",
            config_name="test",
            config=lr_config,
            sr_model_id="llama3.1_8b",
            lr_model_id="test-cfg",
        )

    assert result is None
    build_prov.assert_not_called()
    assert not (model_dir / "lr_result.json").exists()


def test_run_agent4lr_proceeds_when_fault_in_top20(
    tmp_path: Path,
    lr_config: tuple[AgentSpec, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: when fault is in top-20, the skip gate does NOT fire.

    We don't run the whole pipeline here — just confirm execution flows
    past the gate into the checkpoint-store setup. We catch that by
    intercepting ``CheckpointStore.__init__`` to raise a sentinel.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (processed_dir / "faults.csv").write_text(
        "path,line,signature\na/b/X.java,1,a.b$X.foo(int)\n", encoding="utf-8"
    )

    inputs = _make_inputs(["a.b.X.foo(int)", "y.z.Other.bar()"] * 10)

    class _Sentinel(Exception):
        pass

    def _raise_sentinel(*args: Any, **kwargs: Any) -> None:
        raise _Sentinel("reached CheckpointStore — gate did not skip")

    with (
        patch("src.agent4lr.agent.load_lr_bug_inputs", return_value=inputs),
        patch("src.agent4lr.agent.get_lr_model_dir", return_value=model_dir),
        patch("src.agent4lr.agent.get_processed_dir", return_value=processed_dir),
        patch("src.agent4lr.agent.CheckpointStore", side_effect=_raise_sentinel),
        pytest.raises(_Sentinel, match="reached CheckpointStore"),
    ):
        run_agent4lr_for_bug(
            project="Lang",
            bug_id="1",
            config_name="test",
            config=lr_config,
            sr_model_id="llama3.1_8b",
            lr_model_id="test-cfg",
        )


# ---------------------------------------------------------------------------
# candidate_list_skip_reason — the FlexFL exactly-20 gate
# ---------------------------------------------------------------------------


def test_candidate_list_skip_reason_exactly_20_valid() -> None:
    assert candidate_list_skip_reason(["a.b.C.foo()"] * 20) is None


def test_candidate_list_skip_reason_duplicates_are_legal() -> None:
    # FlexFL keeps duplicates across sources; 20 non-unique lines are valid.
    assert candidate_list_skip_reason(["x.Y.a()", "x.Y.b()"] * 10) is None


def test_candidate_list_skip_reason_too_short() -> None:
    reason = candidate_list_skip_reason(["a.b.C.foo()"] * 19)
    assert reason is not None and "19" in reason


def test_candidate_list_skip_reason_too_long() -> None:
    reason = candidate_list_skip_reason(["a.b.C.foo()"] * 21)
    assert reason is not None and "21" in reason


def test_candidate_list_skip_reason_empty() -> None:
    assert candidate_list_skip_reason([]) is not None


def test_run_agent4lr_skips_invalid_universe_before_fault_check(
    tmp_path: Path, lr_config: tuple[AgentSpec, ...]
) -> None:
    """Integration: a 19-line universe skips even though the fault IS listed."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (processed_dir / "faults.csv").write_text(
        "path,line,signature\na/b/X.java,1,a.b$X.foo(int)\n", encoding="utf-8"
    )

    inputs = _make_inputs(["a.b.X.foo(int)"] * 19)

    with (
        patch("src.agent4lr.agent.load_lr_bug_inputs", return_value=inputs),
        patch("src.agent4lr.agent.get_lr_model_dir", return_value=model_dir),
        patch("src.agent4lr.agent.get_processed_dir", return_value=processed_dir),
        patch("src.agent4lr.agent.build_provider") as build_prov,
    ):
        result = run_agent4lr_for_bug(
            project="Lang",
            bug_id="1",
            config_name="test",
            config=lr_config,
            sr_model_id="llama3.1_8b",
            lr_model_id="test-cfg",
        )

    assert result is None
    build_prov.assert_not_called()
    assert not (model_dir / "lr_result.json").exists()
    assert not (model_dir / "top5.txt").exists()
