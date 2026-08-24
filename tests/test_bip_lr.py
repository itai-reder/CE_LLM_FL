"""Agent4LR (FlexFL-LR) BugsInPy support tests.

Covers:

  * ``LRToolContext(dataset="bugsinpy")`` routes ``get_snippet_of_method`` to the
    **BIP** SR corpus and returns **decoded multi-line** Python (not JSON-quoted);
  * the default ``LRToolContext()`` still routes to Defects4J (single-line, no
    JSON decode) — regression;
  * the BugsInPy readiness gate in ``run_lr.run_lr_for_bug`` skips (no LLM, no
    tracker work) when the SR corpus or the SR top-20 candidate list is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import run_lr

from src.agent4lr.tools import LRToolContext, get_snippet_of_method

# ---------------------------------------------------------------------------
# get_snippet_of_method dataset routing
# ---------------------------------------------------------------------------

_BIP_METHODS = "pysnooper.pysnooper$get_write_function(output)\n"
_BIP_CODE = "def get_write_function(output):\n    if output is None:\n        return None\n    return output"


def _write_bip_corpus(sr_dir: Path) -> None:
    """Write a BIP-format SR corpus (codes JSON-encoded, one per physical line)."""
    sr_dir.mkdir(parents=True, exist_ok=True)
    (sr_dir / "corpus_methods.txt").write_text(_BIP_METHODS, encoding="utf-8")
    (sr_dir / "corpus_codes.txt").write_text(json.dumps(_BIP_CODE) + "\n", encoding="utf-8")


def test_get_snippet_routes_to_bip_corpus_and_decodes_multiline(tmp_path: Path) -> None:
    """dataset='bugsinpy' → reads the BIP SR dir and JSON-decodes to real newlines."""
    sr_dir = tmp_path / "FlexFL" / "SR"
    _write_bip_corpus(sr_dir)

    # The candidate list is the dotted form (as written to top20/<id>.txt).
    cands = ["pysnooper.pysnooper.get_write_function(output)"]
    ctx = LRToolContext(project="PySnooper", bug_id="3", candidates=cands, dataset="bugsinpy")

    with patch("src.agent4sr.function_call.get_sr_dir", return_value=sr_dir):
        out = get_snippet_of_method(ctx=ctx, method_number=1)

    assert "```" in out
    body = out.split("```")[1]
    assert "\n" in body  # multi-line
    assert "\\n" not in body  # NOT JSON-escaped
    assert "def get_write_function(output):" in body
    assert "pysnooper.pysnooper.get_write_function(output)" in out


def test_lr_tool_context_defaults_to_defects4j(tmp_path: Path) -> None:
    """Default dataset stays 'defects4j' and the D4J code path does NOT JSON-decode."""
    sr_dir = tmp_path / "FlexFL" / "SR"
    sr_dir.mkdir(parents=True)
    # D4J codes are single-line plain text — deliberately NOT valid JSON, so if the
    # BIP decode path were wrongly taken json.loads would raise.
    (sr_dir / "corpus_methods.txt").write_text("a.b$C.foo(int)\n", encoding="utf-8")
    (sr_dir / "corpus_codes.txt").write_text("int foo(int x) { return x; }\n", encoding="utf-8")

    ctx = LRToolContext(project="Lang", bug_id="1", candidates=["a.b.C.foo(int)"])
    assert ctx.dataset == "defects4j"

    with patch("src.agent4sr.function_call.get_sr_dir", return_value=sr_dir):
        out = get_snippet_of_method(ctx=ctx, method_number=1)

    assert "int foo(int x) { return x; }" in out


# ---------------------------------------------------------------------------
# run_lr.run_lr_for_bug — BugsInPy readiness gate
# ---------------------------------------------------------------------------


def _gate_kwargs() -> dict:
    return {
        "project": "PySnooper",
        "bug_id": "3",
        "config_name": "test",
        "config": (),
        "sr_model_id": "llama3.1_8b",
        "input_keys": ("bug_report", "trigger_test"),
        "dataset": "bugsinpy",
        "ollama_base_url": "",
        "ollama_verify": True,
        "openai_api_key": None,
    }


def test_run_lr_for_bug_skips_when_corpus_missing() -> None:
    """Gate fires (audit/corpus not ready) → SKIPPED, no tracker/agent work."""
    with (
        patch("run_lr.bip_run_skip_reason", return_value="SR corpus missing"),
        patch("run_lr.load_tracker") as load_tracker,
        patch("run_lr.run_agent4lr_for_bug") as run_agent,
    ):
        summary = run_lr.run_lr_for_bug(**_gate_kwargs())

    assert summary["status"] == "SKIPPED"
    assert "corpus missing" in summary["skip_reason"]
    load_tracker.assert_not_called()
    run_agent.assert_not_called()


def test_run_lr_for_bug_skips_when_top20_missing(tmp_path: Path) -> None:
    """Audit/corpus ready but SR top-20 absent → SKIPPED with the top-20 reason."""
    missing = tmp_path / "does_not_exist.txt"
    with (
        patch("run_lr.bip_run_skip_reason", return_value=None),
        patch("run_lr.get_lr_candidate_file", return_value=missing),
        patch("run_lr.load_tracker") as load_tracker,
        patch("run_lr.run_agent4lr_for_bug") as run_agent,
    ):
        summary = run_lr.run_lr_for_bug(**_gate_kwargs())

    assert summary["status"] == "SKIPPED"
    assert "SR top-20 missing" in summary["skip_reason"]
    load_tracker.assert_not_called()
    run_agent.assert_not_called()
