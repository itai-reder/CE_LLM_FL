"""Word-for-word checks pinning the LR prompts to the FlexFL reference.

The strings here are the source of truth (copied verbatim from
``~/git/FlexFL/FlexFL/src/flexfl_pipeline.py`` + ``pipeline_utils.py``).
Any change must update both the reference and these assertions
together — drift is the bug we're trying to prevent.
"""

from __future__ import annotations

from src.agent4lr.io import LRBugInputs
from src.agent4lr.prompts import (
    lr_finisher_user_prompt,
    lr_planner_user_prompt,
    lr_system_prompt,
    lr_tool_loop_user_prompt,
)
from src.agent4lr.tools import (
    LRToolContext,
    execute_tool,
    tool_schemas_finisher,
    tool_schemas_loop,
)


def _inputs(*, br: bool = True, tt: bool = True, candidates: int = 3) -> LRBugInputs:
    return LRBugInputs(
        project="P",
        bug_id="1",
        sr_model_id="m",
        candidates=[f"pkg.Cls.m{i}()" for i in range(1, candidates + 1)],
        bug_report_title="T" if br else None,
        bug_report_description="D" if br else None,
        trigger_test_clean="some test\nstack" if tt else None,
    )


def test_system_prompt_both_inputs_matches_reference() -> None:
    out = lr_system_prompt(max_tool_calls=10, inputs=_inputs())
    assert out == (
        "You are a debugging assistant of our Java software.\n"
        "You will be presented with a bug report, a triggering test, "
        "and be given tools (functions) to help you access the source "
        "code of suspicious methods in the system under test (SUT) and "
        "locate the root cause for the bug.\n"
        "Your task is to locate the Top-5 most likely culprit methods "
        "based on the list of suspicious methods, the bug report, "
        "triggering test and the information you retrieve.\n"
        "You will be given 10 chances to call functions before "
        "finalizing your answer."
    )


def test_system_prompt_omits_budget_sentence_when_none() -> None:
    out = lr_system_prompt(max_tool_calls=None, inputs=_inputs())
    assert "chances to call functions" not in out


def test_system_prompt_bug_report_only() -> None:
    out = lr_system_prompt(max_tool_calls=10, inputs=_inputs(tt=False))
    assert "triggering test" not in out
    assert "with a bug report, and be given tools" in out
    assert "of suspicious methods, the bug report and the information" in out


def test_planner_prompt_block_framing_byte_equals_reference() -> None:
    out = lr_planner_user_prompt(_inputs())
    # bug report block — no \n before closing ```, no \n after Description:
    assert "The bug report is as follows:\n```\nTitle:T\nDescription:D```" in out
    # trigger-test block uses "traceback" wording
    assert "The triggering test traceback is as follows:\n```\nsome test\nstack\n```" in out
    # candidate list: "i. fqn" with space after the dot
    assert (
        "The most likely culprit methods are:\n```\n"
        "1. pkg.Cls.m1()\n2. pkg.Cls.m2()\n3. pkg.Cls.m3()\n```"
    ) in out
    # trailing instruction sentence
    assert out.endswith(
        "Before you start calling functions, reason and plan your approach "
        "to finding the buggy methods."
    )


def test_tool_loop_prompt_wording() -> None:
    assert lr_tool_loop_user_prompt(remaining=7) == "You have 7 tool calls remaining."


def test_finisher_prompt_has_no_reply_instruction() -> None:
    out = lr_finisher_user_prompt()
    assert out == (
        "Based on the available information, please provide the top-5 "
        "most likely culprit methods for the bug."
    )
    assert "Reply by calling" not in out


def test_tool_schema_descriptions_byte_equal_reference() -> None:
    loop = tool_schemas_loop()
    gs = loop[0]["function"]
    ex = loop[1]["function"]
    assert (
        gs["description"]
        == "Return the code snippet for a suggested suspicious method by its 1-based index."
    )
    assert (
        gs["parameters"]["properties"]["method_number"]["description"]
        == "1-based index into the suggested methods list"
    )
    assert ex["description"] == "Stop calling tools and proceed to rank the top-5 methods."

    fin = tool_schemas_finisher()[0]["function"]
    assert fin["description"] == "Rank the top-5 most likely culprit methods for the bug."
    assert (
        fin["parameters"]["properties"]["top_5_methods"]["description"]
        == "A list of exactly 5 1-based indices of the most likely culprit "
        "methods, ranked from most likely to least likely."
    )


def test_exit_tool_returns_exiting_string() -> None:
    ctx = LRToolContext(project="P", bug_id="1", candidates=["a", "b"])
    assert execute_tool(ctx=ctx, name="exit", args={}) == "Exiting..."


def test_rank_methods_uses_space_colon_space_format() -> None:
    ctx = LRToolContext(project="P", bug_id="1", candidates=["a", "b", "c", "d", "e", "f"])
    out = execute_tool(ctx=ctx, name="rank_methods", args={"top_5_methods": [1, 2, 3, 4, 5]})
    lines = out.splitlines()
    assert lines == [
        "Top_1 : a",
        "Top_2 : b",
        "Top_3 : c",
        "Top_4 : d",
        "Top_5 : e",
    ]
