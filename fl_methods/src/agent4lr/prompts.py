"""Prompt templates for the Agent4LR LLM pipeline.

Wording is byte-for-byte aligned with FlexFL's reference pipeline
(``~/git/FlexFL/FlexFL/src/flexfl_pipeline.py`` +
``pipeline_utils.py``). Per-input-vocabulary phrasing collapses when
the corresponding input is absent from :class:`LRBugInputs`.
"""

from __future__ import annotations

from src.agent4lr.io import LRBugInputs


def _input_vocabulary(*, has_br: bool, has_tt: bool) -> tuple[str, str] | None:
    """Render the ``{a_input}`` / ``{the_input}`` clauses.

    Returns ``("a bug report, a triggering test", "the bug report, the
    triggering test")`` when both are present (or the appropriate
    subset). Returns ``None`` when neither input is available — the
    caller then drops the dependent system-prompt clauses entirely.
    """
    names: list[str] = []
    if has_br:
        names.append("bug report")
    if has_tt:
        names.append("triggering test")
    if not names:
        return None
    a_input = ", ".join(f"a {n}" for n in names)
    the_input = "the " + ", ".join(names)
    return a_input, the_input


def lr_system_prompt(*, max_tool_calls: int | None, inputs: LRBugInputs) -> str:
    """Return the system prompt fed to every LR agent.

    Four newline-separated sentences::

        You are a debugging assistant of our Java software.
        You will be presented with {a_input}, and be given tools (functions) ...
        Your task is to locate the Top-5 most likely culprit methods ...
        You will be given {N} chances to call functions before finalizing your answer.

    The fourth sentence is omitted when ``max_tool_calls`` is ``None``.
    Tool descriptions are *not* inlined into the prompt — the provider
    sends the structured tool schemas alongside.
    """
    vocab = _input_vocabulary(has_br=inputs.has_bug_report(), has_tt=inputs.has_trigger_test())
    sentences = ["You are a debugging assistant of our Java software."]
    if vocab is not None:
        a_input, the_input = vocab
        sentences.append(
            f"You will be presented with {a_input}, and be given tools (functions) "
            "to help you access the source code of suspicious methods in the system "
            "under test (SUT) and locate the root cause for the bug."
        )
        sentences.append(
            "Your task is to locate the Top-5 most likely culprit methods based on "
            f"the list of suspicious methods, {the_input} and the information you "
            "retrieve."
        )
    else:
        sentences.append(
            "You will be given tools (functions) to help you access the source code "
            "of suspicious methods in the system under test (SUT) and locate the "
            "root cause for the bug."
        )
        sentences.append(
            "Your task is to locate the Top-5 most likely culprit methods based on "
            "the list of suspicious methods and the information you retrieve."
        )
    if max_tool_calls is not None:
        sentences.append(
            f"You will be given {max_tool_calls} chances to call functions before "
            "finalizing your answer."
        )
    return "\n".join(sentences)


def lr_planner_user_prompt(inputs: LRBugInputs) -> str:
    """Initial user message: bug report + trigger test + numbered candidate list."""
    blocks: list[str] = []
    br = inputs.bug_report_text_block()
    if br is not None:
        blocks.append(br)
    tt = inputs.trigger_test_text_block()
    if tt is not None:
        blocks.append(tt)
    blocks.append(inputs.candidate_list_text_block())
    blocks.append(
        "Before you start calling functions, reason and plan your approach to "
        "finding the buggy methods."
    )
    return "\n".join(blocks)


def lr_tool_loop_user_prompt(*, remaining: int) -> str:
    """Tool-call prompt issued each loop iteration."""
    return f"You have {remaining} tool calls remaining."


def lr_finisher_user_prompt() -> str:
    """Final prompt asking the model to call ``rank_methods``."""
    return (
        "Based on the available information, please provide the top-5 most "
        "likely culprit methods for the bug."
    )
