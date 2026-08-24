"""Parity and unit tests for the Agent4LR per-API-call stepper.

The centerpiece is a byte-parity test: ``run_agent4lr_for_bug`` (now
driven through ``stepper.build_request`` / ``stepper.apply_response``)
must produce ``input_list`` / ``messages`` / ``raw_responses`` identical
to a hand-rolled transcription of the **pre-refactor** sync loop
(``_golden_transcript`` below is a verbatim port of the old slot bodies
from ``agent.py``). If the stepper ever diverges from the historical
transcript construction, this fails.

Unit tests pin the stepper contracts the batch workflow relies on:
request rebuilding is deterministic (idempotent resends), completed
slots refuse further steps, a no-tool-call round still consumes the
iteration budget, tool outputs are appended before the exit-break, and
the finisher's ``Top_N`` block lands in ``messages`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.agent4lr.agent import run_agent4lr_for_bug
from src.agent4lr.agents import AgentSpec, chain_inputs_descriptor
from src.agent4lr.checkpoints import CheckpointStore, PendingSlotState
from src.agent4lr.io import LRBugInputs
from src.agent4lr.prompts import (
    lr_finisher_user_prompt,
    lr_planner_user_prompt,
    lr_system_prompt,
    lr_tool_loop_user_prompt,
)
from src.agent4lr.providers import NormalisedResponse
from src.agent4lr.stepper import (
    apply_response,
    build_request,
    initial_state,
    slot_is_complete,
    state_for_slot,
)
from src.agent4lr.tools import LRToolContext, execute_tool, rank_methods

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Exactly 20 lines (duplicates allowed) — FlexFL's combination rule, enforced
# by the candidate_list_skip_reason gate in run_agent4lr_for_bug.
CANDIDATES = ["a.b.X.foo(int)", "y.z.Other.bar()"] * 10
CORPUS_METHODS = ["a.b$X.foo(int)", "y.z$Other.bar()"]
CORPUS_CODES = ["int foo() { return 1; }", "void bar() {}"]

CHAIN = (
    AgentSpec(role="planner", provider="openai", model="gpt-5-mini"),
    AgentSpec(role="tool_caller", provider="openai", model="gpt-5-nano", iterations=10),
    AgentSpec(role="finisher", provider="openai", model="gpt-5-mini", reasoning_effort="low"),
)


def _make_inputs() -> LRBugInputs:
    return LRBugInputs(
        project="Lang",
        bug_id="1",
        sr_model_id="llama3.1_8b",
        candidates=list(CANDIDATES),
        bug_report_title="NPE in foo",
        bug_report_description="foo throws NPE on negative input",
        trigger_test_clean="public void testFoo() { foo(-1); }",
    )


def _inputs_desc() -> dict[str, Any]:
    return chain_inputs_descriptor(
        project="Lang",
        bug_id="1",
        dataset="defects4j",
        sr_model_id="llama3.1_8b",
        candidate_source="FlexFL/SR/rankings/top20/llama3.1_8b.txt",
        input_keys=("bug_report", "trigger_test"),
    )


def _resp(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    raw_id: str = "resp",
) -> NormalisedResponse:
    """Build a NormalisedResponse with Responses-API-shaped new_items."""
    tcs = tool_calls or []
    new_items: list[dict[str, Any]] = []
    if content:
        new_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    for i, tc in enumerate(tcs):
        new_items.append(
            {
                "type": "function_call",
                "call_id": tc.get("call_id") or f"{raw_id}_c{i}",
                "name": tc["name"],
                "arguments": str(tc["arguments"]),
            }
        )
    return cast(
        NormalisedResponse,
        {
            "content": content,
            "tool_calls": tcs,
            "new_items": new_items,
            "raw": {"id": raw_id},
        },
    )


class FakeProvider:
    """Scripted provider: pops one canned response per chat() call."""

    def __init__(self, responses: list[NormalisedResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **opts: Any,
    ) -> NormalisedResponse:
        self.calls.append(
            {
                "history": [dict(m) for m in history],
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return self._responses.pop(0)


# The scripted run: planner text; tool_caller = snippet call, a no-tool-call
# round, exit; finisher = rank_methods. Exercises every branch of the loop.
def _scripted_responses() -> list[NormalisedResponse]:
    return [
        _resp(content="Plan: inspect method 1 first.", raw_id="r_plan"),
        _resp(
            content="Let me look at the first method.",
            tool_calls=[
                {
                    "name": "get_snippet_of_method",
                    "arguments": {"method_number": 1},
                    "call_id": "c1",
                }
            ],
            raw_id="r_t1",
        ),
        _resp(content="Hmm, thinking without calling tools.", raw_id="r_t2"),
        _resp(
            tool_calls=[{"name": "exit", "arguments": {}, "call_id": "c3"}],
            raw_id="r_t3",
        ),
        _resp(
            tool_calls=[
                {
                    "name": "rank_methods",
                    "arguments": {"top_5_methods": [1, 2, 1, 2, 1]},
                    "call_id": "c4",
                }
            ],
            raw_id="r_fin",
        ),
    ]


@pytest.fixture
def patched_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agent4sr import function_call as sr_fc

    monkeypatch.setattr(sr_fc, "load_corpus_methods", lambda *a, **k: list(CORPUS_METHODS))
    monkeypatch.setattr(sr_fc, "load_corpus_codes", lambda *a, **k: list(CORPUS_CODES))


# ---------------------------------------------------------------------------
# Golden reference: verbatim port of the PRE-REFACTOR sync loop bodies
# ---------------------------------------------------------------------------


def _render_function_call(name: str, arguments: dict[str, Any]) -> str:
    args_str = ", ".join(str(v) for v in arguments.values())
    return f"[Function call: {name}({args_str})]"


def _golden_transcript(
    chain: tuple[AgentSpec, ...],
    responses: list[NormalisedResponse],
    inputs: LRBugInputs,
    ctx: LRToolContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Replay the original (pre-refactor) agent.py slot bodies."""
    queue = list(responses)
    system = lr_system_prompt(max_tool_calls=10, inputs=inputs)
    user0 = lr_planner_user_prompt(inputs)
    seed = [
        {"role": "system", "content": system},
        {"role": "user", "content": user0},
    ]
    input_list = [dict(m) for m in seed]
    messages = [dict(m) for m in seed]
    raw_responses: list[dict[str, Any]] = []
    top5_indices: list[int] = []

    for spec in chain:
        if spec.role == "planner":
            resp = queue.pop(0)
            raw_responses.append(resp["raw"])
            input_list.extend(resp["new_items"])
            if resp["content"]:
                messages.append({"role": "assistant", "content": resp["content"]})
        elif spec.role == "tool_caller":
            iters = spec.iterations or 10
            exited = False
            for i in range(iters):
                remaining = iters - i
                reminder = {
                    "role": "user",
                    "content": lr_tool_loop_user_prompt(remaining=remaining),
                }
                input_list.append(dict(reminder))
                messages.append(dict(reminder))
                resp = queue.pop(0)
                raw_responses.append(resp["raw"])
                input_list.extend(resp["new_items"])

                tcs = resp["tool_calls"]
                pieces: list[str] = []
                if resp["content"]:
                    pieces.append(resp["content"])
                for tc in tcs:
                    pieces.append(_render_function_call(tc["name"], tc["arguments"]))
                if pieces:
                    messages.append({"role": "assistant", "content": "\n".join(pieces)})

                if not tcs:
                    continue

                for tc in tcs:
                    tool_output = execute_tool(ctx=ctx, name=tc["name"], args=tc["arguments"])
                    call_id = tc.get("call_id") or ""
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": tool_output,
                        }
                    )
                    messages.append({"role": "user", "content": tool_output})

                if any(tc["name"] == "exit" for tc in tcs):
                    exited = True
                    break
            del exited
        elif spec.role == "finisher":
            handoff = {"role": "user", "content": lr_finisher_user_prompt()}
            input_list.append(dict(handoff))
            messages.append(dict(handoff))
            resp = queue.pop(0)
            raw_responses.append(resp["raw"])
            input_list.extend(resp["new_items"])

            tcs = resp["tool_calls"]
            for tc in tcs:
                if tc["name"] == "rank_methods":
                    top5_indices = [int(v) for v in (tc["arguments"].get("top_5_methods") or [])]

            pieces = []
            if resp["content"]:
                pieces.append(resp["content"])
            for tc in tcs:
                pieces.append(_render_function_call(tc["name"], tc["arguments"]))
            if pieces:
                messages.append({"role": "assistant", "content": "\n".join(pieces)})

            if top5_indices:
                resolved = rank_methods(ctx=ctx, top_5_methods=top5_indices)
                top_n_block = "\n".join(
                    f"Top_{i} : {fqn}" for i, fqn in enumerate(resolved, start=1)
                )
                if top_n_block:
                    messages.append({"role": "user", "content": top_n_block})
    return input_list, messages, raw_responses, top5_indices


# ---------------------------------------------------------------------------
# Parity: refactored runner vs golden pre-refactor transcript
# ---------------------------------------------------------------------------


def test_runner_matches_golden_pre_refactor_transcript(
    tmp_path: Path, patched_corpus: None
) -> None:
    inputs = _make_inputs()
    ctx = LRToolContext(project="Lang", bug_id="1", candidates=list(CANDIDATES))

    golden_input_list, golden_messages, golden_raws, golden_top5 = _golden_transcript(
        CHAIN, _scripted_responses(), inputs, ctx
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "faults.csv").write_text(
        "path,line,signature\na/b/X.java,1,a.b$X.foo(int)\n", encoding="utf-8"
    )
    model_dir = tmp_path / "model"
    ckpt_dir = tmp_path / "checkpoints"
    provider = FakeProvider(_scripted_responses())

    with (
        patch("src.agent4lr.agent.load_lr_bug_inputs", return_value=inputs),
        patch("src.agent4lr.agent.get_lr_model_dir", return_value=model_dir),
        patch("src.agent4lr.agent.get_processed_dir", return_value=processed_dir),
        patch("src.agent4lr.agent.get_lr_checkpoints_dir", return_value=ckpt_dir),
        patch("src.agent4lr.agent._provider_for", return_value=provider),
    ):
        result = run_agent4lr_for_bug(
            project="Lang",
            bug_id="1",
            config_name="parity-test",
            config=CHAIN,
            sr_model_id="llama3.1_8b",
            lr_model_id="parity-test",
        )

    assert result is not None
    assert result.input_list == golden_input_list
    assert result.messages == golden_messages
    assert result.response_dumps == golden_raws
    assert result.top5_indices == golden_top5
    assert result.top5 == rank_methods(ctx=ctx, top_5_methods=golden_top5)

    # One checkpoint per completed slot.
    assert len(list(ckpt_dir.glob("*.json"))) == len(CHAIN)
    # Requests carried the right shapes per role.
    assert provider.calls[0]["tools"] is None
    assert provider.calls[0]["tool_choice"] == "none"
    assert provider.calls[1]["tool_choice"] == "required"
    assert provider.calls[-1]["tool_choice"] == "required"
    # Result files written.
    assert (model_dir / "lr_result.json").exists()
    assert (model_dir / "top5.txt").read_text(encoding="utf-8").splitlines() == result.top5


# ---------------------------------------------------------------------------
# Stepper unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(patched_corpus: None) -> LRToolContext:
    return LRToolContext(project="Lang", bug_id="1", candidates=list(CANDIDATES))


def _fresh_state(slot_index: int = 0) -> PendingSlotState:
    state = initial_state(inputs_desc=_inputs_desc(), inputs=_make_inputs(), chain=CHAIN)
    if slot_index == 0:
        return state
    raise NotImplementedError


class TestBuildRequest:
    def test_deterministic_rebuild(self) -> None:
        state = _fresh_state()
        r1 = build_request(state)
        r2 = build_request(state)
        assert r1 == r2  # idempotent resend of a lost batch request

    def test_planner_shape(self) -> None:
        req = build_request(_fresh_state())
        assert req.spec.role == "planner"
        assert req.tools is None
        assert req.tool_choice == "none"
        assert req.history == _fresh_state().input_list  # no ephemeral message

    def test_tool_caller_appends_reminder_without_mutating_state(self, ctx: LRToolContext) -> None:
        state = _fresh_state()
        step = apply_response(state, _resp(content="plan"), ctx)
        tc_state = PendingSlotState(
            inputs_descriptor=state.inputs_descriptor,
            completed_chain=[CHAIN[0].identity()],
            slot_spec=CHAIN[1].identity(),
            calls_made=0,
            exited=False,
            input_list=step.state.input_list,
            messages=step.state.messages,
            raw_responses=step.state.raw_responses,
        )
        req = build_request(tc_state)
        assert req.history[-1] == {"role": "user", "content": "You have 10 tool calls remaining."}
        assert len(req.history) == len(tc_state.input_list) + 1
        # State itself unchanged — the reminder is ephemeral until applied.
        assert tc_state.input_list[-1] != req.history[-1]

    def test_raises_on_completed_slot(self, ctx: LRToolContext) -> None:
        state = _fresh_state()
        step = apply_response(state, _resp(content="plan"), ctx)
        assert step.slot_completed
        with pytest.raises(ValueError, match="completed slot"):
            build_request(step.state)


class TestApplyResponse:
    def test_planner_completes_in_one_call(self, ctx: LRToolContext) -> None:
        step = apply_response(_fresh_state(), _resp(content="plan"), ctx)
        assert step.slot_completed
        assert step.top5_indices is None
        assert step.state.calls_made == 1
        assert step.state.messages[-1] == {"role": "assistant", "content": "plan"}

    def test_no_tool_call_round_consumes_budget(self, ctx: LRToolContext) -> None:
        spec = AgentSpec(role="tool_caller", provider="openai", model="m", iterations=2)
        state = PendingSlotState(
            inputs_descriptor=_inputs_desc(),
            completed_chain=[CHAIN[0].identity()],
            slot_spec=spec.identity(),
            calls_made=0,
            exited=False,
            input_list=[{"role": "system", "content": "s"}],
            messages=[{"role": "system", "content": "s"}],
            raw_responses=[],
        )
        step1 = apply_response(state, _resp(content="thinking only"), ctx)
        assert not step1.slot_completed
        assert step1.state.calls_made == 1
        step2 = apply_response(step1.state, _resp(content="still thinking"), ctx)
        assert step2.slot_completed  # budget exhausted at iterations=2
        assert not step2.state.exited

    def test_exit_outputs_appended_before_completion(self, ctx: LRToolContext) -> None:
        spec = AgentSpec(role="tool_caller", provider="openai", model="m", iterations=10)
        state = PendingSlotState(
            inputs_descriptor=_inputs_desc(),
            completed_chain=[CHAIN[0].identity()],
            slot_spec=spec.identity(),
            calls_made=3,
            exited=False,
            input_list=[{"role": "system", "content": "s"}],
            messages=[{"role": "system", "content": "s"}],
            raw_responses=[],
        )
        step = apply_response(
            state, _resp(tool_calls=[{"name": "exit", "arguments": {}, "call_id": "c"}]), ctx
        )
        assert step.slot_completed
        assert step.state.exited
        # Reminder reflects calls_made=3 → 7 remaining; then the exit round.
        assert step.state.messages[1] == {
            "role": "user",
            "content": "You have 7 tool calls remaining.",
        }
        assert step.state.messages[-2] == {
            "role": "assistant",
            "content": "[Function call: exit()]",
        }
        assert step.state.messages[-1] == {"role": "user", "content": "Exiting..."}
        assert step.state.input_list[-1]["type"] == "function_call_output"
        assert step.state.input_list[-1]["output"] == "Exiting..."

    def test_finisher_top_n_block_in_messages_only(self, ctx: LRToolContext) -> None:
        spec = AgentSpec(role="finisher", provider="openai", model="m")
        state = PendingSlotState(
            inputs_descriptor=_inputs_desc(),
            completed_chain=[CHAIN[0].identity(), CHAIN[1].identity()],
            slot_spec=spec.identity(),
            calls_made=0,
            exited=False,
            input_list=[{"role": "system", "content": "s"}],
            messages=[{"role": "system", "content": "s"}],
            raw_responses=[],
        )
        step = apply_response(
            state,
            _resp(
                tool_calls=[
                    {
                        "name": "rank_methods",
                        "arguments": {"top_5_methods": [1, 2, 1, 2, 1]},
                        "call_id": "c",
                    }
                ]
            ),
            ctx,
        )
        assert step.slot_completed
        assert step.top5_indices == [1, 2, 1, 2, 1]
        top_n = step.state.messages[-1]["content"]
        assert top_n.startswith("Top_1 : a.b.X.foo(int)")
        # Top_N is a messages-only artifact — never sent back to the API.
        assert all(top_n != item.get("content") for item in step.state.input_list)


class TestStateForSlot:
    def test_seeds_from_checkpoint(self, tmp_path: Path) -> None:
        desc = _inputs_desc()
        store = CheckpointStore(tmp_path)
        input_list = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        store.save(
            inputs_descriptor=desc,
            completed_chain=[CHAIN[0]],
            input_list=input_list,
            messages=input_list,
            raw_responses=[{"id": "r"}],
        )
        ckpt = store.lookup(store.chain_hash(desc, [CHAIN[0]]))
        assert ckpt is not None
        state = state_for_slot(inputs_desc=desc, chain=CHAIN, slot_index=1, ckpt=ckpt)
        assert state.completed_chain == [CHAIN[0].identity()]
        assert state.slot_spec == CHAIN[1].identity()
        assert state.calls_made == 0
        assert state.input_list == input_list
        # The pending state's hash targets slot 1's completed checkpoint.
        assert state.slot_hash() == store.chain_hash(desc, [CHAIN[0], CHAIN[1]])
        assert not slot_is_complete(state)
