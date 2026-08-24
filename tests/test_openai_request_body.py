"""Tests for the module-level OpenAI request/response codecs.

``build_openai_request_body`` and ``normalise_response_payload`` are the
single source of truth shared by the synchronous ``OpenAIProvider.chat``
path and the Batch API JSONL preparation, so their contract is pinned
here independently of any provider instance:

* reasoning-vs-sampling branch (gpt-5/o-series reject temperature/top_p)
* ``status`` scrubbing on reasoning items (both directions)
* nested→flat tool schema conversion and ``parallel_tool_calls=False``
* response normalisation from a plain Responses API payload dict

Also pins that the module-level ``chain_hash`` matches the
``CheckpointStore.chain_hash`` method (which now delegates to it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent4lr.agents import AgentSpec, chain_inputs_descriptor
from src.agent4lr.checkpoints import CheckpointStore, chain_hash
from src.agent4lr.providers.openai import (
    build_openai_request_body,
    normalise_response_payload,
)
from src.agent4lr.tools import tool_schemas_finisher, tool_schemas_loop

HISTORY = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "user0"},
]


class TestBuildOpenAIRequestBody:
    def test_sampling_branch_sends_temperature_and_top_p(self) -> None:
        body = build_openai_request_body(
            model="gpt-4o-mini",
            history=HISTORY,
            temperature=0.3,
            top_p=0.9,
            reasoning_effort=None,
        )
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9
        assert "reasoning" not in body

    def test_reasoning_branch_omits_temperature_and_top_p(self) -> None:
        body = build_openai_request_body(
            model="gpt-5-mini",
            history=HISTORY,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort="minimal",
        )
        assert body["reasoning"] == {"effort": "minimal"}
        assert "temperature" not in body
        assert "top_p" not in body

    def test_parallel_tool_calls_always_false(self) -> None:
        body = build_openai_request_body(model="gpt-5", history=HISTORY)
        assert body["parallel_tool_calls"] is False

    def test_history_scrubs_reasoning_status(self) -> None:
        history: list[dict[str, Any]] = [
            *HISTORY,
            {"type": "reasoning", "id": "rs_1", "summary": [], "status": None},
            {"type": "function_call", "call_id": "c1", "name": "exit", "arguments": "{}"},
        ]
        body = build_openai_request_body(model="gpt-5", history=history)
        reasoning_items = [i for i in body["input"] if i.get("type") == "reasoning"]
        assert reasoning_items == [{"type": "reasoning", "id": "rs_1", "summary": []}]
        # Other typed items pass through unchanged.
        assert history[3] in body["input"]

    def test_tools_flattened_and_tool_choice_forwarded(self) -> None:
        body = build_openai_request_body(
            model="gpt-5",
            history=HISTORY,
            tools=tool_schemas_loop(),
            tool_choice="required",
        )
        assert body["tool_choice"] == "required"
        names = {t["name"] for t in body["tools"]}
        assert names == {"get_snippet_of_method", "exit"}
        assert all(t["type"] == "function" and "parameters" in t for t in body["tools"])

    def test_no_tools_omits_tools_and_tool_choice_keys(self) -> None:
        body = build_openai_request_body(
            model="gpt-5", history=HISTORY, tools=None, tool_choice=None
        )
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_body_is_json_serialisable(self) -> None:
        body = build_openai_request_body(
            model="gpt-5-nano",
            history=HISTORY,
            tools=tool_schemas_finisher(),
            tool_choice="required",
            reasoning_effort="low",
        )
        line = json.dumps(
            {"custom_id": "x", "method": "POST", "url": "/v1/responses", "body": body}
        )
        assert json.loads(line)["body"] == body


class TestNormaliseResponsePayload:
    def test_extracts_text_reasoning_and_tool_calls(self) -> None:
        raw = {
            "id": "resp_1",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "status": "completed"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "thinking done"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c9",
                    "name": "get_snippet_of_method",
                    "arguments": '{"method_number": 3}',
                },
            ],
        }
        resp = normalise_response_payload(raw)
        assert resp["content"] == "thinking done"
        assert resp["tool_calls"] == [
            {"name": "get_snippet_of_method", "arguments": {"method_number": 3}, "call_id": "c9"}
        ]
        # Reasoning item kept in new_items with status stripped.
        assert resp["new_items"][0] == {"type": "reasoning", "id": "rs_1", "summary": []}
        assert resp["raw"] is raw

    def test_malformed_arguments_become_empty_dict(self) -> None:
        raw = {
            "output": [
                {"type": "function_call", "call_id": "c1", "name": "exit", "arguments": "{oops"}
            ]
        }
        resp = normalise_response_payload(raw)
        assert resp["tool_calls"] == [{"name": "exit", "arguments": {}, "call_id": "c1"}]

    def test_empty_output_yields_empty_response(self) -> None:
        resp = normalise_response_payload({"output": []})
        assert resp["content"] == ""
        assert resp["tool_calls"] == []
        assert resp["new_items"] == []


class TestModuleLevelChainHash:
    def test_matches_store_method(self, tmp_path: Path) -> None:
        desc = chain_inputs_descriptor(
            project="Lang",
            bug_id=1,
            dataset="defects4j",
            sr_model_id="llama3.1_8b",
            candidate_source="FlexFL/SR/rankings/top20/llama3.1_8b.txt",
            input_keys=("bug_report", "trigger_test"),
        )
        chain = [
            AgentSpec(role="planner", provider="openai", model="gpt-5-mini"),
            AgentSpec(role="tool_caller", provider="openai", model="gpt-5-nano", iterations=10),
        ]
        store = CheckpointStore(tmp_path)
        assert chain_hash(desc, chain) == store.chain_hash(desc, chain)
        # Identity dicts hash identically to AgentSpec instances.
        assert chain_hash(desc, [s.identity() for s in chain]) == chain_hash(desc, chain)
