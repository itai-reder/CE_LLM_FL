"""Unit tests for ``src.agent4lr.providers``.

Network IO is mocked via ``monkeypatch``; we test request shaping,
response normalisation, the ``tool_choice='required'`` retry path, and
the context-overflow detection branch.

OpenAI provider tests cover only the parts that don't require the
``openai`` SDK (which the cefl conda env may not have installed): the
normalisation helper and the tool-schema flattening.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent4lr.providers import ContextOverflowError
from src.agent4lr.providers.ollama import OllamaProvider, _context_overflow_signal
from src.agent4lr.providers.openai import _flatten_tools

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, *, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"{self.status_code} error: {self.text}")


def _ok(content: str = "ok", tool_calls: list[dict[str, Any]] | None = None) -> _FakeResponse:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return _FakeResponse(payload={"message": msg})


def test_ollama_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaProvider(model="x", base_url="")


def test_ollama_provider_posts_payload_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = kw["json"]
        return _ok(
            content="thinking",
            tool_calls=[
                {
                    "function": {
                        "name": "get_snippet_of_method",
                        "arguments": {"method_number": 3},
                    },
                    "id": "call-1",
                }
            ],
        )

    monkeypatch.setattr("src.agent4lr.providers.ollama.requests.post", fake_post)
    p = OllamaProvider(model="llama3.1:8b", base_url="http://x:1/", temperature=0.1)
    out = p.chat(history=[{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert captured["url"] == "http://x:1/api/chat"
    assert captured["json"]["model"] == "llama3.1:8b"
    assert captured["json"]["options"]["temperature"] == 0.1
    assert "tools" in captured["json"]
    assert out["content"] == "thinking"
    assert out["tool_calls"] == [
        {
            "name": "get_snippet_of_method",
            "arguments": {"method_number": 3},
            "call_id": "call-1",
        }
    ]
    # new_items mirrors the response in Responses-API-native shape.
    assert {"role": "assistant", "content": "thinking"} in out["new_items"]
    fc_items = [it for it in out["new_items"] if it.get("type") == "function_call"]
    assert len(fc_items) == 1
    assert fc_items[0]["name"] == "get_snippet_of_method"
    assert fc_items[0]["call_id"] == "call-1"
    # arguments are JSON-serialised in the Responses-API form
    assert json.loads(fc_items[0]["arguments"]) == {"method_number": 3}


def test_ollama_provider_decodes_string_encoded_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        return _ok(tool_calls=[{"function": {"name": "f", "arguments": json.dumps({"x": 1})}}])

    monkeypatch.setattr("src.agent4lr.providers.ollama.requests.post", fake_post)
    p = OllamaProvider(model="m", base_url="http://x:1")
    out = p.chat(history=[{"role": "user", "content": "hi"}])
    assert out["tool_calls"][0]["arguments"] == {"x": 1}


def test_ollama_provider_retries_when_required_yields_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return _ok(content="words only", tool_calls=[])
        return _ok(tool_calls=[{"function": {"name": "exit", "arguments": {}}, "id": "call-2"}])

    monkeypatch.setattr("src.agent4lr.providers.ollama.requests.post", fake_post)
    p = OllamaProvider(model="m", base_url="http://x:1")
    out = p.chat(
        history=[{"role": "user", "content": "do it"}],
        tools=[{"type": "function"}],
        tool_choice="required",
    )
    assert calls["n"] == 2
    assert out["tool_calls"] and out["tool_calls"][0]["name"] == "exit"


def test_ollama_provider_raises_context_overflow_on_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        return _FakeResponse(status_code=500, text="model context length exceeded for prompt")

    monkeypatch.setattr("src.agent4lr.providers.ollama.requests.post", fake_post)
    p = OllamaProvider(model="m", base_url="http://x:1")
    with pytest.raises(ContextOverflowError):
        p.chat(history=[{"role": "user", "content": "hi"}])


def test_ollama_history_projection_drops_reasoning_and_folds_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning items are dropped; function_call/output pairs become
    assistant tool_calls + tool-role messages when projecting an
    input_list down to chat-completions for Ollama."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        captured["messages"] = kw["json"]["messages"]
        return _ok(content="ack", tool_calls=[])

    monkeypatch.setattr("src.agent4lr.providers.ollama.requests.post", fake_post)
    p = OllamaProvider(model="m", base_url="http://x:1")
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"id": "rs_x", "type": "reasoning", "summary": []},
        {
            "type": "function_call",
            "name": "get_snippet_of_method",
            "arguments": json.dumps({"method_number": 1}),
            "call_id": "call-A",
        },
        {
            "type": "function_call_output",
            "call_id": "call-A",
            "output": "snippet body",
        },
    ]
    p.chat(history=history)
    msgs = captured["messages"]
    # No reasoning item should appear in the projected messages.
    assert not any(m.get("type") == "reasoning" for m in msgs)
    # The function_call should appear as an assistant turn with tool_calls.
    asst_with_call = next(
        (m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")), None
    )
    assert asst_with_call is not None
    assert asst_with_call["tool_calls"][0]["function"]["name"] == "get_snippet_of_method"
    assert asst_with_call["tool_calls"][0]["function"]["arguments"] == {"method_number": 1}
    # The function_call_output should appear as a tool-role message.
    tool_msg = next((m for m in msgs if m.get("role") == "tool"), None)
    assert tool_msg is not None
    assert tool_msg["tool_name"] == "get_snippet_of_method"
    assert tool_msg["content"] == "snippet body"


def test_context_overflow_signal_matches_common_phrasings() -> None:
    assert _context_overflow_signal("Context length exceeded")
    assert _context_overflow_signal("the context window is too small")
    assert not _context_overflow_signal("403 Forbidden")


def test_required_addendum_appends_to_last_user_message() -> None:
    msgs = [{"role": "user", "content": "first"}]
    appended = OllamaProvider._with_required_addendum(msgs)
    assert appended[-1]["content"].startswith("first")
    assert "IMPORTANT" in appended[-1]["content"]


def test_required_addendum_injects_new_user_message_if_last_is_not_user() -> None:
    msgs = [{"role": "assistant", "content": "thought"}]
    appended = OllamaProvider._with_required_addendum(msgs)
    assert len(appended) == 2
    assert appended[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# OpenAI helpers (no SDK required)
# ---------------------------------------------------------------------------


def test_flatten_tools_converts_nested_to_openai_shape() -> None:
    nested = [
        {
            "type": "function",
            "function": {
                "name": "x",
                "description": "do x",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    flat = _flatten_tools(nested)
    assert flat == [
        {
            "type": "function",
            "name": "x",
            "description": "do x",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_flatten_tools_passes_through_already_flat_entries() -> None:
    flat_in = [{"type": "function", "name": "x", "description": "", "parameters": {}}]
    out = _flatten_tools(flat_in)
    assert out[0]["name"] == "x"
