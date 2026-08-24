"""OpenAI backend for Agent4LR.

Wraps the OpenAI Responses API. ``tool_choice`` maps directly to
OpenAI's native value (``"none"`` / ``"auto"`` / ``"required"``).

The runner now passes the canonical Responses-API-native ``history``
straight through as the ``input`` parameter — no client-side
translation. Reasoning items, function_call items, and
function_call_output items survive round-trips so encrypted reasoning
state is preserved across slot transitions.

Tool schemas accepted by this module are the **nested** Ollama shape
emitted by :mod:`src.agent4lr.tools` (``{"type": "function",
"function": {...}}``); the provider flattens them to the OpenAI
Responses API shape (``{"type": "function", "name": ..., "parameters":
...}``) at send time.

The ``openai`` SDK is imported lazily so the rest of the package
remains usable on workstations without it installed. Constructing
:class:`OpenAIProvider` on a system that lacks the dep raises a clear
``ImportError``.

Raises :class:`ContextOverflowError` when the OpenAI client reports a
context-window error.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from src.agent4lr.providers import (
    ContextOverflowError,
    NormalisedResponse,
    NormalisedToolCall,
    ToolChoice,
)

logger = logging.getLogger(__name__)

_CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"context.{0,20}(length|window|limit)|maximum context|token.{0,5}limit",
    re.IGNORECASE,
)


def _is_context_overflow(exc: Exception) -> bool:
    """Match a few documented OpenAI context-length error fingerprints."""
    msg = str(exc)
    if _CONTEXT_OVERFLOW_PATTERN.search(msg):
        return True
    code = getattr(exc, "code", None) or getattr(
        getattr(exc, "body", None), "get", lambda *_: None
    )("code")
    return code == "context_length_exceeded"


def _flatten_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map nested Ollama-style tool dicts to OpenAI Responses API shape."""
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", t)
        out.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return out


def _scrub_for_send(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop server-only fields the Responses API rejects on re-input.

    Reasoning items round-trip cleanly *except* for a ``status: null``
    leftover (matches FlexFL's ``process_response`` line 435-437). Other
    typed items (function_call / function_call_output) pass through
    unchanged.
    """
    out: list[dict[str, Any]] = []
    for item in history:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            sanitised = {k: v for k, v in item.items() if k != "status"}
            out.append(sanitised)
        else:
            out.append(item)
    return out


def build_openai_request_body(
    *,
    model: str,
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: ToolChoice = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build the ``responses.create`` kwargs / Batch API request body.

    Single source of truth for the request shape: the synchronous
    :meth:`OpenAIProvider.chat` path and the Batch API JSONL ``body``
    field both go through here, so scrubbing, tool flattening,
    ``parallel_tool_calls=False``, and the reasoning-vs-sampling branch
    can never drift between the two.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "input": _scrub_for_send(history),
        "parallel_tool_calls": False,
    }
    if reasoning_effort is not None:
        # gpt-5 / o-series reasoning models reject temperature/top_p.
        kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    if tools:
        kwargs["tools"] = _flatten_tools(tools)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return kwargs


def normalise_response_payload(raw: dict[str, Any]) -> NormalisedResponse:
    """Map a Responses API payload dict to :class:`NormalisedResponse`.

    Returns the response's ``output`` items (reasoning + message +
    function_call) as ``new_items`` so the runner can append them
    verbatim to the canonical ``input_list``. ``status`` is stripped
    from reasoning items — leaving ``status: null`` in a follow-up
    ``input`` field causes the API to reject the request.

    Consumes a plain dict so both the sync path (``response.to_dict()``)
    and the Batch API output lines (``line["response"]["body"]``) share
    one parser.
    """
    outputs = raw.get("output") or []
    text_chunks: list[str] = []
    tool_calls: list[NormalisedToolCall] = []
    new_items: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            sanitised = {k: v for k, v in item.items() if k != "status"}
            new_items.append(sanitised)
            continue
        new_items.append(item)
        if kind == "message":
            content = item.get("content") or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    text = first.get("text") or ""
                    if text:
                        text_chunks.append(text)
        elif kind == "function_call":
            name = item.get("name", "")
            args_raw = item.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                {
                    "name": name,
                    "arguments": args,
                    "call_id": item.get("call_id"),
                }
            )
    return {
        "content": "\n".join(text_chunks),
        "tool_calls": tool_calls,
        "new_items": new_items,
        "raw": raw,
    }


class OpenAIProvider:
    """Provider implementation for the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_effort: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        try:
            import openai  # noqa: F401 — lazy import; surfaces ImportError if missing
        except ImportError as exc:  # pragma: no cover — covered by user environment
            raise ImportError(
                "openai SDK is required for OpenAIProvider. "
                "Install via `pip install openai` in the cefl conda env."
            ) from exc

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAIProvider needs an API key. Pass --openai-api-key on the "
                "CLI or set the OPENAI_API_KEY env var."
            )
        from openai import OpenAI  # local import to keep top-level light

        self._client = OpenAI(api_key=key, timeout=timeout)

    @staticmethod
    def _normalise(response_obj: Any) -> NormalisedResponse:
        """Map a ``client.responses.create(...)`` payload to NormalisedResponse.

        Thin wrapper over :func:`normalise_response_payload` that first
        coerces the SDK response object to a plain dict.
        """
        raw = response_obj.to_dict() if hasattr(response_obj, "to_dict") else dict(response_obj)
        return normalise_response_payload(raw)

    def chat(
        self,
        *,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = None,
        **opts: Any,
    ) -> NormalisedResponse:
        """Call the Responses API and normalise the response.

        ``history`` is the canonical Responses-API-native input list,
        passed straight through as ``input``. Raises
        :class:`src.agent4lr.providers.ContextOverflowError` when the
        OpenAI client reports a context-window violation.
        """
        kwargs = build_openai_request_body(
            model=self.model,
            history=history,
            tools=tools,
            tool_choice=tool_choice,
            temperature=self.temperature,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
        )
        try:
            response = self._client.responses.create(**kwargs)
        except Exception as exc:
            if _is_context_overflow(exc):
                raise ContextOverflowError(f"OpenAI reported context overflow: {exc}") from exc
            raise
        return self._normalise(response)
