"""Ollama backend for Agent4LR.

Wraps Ollama's ``/api/chat`` endpoint. Ollama's HTTP API accepts a
``tools`` field but, as of writing, does not honor an explicit
``tool_choice="required"`` constraint. We emulate enforcement at the
provider boundary:

1. When ``tool_choice == "required"``, prepend a one-line addendum to
   the last user message asking the model to call exactly one tool.
2. On a response with no ``tool_calls``, retry once with the addendum
   reinforced; if the second attempt also produces no tool call, return
   the empty-``tool_calls`` response — the runner moves on without
   surfacing a separate retry-user-message into the transcript.

The runner now hands us the canonical Responses-API-native ``history``
(reasoning items + ``function_call`` / ``function_call_output``
items). We project it to Ollama's chat-completions shape internally:
reasoning items are dropped, function_call items fold back onto
synthetic assistant ``tool_calls``, and function_call_output items
become ``role: "tool"`` messages.

The provider raises :class:`ContextOverflowError` when Ollama's
response indicates the conversation exceeded the model's context
window. Detection is best-effort: Ollama's error vocabulary varies
across versions, so the matcher only catches the obvious cases — other
overflow modes propagate as ``requests.HTTPError`` and surface to the
runner (which does not catch them in v1).
"""

from __future__ import annotations

import json as _json
import logging
import uuid
import warnings
from copy import deepcopy
from typing import Any

import requests
from urllib3.exceptions import InsecureRequestWarning

from src.agent4lr.providers import (
    ContextOverflowError,
    NormalisedResponse,
    NormalisedToolCall,
    ToolChoice,
)

logger = logging.getLogger(__name__)

_REQUIRED_ADDENDUM = (
    "\n\n(IMPORTANT: You must call exactly one of the available tools "
    "in this response. Do not reply with free text.)"
)
_BASE_URL = "http://localhost:11434"
_RECOGNIZED_PAYLOAD_KEYS = ["think"]


def _context_overflow_signal(text: str) -> bool:
    """Conservative substring match for Ollama context-overflow errors."""
    haystack = text.lower()
    return any(
        token in haystack
        for token in (
            "context length",
            "context window",
            "ctx length",
            "too long",
            "max_tokens exceeded",
            "exceeds the maximum",
        )
    )


def _get_ollama_models(
    base_url: str = _BASE_URL, verify: bool = True, timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Query Ollama for the list of available models. (JSON response from `/api/tags` endpoint)"""
    endpoint = f"{base_url}/api/tags"
    if verify:
        response = requests.get(endpoint, timeout=timeout, verify=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            response = requests.get(endpoint, timeout=timeout, verify=False)
    response.raise_for_status()
    models: list[dict[str, Any]] = response.json().get("models", [])
    return models


def _list_ollama_model_names(
    base_url: str = _BASE_URL, verify: bool = True, timeout: float = 10.0
) -> list[str]:
    """Utility function, returns a list of available Ollama model names."""
    models = _get_ollama_models(base_url=base_url, verify=verify, timeout=timeout)
    return [model["model"] for model in models if "model" in model]


def _model_is_available(
    model: str, base_url: str = _BASE_URL, verify: bool = True, timeout: float = 10.0
) -> bool:
    """Check if the specified model is available in Ollama."""
    available_models = _list_ollama_model_names(base_url=base_url, verify=verify, timeout=timeout)
    return model in available_models


def _fallback_to_lowercase(
    model, base_url: str = _BASE_URL, verify: bool = True, timeout: float = 10.0
) -> str:
    """If the specified model is not found, attempt a case-insensitive match."""
    available_models = _list_ollama_model_names(base_url=base_url, verify=verify, timeout=timeout)
    l_model_map = {m.lower(): m for m in available_models}
    l_model = model.lower()
    if l_model in l_model_map:
        matched_model = l_model_map[l_model]
        return matched_model
    raise ValueError(
        f"Model '{model}' not found in Ollama and no case-insensitive match found among available models: {available_models}"
    )


class OllamaProvider:
    """Provider implementation for Ollama (``/api/chat``)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = _BASE_URL,
        verify: bool = True,
        temperature: float = 0.0,
        timeout: float = 600.0,
        **opts: Any,
    ) -> None:
        """Store config and probe ``/api/tags`` for a case-insensitive model match.

        Network failures during the probe are tolerated: a warning is logged and
        the model name is used as-given. Real misconfigurations surface at the
        first ``chat()`` call.
        """
        if not base_url:
            raise ValueError("OllamaProvider requires a non-empty base_url")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.verify = verify
        self.temperature = temperature
        self.timeout = timeout
        self._default_opts: dict[str, Any] = dict(opts)

        self.name = model
        self._validate_model_name()

    def _validate_model_name(self) -> bool:
        """Probe Ollama for the configured model; case-insensitive-fallback if needed.

        Returns ``True`` if a fallback was applied. Tolerates network errors so
        the provider stays constructible in offline tests / cold environments.
        """
        try:
            available = _model_is_available(
                self.model,
                base_url=self.base_url,
                verify=self.verify,
                timeout=self.timeout,
            )
        except (requests.ConnectionError, requests.HTTPError, requests.Timeout) as exc:
            logger.warning(
                "Ollama /api/tags probe failed (%s); proceeding with model=%r as-given.",
                exc,
                self.model,
            )
            return False
        if available:
            return False
        logger.warning(
            "Model '%s' not found in Ollama. Attempting case-insensitive match...",
            self.model,
        )
        self.model = _fallback_to_lowercase(
            self.model,
            base_url=self.base_url,
            verify=self.verify,
            timeout=self.timeout,
        )
        logger.info("Using model '%s' after case-insensitive match.", self.model)
        return True

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        if self.verify:
            resp = requests.post(url, json=payload, timeout=self.timeout, verify=True)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InsecureRequestWarning)
                resp = requests.post(url, json=payload, timeout=self.timeout, verify=False)
        if resp.status_code >= 400:
            body = resp.text or ""
            if _context_overflow_signal(body):
                raise ContextOverflowError(
                    f"Ollama reported context overflow (status {resp.status_code}): {body[:200]}"
                )
            resp.raise_for_status()
        parsed: dict[str, Any] = resp.json()
        return parsed

    def _normalise(self, raw: dict[str, Any]) -> NormalisedResponse:
        """Map Ollama's response to a NormalisedResponse + Responses-native new_items.

        Builds ``new_items`` so the runner can append them onto its
        canonical ``input_list`` without caring which provider produced
        them: an assistant message item when there's content, plus one
        ``function_call`` item per tool call (with a synthetic call_id
        since Ollama doesn't always emit one and the OpenAI Responses
        API requires the id to be unique-and-present when chained later).
        """
        msg = raw.get("message", {}) or {}
        content = msg.get("content") or ""
        tool_calls_raw = msg.get("tool_calls") or []
        tcs: list[NormalisedToolCall] = []
        new_items: list[dict[str, Any]] = []
        if content:
            new_items.append({"role": "assistant", "content": content})
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments")
            if isinstance(args, str):
                # Ollama sometimes returns arguments as a JSON-encoded string
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            call_id = tc.get("id") or f"call_{name}_{uuid.uuid4().hex[:12]}"
            tcs.append({"name": name, "arguments": args, "call_id": call_id})
            new_items.append(
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": _json.dumps(args, separators=(",", ":")),
                    "call_id": call_id,
                }
            )
        fallback_model = raw.get("model")
        if fallback_model and self.name != fallback_model:
            # Preserve the original model/tag info for debugging and restore the
            # caller-facing name (which may differ in case from what Ollama echoed).
            raw["fallback_model"] = fallback_model
            raw["model"] = self.name
        return {"content": content, "tool_calls": tcs, "new_items": new_items, "raw": raw}

    @staticmethod
    def _with_required_addendum(
        messages: list[dict[str, Any]], reinforce: bool = False
    ) -> list[dict[str, Any]]:
        """Append the ``tool_choice="required"`` addendum to the last user message.

        If the last message is not a user message, inject a fresh user
        message with the addendum. ``reinforce=True`` doubles it.
        """
        addendum = _REQUIRED_ADDENDUM if not reinforce else (_REQUIRED_ADDENDUM * 2)
        out = deepcopy(messages)
        if out and out[-1].get("role") == "user":
            content = out[-1].get("content") or ""
            if isinstance(content, str):
                out[-1] = {**out[-1], "content": content + addendum}
                return out
        out.append({"role": "user", "content": addendum.strip()})
        return out

    @staticmethod
    def _project_history_to_messages(
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Translate a Responses-API ``input_list`` to Ollama chat-completions.

        Rules (walking in order; positional only):

        * ``{"type": "reasoning", ...}`` items are dropped entirely —
          Ollama doesn't consume them and they'd waste tokens.
        * ``{"type": "function_call", ...}`` items become a synthetic
          assistant message: ``{"role": "assistant", "content": "",
          "tool_calls": [{"function": {"name", "arguments"}, "id"}]}``.
          Consecutive function_call items coalesce onto a single
          assistant message so Ollama sees one assistant turn per
          provider turn.
        * ``{"type": "function_call_output", ...}`` items become
          ``{"role": "tool", "tool_name": <resolved>, "content":
          <output>}``. The tool name is resolved by remembering each
          function_call's ``call_id → name`` mapping in order.
        * Role/content items pass through with their fields preserved.
        """
        out: list[dict[str, Any]] = []
        call_id_to_name: dict[str, str] = {}
        for item in history:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "reasoning":
                continue
            if kind == "function_call":
                name = item.get("name", "")
                args_raw = item.get("arguments", "{}")
                args = args_raw
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                call_id = item.get("call_id") or f"call_{name}_{uuid.uuid4().hex[:12]}"
                call_id_to_name[call_id] = name
                tc_entry = {
                    "function": {"name": name, "arguments": args},
                    "id": call_id,
                }
                if (
                    out
                    and out[-1].get("role") == "assistant"
                    and isinstance(out[-1].get("tool_calls"), list)
                ):
                    out[-1]["tool_calls"].append(tc_entry)
                else:
                    out.append({"role": "assistant", "content": "", "tool_calls": [tc_entry]})
                continue
            if kind == "function_call_output":
                call_id = item.get("call_id", "")
                tool_name = call_id_to_name.get(call_id, "")
                out.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(item.get("output", "")),
                    }
                )
                continue
            role = item.get("role")
            if role in {"system", "user", "assistant"}:
                msg: dict[str, Any] = {"role": role, "content": item.get("content", "")}
                # Pass through any tool_calls if a caller pre-baked the
                # chat-completions shape — useful for tests.
                if "tool_calls" in item:
                    msg["tool_calls"] = item["tool_calls"]
                out.append(msg)
        return out

    def chat(
        self,
        *,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = None,
        **opts: Any,
    ) -> NormalisedResponse:
        """POST to ``<base_url>/api/chat`` and normalise the response.

        ``history`` is in Responses-API-native shape; we project to
        chat-completions internally. On
        ``tool_choice="required"`` with an empty initial response,
        retries once with a reinforced addendum before returning. On
        context-window overflow, raises
        :class:`src.agent4lr.providers.ContextOverflowError`.
        """
        chat_messages = self._project_history_to_messages(history)
        payload_messages = chat_messages
        if tool_choice == "required":
            payload_messages = self._with_required_addendum(chat_messages)
        merged_opts = {**self._default_opts, **opts}
        temperature = merged_opts.get("temperature", self.temperature)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        for opt in _RECOGNIZED_PAYLOAD_KEYS:
            if opt in merged_opts:
                payload[opt] = merged_opts[opt]

        if tools:
            payload["tools"] = tools
        raw = self._post(payload)
        norm = self._normalise(raw)

        if tool_choice == "required" and not norm["tool_calls"]:
            logger.debug("Ollama: empty tool_calls under tool_choice='required'; retrying once")
            payload["messages"] = self._with_required_addendum(chat_messages, reinforce=True)
            raw2 = self._post(payload)
            norm = self._normalise(raw2)

        return norm
