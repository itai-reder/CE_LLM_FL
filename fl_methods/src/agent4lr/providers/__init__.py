"""Provider abstraction for Agent4LR.

Defines the ``Provider`` Protocol that the runner talks to, and a normalised
response shape so the loop logic is identical regardless of backend.

Concrete providers in this package:

* :class:`src.agent4lr.providers.ollama.OllamaProvider` — POSTs to
  ``<base_url>/api/chat``. Emulates ``tool_choice="required"`` (Ollama has
  no native enforcement) via a system-prompt addendum + retry.
* :class:`src.agent4lr.providers.openai.OpenAIProvider` — Responses API.
  ``tool_choice`` maps to OpenAI's native value.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

ToolChoice = Literal["none", "auto", "required"] | None


class NormalisedToolCall(TypedDict):
    """One structured tool call extracted from a provider response."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None


class NormalisedResponse(TypedDict):
    """Provider-agnostic response shape consumed by the LR runner.

    ``new_items`` is the **list of OpenAI Responses-API native items**
    produced by this call (reasoning, message, function_call). The
    runner appends these directly onto the canonical ``input_list``;
    OpenAI calls hand them back on the next turn, preserving encrypted
    reasoning. Ollama synthesises equivalent items (no reasoning) so
    cross-provider chains stay coherent.
    """

    content: str
    tool_calls: list[NormalisedToolCall]
    new_items: list[dict[str, Any]]
    raw: dict[str, Any]


class Provider(Protocol):
    """One call into the underlying LLM backend.

    ``history`` is the canonical conversation state in OpenAI
    Responses-API native shape: role/content items plus typed items
    (``{"type": "reasoning", ...}``, ``{"type": "function_call", ...}``,
    ``{"type": "function_call_output", ...}``). The OpenAI provider
    feeds it straight back to ``responses.create``; the Ollama provider
    projects it to chat-completions internally (dropping reasoning,
    folding function_call/output pairs back onto assistant tool_calls +
    tool-role messages).

    Implementations must normalise the response into ``NormalisedResponse``
    so the runner can treat ``tool_calls``, ``content``, and the
    appendable ``new_items`` uniformly. Implementations should raise
    :class:`ContextOverflowError` on context-window overflow.
    """

    def chat(
        self,
        *,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = None,
        **opts: Any,
    ) -> NormalisedResponse:
        """Send one chat request and return a normalised response."""
        ...


class ContextOverflowError(Exception):
    """Raised by a provider when the request exceeds the model's context window."""


def build_provider(
    *,
    provider: Literal["ollama", "openai"],
    model: str,
    base_url: str = "",
    verify: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reasoning_effort: str | None = None,
    api_key: str | None = None,
    **opts: Any,
) -> Provider:
    """Factory: instantiate the requested provider with shared options.

    ``api_key`` is only consumed by the OpenAI provider; ``base_url`` /
    ``verify`` only by the Ollama provider. Both kwargs exist on the
    call so the LR runner can pass a uniform options dict.
    """
    if provider == "ollama":
        from src.agent4lr.providers.ollama import OllamaProvider

        base_url = base_url if base_url else "http://localhost:11434"
        return OllamaProvider(
            model=model, base_url=base_url, verify=verify, temperature=temperature, **opts
        )
    if provider == "openai":
        from src.agent4lr.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unknown provider {provider!r}")
