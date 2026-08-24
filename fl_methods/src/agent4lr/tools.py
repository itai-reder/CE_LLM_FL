"""Tool schemas and dispatcher for Agent4LR.

Three role-specific tool sets, each mapped to the appropriate
``tool_choice`` mode by the runner:

* **Planner** — no tools (``tool_choice="none"``).
* **Tool loop** — ``get_snippet_of_method(method_number: int)`` and
  ``exit()`` (``tool_choice="required"``).
* **Finisher** — ``rank_methods(top_5_methods: int[5])``
  (``tool_choice="required"``).

Schemas are JSON-Schema-shaped so they pass through both the OpenAI
Responses API and Ollama's ``/api/chat`` ``tools`` field unchanged. The
descriptions are copied from
``~/git/FlexFL/FlexFL/src/modular_utils.py`` (the behavioural reference)
with minimal rewording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LRToolContext:
    """Identifies the bug and pins the candidate list.

    ``candidates`` is the 1-based list rendered into the planner prompt
    (dotted FQNs from ``rankings/top20/<sr_model_id>.txt``);
    ``get_snippet_of_method`` resolves ``method_number`` against this
    list.

    ``dataset`` is threaded to the shared Agent4SR corpus loaders so a
    BugsInPy bug reads the **BIP** SR corpus (and ``load_corpus_codes``
    JSON-decodes the multi-line Python source) instead of defaulting to
    the Defects4J dir. Defaults to ``"defects4j"`` so D4J call sites are
    unchanged.
    """

    project: str
    bug_id: str
    candidates: list[str]
    dataset: str = "defects4j"


def tool_schemas_planner() -> list[dict[str, Any]] | None:
    """Return the planner's tool list.

    Planner has no tools — returns ``None`` so the runner can pass it
    directly as the provider's ``tools`` argument together with
    ``tool_choice="none"``.
    """
    return None


def tool_schemas_loop() -> list[dict[str, Any]]:
    """Return the tool-loop schemas: ``get_snippet_of_method`` + ``exit``."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_snippet_of_method",
                "description": (
                    "Return the code snippet for a suggested suspicious method "
                    "by its 1-based index."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method_number": {
                            "type": "integer",
                            "description": "1-based index into the suggested methods list",
                        },
                    },
                    "required": ["method_number"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exit",
                "description": ("Stop calling tools and proceed to rank the top-5 methods."),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def tool_schemas_finisher() -> list[dict[str, Any]]:
    """Return the finisher schemas: ``rank_methods`` only."""
    return [
        {
            "type": "function",
            "function": {
                "name": "rank_methods",
                "description": ("Rank the top-5 most likely culprit methods for the bug."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_5_methods": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 5,
                            "maxItems": 5,
                            "description": (
                                "A list of exactly 5 1-based indices of the most likely "
                                "culprit methods, ranked from most likely to least likely."
                            ),
                        },
                    },
                    "required": ["top_5_methods"],
                },
            },
        },
    ]


def _coerce_int(value: Any) -> int | None:
    """Best-effort coercion: accept int, str-int, or float-with-integer-value."""
    if isinstance(value, bool):  # bool is int subclass; reject
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def execute_tool(*, ctx: LRToolContext, name: str, args: dict[str, Any]) -> str:
    """Dispatch one tool call.

    Returns the tool output as a string (always succeeds; errors are
    formatted as a user-facing message the loop appends as an
    observation). The bad call still counts against the iteration
    budget — matching ``modular_pipeline.py``'s behaviour.
    """
    if name == "get_snippet_of_method":
        raw = args.get("method_number")
        idx = _coerce_int(raw)
        if idx is None:
            return (
                f"Invalid method_number {raw!r}. "
                f"It must be an integer between 1 and {len(ctx.candidates)}."
            )
        return get_snippet_of_method(ctx=ctx, method_number=idx)
    if name == "exit":
        return "Exiting..."
    if name == "rank_methods":
        ids = args.get("top_5_methods")
        if not isinstance(ids, list):
            return f"Invalid top_5_methods {ids!r}. Must be a list of exactly 5 integers."
        coerced: list[int] = []
        for v in ids:
            c = _coerce_int(v)
            if c is None:
                return f"Invalid index {v!r} in top_5_methods; integers required."
            coerced.append(c)
        resolved = rank_methods(ctx=ctx, top_5_methods=coerced)
        return "\n".join(f"Top_{i} : {fqn}" for i, fqn in enumerate(resolved, start=1))
    return f"Unknown tool {name!r}. Use the listed tools only."


def get_snippet_of_method(*, ctx: LRToolContext, method_number: int) -> str:
    """Resolve a 1-based candidate index to a source snippet.

    Format mirrors the reference pipeline's ``process_response``::

        The code snippet of <FQN> is as follows.
        ```
        <snippet>
        ```
    """
    if method_number < 1 or method_number > len(ctx.candidates):
        return f"Index {method_number} out of range. Must be between 1 and {len(ctx.candidates)}."
    fqn = ctx.candidates[method_number - 1]

    # Reuse Agent4SR's corpus loaders to avoid duplicating IO.
    from src.agent4sr import function_call as sr_fc

    methods = sr_fc.load_corpus_methods(ctx.project, ctx.bug_id, dataset=ctx.dataset)
    codes = sr_fc.load_corpus_codes(ctx.project, ctx.bug_id, dataset=ctx.dataset)

    for m, code in zip(methods, codes, strict=False):
        if m == fqn or m.replace("$", ".") == fqn:
            return f"The code snippet of {fqn} is as follows.\n```\n{code}\n```"

    logger.warning(
        "get_snippet_of_method: %r not found in corpus for %s-%s",
        fqn,
        ctx.project,
        ctx.bug_id,
    )
    return (
        f"Code snippet for {fqn} could not be located in the corpus. "
        f"You may pick a different index."
    )


def rank_methods(*, ctx: LRToolContext, top_5_methods: list[int]) -> list[str]:
    """Map a list of 1-based indices to candidate FQNs.

    Out-of-range indices are dropped with a logged warning; the result
    may be shorter than ``len(top_5_methods)``.
    """
    out: list[str] = []
    for idx in top_5_methods:
        if 1 <= idx <= len(ctx.candidates):
            out.append(ctx.candidates[idx - 1])
        else:
            logger.warning(
                "rank_methods: index %d out of range (candidates=%d)",
                idx,
                len(ctx.candidates),
            )
    return out
