"""Tool schemas and dispatcher for Agent4SR.

Defines the Ollama tool-calling JSON schemas and routes tool calls
to the backing functions in :mod:`src.agent4sr.function_call`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.agent4sr import function_call
from src.core.layout import normalize_benchmark_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Identifies the bug whose corpus the tools operate on.

    ``dataset`` selects which benchmark's ``FlexFL/SR/`` corpus the tools read
    (and, for BugsInPy, the JSON-decoding of ``corpus_codes.txt``). Defaults to
    Defects4J so existing D4J call sites are unchanged.
    """

    project: str
    bug_id: str
    dataset: str = "defects4j"


def tool_schemas(dataset: str = "defects4j") -> list[dict[str, Any]]:
    """Return the Ollama tool-calling schemas for all 7 Agent4SR tools.

    The descriptions feed the LLM, so the language noun adapts to ``dataset``:
    Defects4J keeps the original "Java" wording byte-identical; BugsInPy says
    "Python".
    """
    lang = "Python" if normalize_benchmark_name(dataset) == "BIP" else "Java"
    return [
        {
            "type": "function",
            "function": {
                "name": "get_paths",
                "description": f"Get all path names of {lang} source files in the repository.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_classes_of_path",
                "description": "Get all classes under a given path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path_name": {"type": "string"}},
                    "required": ["path_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_methods_of_class",
                "description": "Get all methods of a given class.",
                "parameters": {
                    "type": "object",
                    "properties": {"class_name": {"type": "string"}},
                    "required": ["class_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_code_snippet_of_method",
                "description": f"Get the code snippet of a {lang} method.",
                "parameters": {
                    "type": "object",
                    "properties": {"method_name": {"type": "string"}},
                    "required": ["method_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_class",
                "description": "Fuzzy search classes in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {"class_name": {"type": "string"}},
                    "required": ["class_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_method",
                "description": "Fuzzy search methods in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {"method_name": {"type": "string"}},
                    "required": ["method_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exit",
                "description": "Exit function calling.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def _get_str_arg(args: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Extract a string argument from *args*, trying multiple key names."""
    for key in keys:
        if key not in args:
            continue
        val = args.get(key)
        if isinstance(val, str):
            s = val.strip()
            if s:
                return s
        if isinstance(val, (int, float, bool)):
            return str(val)
    raise ValueError(f"Expected one of string arguments: {', '.join(keys)}")


def execute_tool(*, ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call to the appropriate function_call function.

    Returns the tool output as a string (always succeeds — errors are
    formatted as user-friendly messages).
    """
    try:
        if name == "get_paths":
            return function_call.get_paths(ctx.project, ctx.bug_id, dataset=ctx.dataset)
        if name == "get_classes_of_path":
            return function_call.get_classes(
                ctx.project,
                ctx.bug_id,
                _get_str_arg(args, ("path_name", "path", "name")),
                dataset=ctx.dataset,
            )
        if name == "get_methods_of_class":
            return function_call.get_methods(
                ctx.project,
                ctx.bug_id,
                _get_str_arg(args, ("class_name", "class", "name")),
                dataset=ctx.dataset,
            )
        if name == "get_code_snippet_of_method":
            return function_call.get_code_snippet(
                ctx.project,
                ctx.bug_id,
                _get_str_arg(args, ("method_name", "method", "function", "name")),
                dataset=ctx.dataset,
            )
        if name == "find_class":
            return function_call.find_class(
                ctx.project,
                ctx.bug_id,
                _get_str_arg(args, ("class_name", "class", "name")),
                dataset=ctx.dataset,
            )
        if name == "find_method":
            return function_call.find_method(
                ctx.project,
                ctx.bug_id,
                _get_str_arg(args, ("method_name", "method", "function", "name")),
                dataset=ctx.dataset,
            )
        if name == "exit":
            return "exit"
        return f"Unknown tool: {name}. Use one of the supported tools."
    except Exception as exc:
        return (
            f"Tool argument error for {name}: {exc}. "
            f"Please retry with correct arguments. Args={json.dumps(args)}"
        )


def normalize_method_name(*, ctx: ToolContext, method_name: str) -> str:
    """Attempt to normalise *method_name* against the corpus.

    Returns the canonical corpus identity (``pkg$Class.method(Params)`` with
    the ``$`` package/class separator) when a match is found.  The LLM often
    emits the dotted form (``pkg.Class.method(Params)``) since that's what
    ``get_code_snippet`` accepts; we restore the canonical separator so
    downstream ranking joins succeed.

    Falls back to the existing fuzzy-suggestion path for misses.
    """
    method_name = method_name.replace(", ", ",").replace(" ,", ",").strip()

    methods = function_call.load_corpus_methods(ctx.project, ctx.bug_id, dataset=ctx.dataset)
    for m in methods:
        if m == method_name or m.replace("$", ".") == method_name:
            return m

    snippet = function_call.get_code_snippet(
        ctx.project, ctx.bug_id, method_name, dataset=ctx.dataset
    )
    if snippet.startswith("Do you mean `"):
        parts = snippet.split("`")
        if len(parts) >= 2:
            return parts[1]
        return method_name
    if snippet.startswith("You provide a wrong method name") and "\n" in snippet:
        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[1]
    return method_name
