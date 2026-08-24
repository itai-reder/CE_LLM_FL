"""Prompt templates for the Agent4SR LLM pipeline.

Each function returns a string suitable for an Ollama ``/api/chat`` message.
The prompts are direct migrations from the FlexFL originals.
"""

from __future__ import annotations

from src.agent4sr.io import BugInputs
from src.core.layout import normalize_benchmark_name


def sr_system_prompt(*, max_tool_calls: int, dataset: str = "defects4j") -> str:
    """System prompt describing the debugging assistant role and available tools.

    The language noun adapts to ``dataset``: Defects4J keeps the original "Java"
    wording **byte-identical** (the D4J results were tuned on it); BugsInPy says
    "Python".
    """
    lang = "Python" if normalize_benchmark_name(dataset) == "BIP" else "Java"
    functions = (
        "\nFunction calls you can use are as follows.\n"
        f"* get_paths() -> Get all path names of {lang} source files in the repository. *\n"
        "* get_classes_of_path(path_name) -> Get all classes under a given path. *\n"
        "* get_methods_of_class(class_name) -> Get all methods of a given class. *\n"
        f"* get_code_snippet_of_method(method_name) -> Get the code snippet of a {lang} "
        "method. *\n"
        "* find_class(class_name) -> Fuzzy search classes in the repository. *\n"
        "* find_method(method_name) -> Fuzzy search methods in the repository. *\n"
        "* exit() -> Exit function calling to give your final answer when you are "
        "confident of the answer. *\n"
    )
    return (
        f"You are a debugging assistant of our {lang} software. You will be presented with a "
        "bug report, a trigger test and tools (functions) to access the source code of the "
        "system under test (SUT). Your task is to locate the top-5 most likely culprit "
        "methods based on the bug report, the trigger test and the information you retrieve "
        f"using given functions. {functions}"
        f"You have {max_tool_calls} chances to call function."
    )


def sr_initial_user_prompt(inputs: BugInputs) -> str:
    """Initial user message with bug report + trigger test + planning instruction."""
    blocks: list[str] = []
    bug_report = inputs.bug_report_text_block()
    if bug_report:
        blocks.append(bug_report)
    blocks.append(inputs.trigger_test_text_block())
    blocks.append(
        "Let's locate the faulty method step by step using reasoning and function calls. "
        "Now reason and plan how to locate the buggy methods."
    )
    return "\n".join(blocks)


def sr_tool_call_user_prompt() -> str:
    """Prompt the LLM to make a tool call."""
    return (
        "Now call a function in this format `FunctionName(Argument)` "
        "in a single line without any other word."
    )


def sr_retry_user_prompt() -> str:
    """Nudge the LLM when it fails to produce a valid tool call."""
    return "Please call functions in the right format and choose from the available function list."


def sr_finisher_user_prompt(*, dataset: str = "defects4j") -> str:
    """Final prompt asking for the Top-5 culprit methods.

    The answer template matches the corpus id shape: Defects4J uses
    ``PathName.ClassName.MethodName(ArgType1, ArgType2)`` (parameter **types**),
    kept **byte-identical**; BugsInPy uses parameter **names** and spells out the
    module-level-function case (``module.function(arg1, arg2)`` with no class),
    per the canonical corpus-id shape.
    """
    if normalize_benchmark_name(dataset) == "BIP":
        template = "ModuleName.ClassName.method_name(arg1, arg2)"
        note = (
            "Use dotted module paths; parameters are argument names (not types). "
            "A module-level function has no class: ModuleName.function_name(arg1, arg2).\n"
        )
        return (
            "Based on the available information, provide complete name of the "
            "top-5 most likely culprit methods for the bug please. "
            "Since your answer will be processed automatically, please give your answer "
            "in the format as follows.\n"
            f"{note}"
            f"Top_1 : {template}\n"
            f"Top_2 : {template}\n"
            f"Top_3 : {template}\n"
            f"Top_4 : {template}\n"
            f"Top_5 : {template}\n"
        )
    return (
        "Based on the available information, provide complete name of the "
        "top-5 most likely culprit methods for the bug please. "
        "Since your answer will be processed automatically, please give your answer "
        "in the format as follows.\n"
        "Top_1 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
        "Top_2 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
        "Top_3 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
        "Top_4 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
        "Top_5 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
    )
