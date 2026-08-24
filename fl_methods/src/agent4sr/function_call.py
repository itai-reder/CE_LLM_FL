"""Corpus loading, fuzzy search, and the 7 search functions for Agent4SR.

These functions back the LLM tool-calling interface: the agent calls tools
like ``get_paths()`` or ``find_method(name)``, and the dispatcher in
:mod:`src.agent4sr.tools` routes to the functions here.

Corpus files are read from the bug's ``FlexFL/SR/`` dir (resolved by
:func:`src.common.config.get_sr_dir` for the given ``dataset``):
  - ``corpus_methods.txt`` — method IDs (one per line)
  - ``corpus_codes.txt`` — raw method code, one method per physical line, line-aligned
    with the IDs. D4J codes are single-line plain text; **BugsInPy codes are
    ``json.dumps``-encoded** (Python source is multi-line — significant indentation),
    so :func:`load_corpus_codes` ``json.loads`` them back to faithful multi-line code.

Corpus ids carry the canonical shape ``<pkg-or-module>$<qualname>(<params>)``. The
functions are language-agnostic with one Python-specific case: a **module-level
function** has qualname ``func`` (no dot), so its owner is the *module* itself.
The per-path class listing (:func:`get_classes`) therefore yields no
sub-class for it (the empty class is filtered); it is reached via
:func:`get_methods` / :func:`find_method` / :func:`get_code_snippet` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import Levenshtein

from src.common.config import get_sr_dir
from src.core.layout import normalize_benchmark_name

# ---------------------------------------------------------------------------
# Fuzzy search helpers
# ---------------------------------------------------------------------------


def split4search(query: str) -> list[str]:
    """Split a method signature into searchable tokens.

    Splits on ``.`` for the class/method part and extracts simplified
    parameter types from the parenthesised signature.

    Examples::

        >>> split4search("StringUtils.isEmpty")
        ['StringUtils', 'isEmpty']
        >>> split4search("StringUtils.isEmpty(CharSequence)")
        ['StringUtils', 'isEmpty', 'CharSequence']
    """
    if "(" not in query:
        return query.split(".")
    signature = query[query.find("(") + 1 : query.find(")")]
    method = query[: query.find("(")]
    return method.split(".") + [e.strip().split(".")[-1] for e in signature.split(",") if e.strip()]


def fuzzy_search(query: str, choices: list[str]) -> list[str]:
    """Find the best-matching method/class names from *choices*.

    Strategy:
      1. Try exact token containment (all tokens from *query* must appear
         as tokens in a candidate).
      2. If no containment match, simplify parameter FQCNs and use
         Levenshtein distance (threshold <= 5).
      3. If still nothing, return the 5 closest candidates.
    """
    query = query.replace("#", ".").replace("$", ".")
    match_res: list[str] = []
    querys = split4search(query)
    for choice in choices:
        match_choice = split4search(choice)
        if all(match_query in match_choice for match_query in querys):
            match_res.append(choice)
    if match_res:
        return match_res

    # Simplify FQ parameter types for distance comparison
    if "(" in query:
        signature = query[query.find("(") + 1 : query.find(")")]
        query = (
            query.split("(")[0]
            + "("
            + ",".join(e.strip().split(".")[-1] for e in signature.split(","))
            + ")"
        )

    distances = [(choice, Levenshtein.distance(query, choice)) for choice in choices]
    distances.sort(key=lambda x: x[1])
    for choice, dist in distances:
        if dist <= 5:
            match_res.append(choice)
        else:
            break
    if not match_res:
        match_res = [choice for choice, _ in distances[:5]]
    return match_res


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def corpus_methods_path(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> Path:
    """Return the path to ``corpus_methods.txt`` for a given bug."""
    return get_sr_dir(project, bug_id, dataset=dataset) / "corpus_methods.txt"


def corpus_codes_path(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> Path:
    """Return the path to ``corpus_codes.txt`` for a given bug."""
    return get_sr_dir(project, bug_id, dataset=dataset) / "corpus_codes.txt"


def load_corpus_methods(
    project: str, bug_id: str | int, *, dataset: str = "defects4j"
) -> list[str]:
    """Load method IDs from the corpus file."""
    path = corpus_methods_path(project, bug_id, dataset=dataset)
    return [e.strip() for e in path.read_text(encoding="utf-8").splitlines() if e.strip()]


def load_corpus_codes(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> list[str]:
    """Load raw method code from the corpus file (one entry per method, line-aligned
    with :func:`load_corpus_methods`).

    BugsInPy codes are ``json.dumps``-encoded per physical line (Python source is
    multi-line); decode them back to faithful multi-line code. D4J codes are single-line
    plain text, kept verbatim (stripped).
    """
    text = corpus_codes_path(project, bug_id, dataset=dataset).read_text(encoding="utf-8")
    if normalize_benchmark_name(dataset) == "BIP":
        return [json.loads(e) for e in text.splitlines() if e.strip()]
    return [e.strip() for e in text.splitlines() if e.strip()]


# ---------------------------------------------------------------------------
# Tool-backing functions (called by tools.execute_tool)
# ---------------------------------------------------------------------------


def get_code_snippet(
    project: str, bug_id: str | int, function: str, *, dataset: str = "defects4j"
) -> str:
    """Return the source code of *function*, or fuzzy suggestions if not found."""
    function = function.replace(", ", ",").replace(" ,", ",")
    methods = load_corpus_methods(project, bug_id, dataset=dataset)
    codes = load_corpus_codes(project, bug_id, dataset=dataset)
    for method, code in zip(methods, codes, strict=True):
        if method.replace("$", ".", 1) == function:
            return code
    methods2 = [m.replace("$", ".", 1) for m in methods]
    results = fuzzy_search(function, methods2)
    if len(results) == 1:
        method = results[0]
        code = get_code_snippet(project, bug_id, method, dataset=dataset)
        return f"Do you mean `{method}`? Its code snippet is as follows.\n{code}"
    if not results:
        return (
            "You provide a wrong method name. "
            "You can call `get_methods_of_class` first to get a right method name."
        )
    return "You provide a wrong method name. Please try the following method names.\n" + "\n".join(
        results
    )


def get_paths(project: str, bug_id: str | int, *, dataset: str = "defects4j") -> str:
    """Return all unique package paths from the corpus."""
    methods = load_corpus_methods(project, bug_id, dataset=dataset)
    paths = sorted(set(e.split("$")[0] for e in methods))
    return "\n".join(paths)


def get_classes(
    project: str, bug_id: str | int, path_name: str, *, dataset: str = "defects4j"
) -> str:
    """Return all classes under *path_name*, or fuzzy suggestions.

    A module-level function (qualname with no dot) yields an empty class string;
    those are filtered out so the agent never sees a blank class — its owner is the
    module itself, reached via ``get_methods_of_class`` on the path.
    """
    methods = load_corpus_methods(project, bug_id, dataset=dataset)
    paths = set(e.split("$")[0] for e in methods)
    classes = sorted(
        c
        for c in {
            ".".join(e.strip().split("$")[1].split("(")[0].split(".")[:-1])
            for e in methods
            if e.startswith(path_name)
        }
        if c  # drop the empty class of module-level functions
    )
    if classes:
        return "\n".join(classes)

    # A valid path whose only members are module-level functions has no classes.
    # Guide the agent to the methods directly (and avoid recursing on an exact match).
    if path_name in paths:
        return (
            f"`{path_name}` has no classes (only module-level functions). "
            "Call `get_methods_of_class` with this path to list them."
        )

    results = fuzzy_search(path_name, sorted(paths))
    if len(results) == 1:
        return f"Do you mean `{results[0]}`? Its classes are as follows.\n" + get_classes(
            project, bug_id, results[0], dataset=dataset
        )
    if results:
        return "You provide a wrong path name. Please try the following path names.\n" + "\n".join(
            sorted(results)
        )
    return "You provide a wrong path name. You can call `get_paths` first to get a right path name."


def get_methods(
    project: str, bug_id: str | int, class_name: str, *, dataset: str = "defects4j"
) -> str:
    """Return all methods of *class_name*, or fuzzy suggestions.

    The owner is derived as ``<module>.<DottedClass>`` for class methods and as
    ``<module>`` for module-level functions, so passing the module path as
    *class_name* lists that module's module-level functions.
    """
    method_list: list[str] = []
    for e in load_corpus_methods(project, bug_id, dataset=dataset):
        e2 = e.replace("$", ".", 1).strip()
        pos = e2.find("(")
        class_ = ".".join(e2[:pos].split(".")[:-1])
        if class_ == class_name:
            method_list.append(e2[len(class_) + 1 :])
    methods_set = sorted(set(method_list))
    if methods_set:
        return "\n".join(methods_set)

    all_classes = sorted(
        set(
            ".".join(e.strip().replace("$", ".", 1).split("(")[0].split(".")[:-1])
            for e in load_corpus_methods(project, bug_id, dataset=dataset)
        )
    )
    results = fuzzy_search(class_name, all_classes)
    if len(results) == 1:
        return f"Do you mean `{results[0]}`? Its methods are as follows.\n" + get_methods(
            project, bug_id, results[0], dataset=dataset
        )
    if results:
        return (
            "You provide a wrong class name. Please try the following class names.\n"
            + "\n".join(sorted(results))
        )
    return (
        "You provide a wrong class name. "
        "You can call `get_classes_of_path` first to get a right class name."
    )


def find_class(
    project: str, bug_id: str | int, class_name: str, *, dataset: str = "defects4j"
) -> str:
    """Fuzzy-search for *class_name* in the corpus."""
    classes = sorted(
        set(
            ".".join(e.strip().replace("$", ".", 1).split("(")[0].split(".")[:-1])
            for e in load_corpus_methods(project, bug_id, dataset=dataset)
        )
    )
    if "." in class_name:
        found = fuzzy_search(class_name, classes)
    else:
        found = [c for c in classes if c.split(".")[-1] == class_name]
        if not found:
            last_parts = sorted(set(c.split(".")[-1] for c in classes))
            cand = fuzzy_search(class_name, last_parts)
            if len(cand) == 1:
                return (
                    f"Do you mean `{cand[0]}`? Its result of fuzzy search is as follows.\n"
                    + find_class(project, bug_id, cand[0], dataset=dataset)
                )
            return (
                f"Do not find `{class_name}` again because it is an invalid name. "
                "You can try the following names.\n" + "\n".join(cand)
            )
    return "\n".join(sorted(found))


def find_method(
    project: str, bug_id: str | int, method_name: str, *, dataset: str = "defects4j"
) -> str:
    """Fuzzy-search for *method_name* in the corpus."""
    methods = [
        e.strip().replace("$", ".", 1)
        for e in load_corpus_methods(project, bug_id, dataset=dataset)
    ]
    results = fuzzy_search(method_name, sorted(set(methods)))
    return "\n".join(results)
