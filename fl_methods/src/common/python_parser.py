"""Python source code parsing for FL methods.

The ``javalang`` analogue for BugsInPy: method-level and statement-level
extraction from ``.py`` files using the stdlib :mod:`ast`.  Produces the **same**
:class:`~src.common.java_parser.MethodInfo` / :class:`~src.common.java_parser.StatementInfo`
shapes as :mod:`src.common.java_parser`, so BoostN (method-level) and Blues
(statement-level) reuse them unchanged.

Python entity conventions:

- **Module path** = file path relative to the source root, ``/`` → ``.``, drop
  ``.py``; ``__init__.py`` collapses to its package path.
  (``youtube_dl/utils.py`` → ``youtube_dl.utils``; ``pkg/__init__.py`` → ``pkg``.)
- **Owner (``class_fqn``, the statement owner)** = ``<module>.<DottedClass>`` for a
  method inside a class (nested → ``<module>.Outer.Inner``); ``<module>`` for a
  module-level function (the module *is* the owner).
- **Corpus / method ID** = ``<module>$<qualname>(<param_names>)`` — ``$`` splits the
  module dotted-path from the in-module qualname.  Built by
  :func:`src.common.method_entity.method_entity_from_python_method_info`
  (``MethodInfo.method_id`` itself stays in the Java-parallel dotted form so
  ``MethodInfo`` is benchmark-agnostic).
- **Statement ID** = ``<owner>#<line>`` (class-bearing for lines inside a class).

Param **names** (not types) are recorded: Python type hints are frequently absent
and there is no overloading, so ``(module, qualname)`` is already unique.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from src.common.java_parser import MethodInfo, StatementInfo

# ---------------------------------------------------------------------------
# Module-path derivation
# ---------------------------------------------------------------------------


def module_path_for_python(path: str | Path, source_root: str | Path | None) -> str:
    """Derive the dotted module path of *path* relative to *source_root*.

    ``<root>/youtube_dl/utils.py`` → ``youtube_dl.utils``;
    ``<root>/pkg/sub/__init__.py`` → ``pkg.sub``.  Falls back to the file stem
    when *source_root* is absent or *path* is outside it.
    """
    path = Path(path)
    if source_root is None:
        return path.stem
    try:
        rel = path.resolve().relative_to(Path(source_root).resolve())
    except (ValueError, OSError):
        return path.stem
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _render_params(args: ast.arguments) -> list[str]:
    """Render an ``ast.arguments`` node to ordered parameter names (no spaces).

    Includes ``self``/``cls``; renders varargs as ``*args`` and ``**kwargs``.
    """
    names: list[str] = []
    for a in getattr(args, "posonlyargs", []):
        names.append(a.arg)
    for a in args.args:
        names.append(a.arg)
    if args.vararg is not None:
        names.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        names.append(a.arg)
    if args.kwarg is not None:
        names.append(f"**{args.kwarg.arg}")
    return names


def _func_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    """Return 1-based inclusive ``(start_line, end_line)``, expanding over decorators."""
    start = node.lineno
    if node.decorator_list:
        start = min(start, min(d.lineno for d in node.decorator_list))
    end = node.end_lineno or node.lineno
    return start, end


# ---------------------------------------------------------------------------
# Method extraction (for BoostN / method_signatures.csv)
# ---------------------------------------------------------------------------


def extract_methods_from_python(
    py_file_path: str | Path,
    *,
    module: str | None = None,
    source_root: str | Path | None = None,
) -> list[MethodInfo]:
    """Extract module-level functions and class methods from a Python file.

    Skips functions nested inside other functions (closures/locals) — they are
    not independently addressable corpus entities.  ``MethodInfo.method_id`` is
    the Java-parallel dotted form ``<module>.<qualname>(<params>).<start>.<end>``;
    ``class_fqn`` is the owner (``<module>`` or ``<module>.<DottedClass>``);
    ``param_types`` holds parameter *names*.
    """
    py_file_path = Path(py_file_path)
    try:
        source = py_file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    if module is None:
        module = module_path_for_python(py_file_path, source_root)
    lines = source.splitlines()
    methods: list[MethodInfo] = []

    def build(node: ast.FunctionDef | ast.AsyncFunctionDef, class_stack: list[str]) -> MethodInfo:
        name = node.name
        param_names = _render_params(node.args)
        start, end = _func_span(node)
        if class_stack:
            owner = f"{module}." + ".".join(class_stack)
            qualname = ".".join([*class_stack, name])
        else:
            owner = module or py_file_path.stem
            qualname = name
        method_id = f"{module}.{qualname}({','.join(param_names)}).{start}.{end}"
        content = " ".join(line.strip() for line in lines[start - 1 : end])
        return MethodInfo(
            method_id=method_id,
            class_fqn=owner,
            method_name=name,
            param_types=param_names,
            start_line=start,
            end_line=end,
            content=content,
            file_path=str(py_file_path),
        )

    def visit(node: ast.AST, class_stack: list[str], in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*class_stack, child.name], in_function)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not in_function:
                    methods.append(build(child, class_stack))
                visit(child, class_stack, True)
            else:
                visit(child, class_stack, in_function)

    visit(tree, [], False)
    return methods


# ---------------------------------------------------------------------------
# Statement extraction (for Blues — class-bearing owner IDs)
# ---------------------------------------------------------------------------


def _class_ranges(tree: ast.AST) -> list[tuple[int, int, str]]:
    """Collect ``(start_line, end_line, dotted_class)`` for every class definition."""
    ranges: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                dotted = ".".join([*stack, child.name])
                ranges.append((child.lineno, child.end_lineno or child.lineno, dotted))
                walk(child, [*stack, child.name])
            else:
                walk(child, stack)

    walk(tree, [])
    return ranges


def _owner_for_line(line: int, module: str, class_ranges: list[tuple[int, int, str]]) -> str:
    """Return the class-bearing owner FQN for *line* (innermost class, else module)."""
    best: tuple[int, str] | None = None  # (span, dotted) — smallest span wins
    for start, end, dotted in class_ranges:
        if start <= line <= end:
            span = end - start
            if best is None or span < best[0]:
                best = (span, dotted)
    if best is None:
        return module
    return f"{module}.{best[1]}"


def extract_statements_from_python(
    py_file_path: str | Path,
    source_root: str | Path | None = None,
    *,
    module: str | None = None,
) -> list[StatementInfo]:
    """Extract statement-level entries (Blues parity) with class-bearing owner IDs.

    Each non-blank, non-``#``-comment source line becomes a statement with
    ``stmt_id = <owner>#<line>`` where owner is the innermost enclosing class
    (``<module>.<DottedClass>``) or the module for module-level lines.  This is a
    practical line-based approximation (multiline strings are not specially
    handled), mirroring the Java statement extractor.
    """
    py_file_path = Path(py_file_path)
    try:
        source = py_file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    if module is None:
        module = module_path_for_python(py_file_path, source_root)
    class_ranges = _class_ranges(tree)
    lines = source.splitlines()
    statements: list[StatementInfo] = []

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        owner = _owner_for_line(i, module, class_ranges)
        statements.append(
            StatementInfo(
                stmt_id=f"{owner}#{i}",
                content=stripped,
                line_number=i,
                class_fqn=owner,
                file_path=str(py_file_path),
            )
        )
    return statements


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_python_files(
    root_dir: str | Path,
    exclude_tests: bool = True,
) -> list[Path]:
    """Recursively find ``.py`` source files under *root_dir*.

    Mirrors :func:`src.common.java_parser.find_java_files`: when *exclude_tests*
    is set, skips directories whose path contains ``test`` (case-insensitive) and
    pytest/unittest file names (``test_*.py``, ``*_test.py``, ``conftest.py``).
    Always skips ``__pycache__`` and dot-directories.
    """
    root_dir = Path(root_dir)
    py_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not d.startswith(".")]
        rel = Path(dirpath).relative_to(root_dir)
        if exclude_tests and any("test" in p.lower() for p in rel.parts):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if exclude_tests and (
                fname.startswith("test_") or fname.endswith("_test.py") or fname == "conftest.py"
            ):
                continue
            py_files.append(Path(dirpath) / fname)
    return sorted(py_files)
