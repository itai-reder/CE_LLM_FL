"""
Java source code parsing for FL methods.

Provides method-level and statement-level extraction from .java files
using the ``javalang`` library.  Shared by BoostN (method-level) and
Blues (statement-level).

Reference implementations:
  - ParserCorpusMethodLevelGranularity.java  (Eclipse JDT AST)
  - Blues statement extraction (57 AST node types)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import javalang  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MethodInfo:
    """A single extracted Java method."""

    method_id: str  # e.g. "pkg.Class.method(Param1,Param2).94.105"
    class_fqn: str  # e.g. "org.jfree.data.xy.XYSeries"
    method_name: str  # e.g. "addOrUpdate"
    param_types: list[str]  # e.g. ["Number", "Number"]
    start_line: int
    end_line: int
    content: str  # raw method source (newlines replaced with spaces)
    file_path: str = ""  # absolute path to .java file


@dataclass
class StatementInfo:
    """A single source statement (line) for Blues IR indexing."""

    stmt_id: str  # e.g. "org.jfree.data.xy.XYSeries#542"
    content: str  # the source line text
    line_number: int
    class_fqn: str  # fully qualified class name
    file_path: str = ""


def _type_to_string(type_node: object) -> str:
    """Render a javalang type node to a BoostN/FlexFL-like parameter string."""
    name = getattr(type_node, "name", None)
    if not isinstance(name, str) or not name:
        return "Object"

    rendered = name

    arguments = getattr(type_node, "arguments", None)
    if arguments:
        rendered_args = []
        for argument in arguments:
            pattern_type = getattr(argument, "pattern_type", None)
            arg_type = getattr(argument, "type", None)
            if pattern_type == "?" and arg_type is None:
                rendered_args.append("?")
            elif pattern_type in {"extends", "super"} and arg_type is not None:
                rendered_args.append(f"? {pattern_type} {_type_to_string(arg_type)}")
            elif arg_type is not None:
                rendered_args.append(_type_to_string(arg_type))
            else:
                rendered_args.append("?")
        rendered += f"<{','.join(rendered_args)}>"

    sub_type = getattr(type_node, "sub_type", None)
    if sub_type is not None:
        rendered += f".{_type_to_string(sub_type)}"

    dimensions = getattr(type_node, "dimensions", None)
    if dimensions:
        rendered += "[]" * len(dimensions)

    return rendered


# ---------------------------------------------------------------------------
# Method extraction (for BoostN)
# ---------------------------------------------------------------------------


def extract_methods_from_java(
    java_file_path: str | Path,
) -> list[MethodInfo]:
    """Extract all non-abstract method bodies from a Java source file.

    Uses ``javalang`` to parse the AST, then brace-matching to find method
    end-lines (since javalang only gives start positions).

    Returns a list of MethodInfo with method_id in the format:
        package.Class.method(ParamType1,ParamType2).startLine.endLine
    matching the BoostN/FlexFL convention.
    """
    java_file_path = Path(java_file_path)
    try:
        content = java_file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    try:
        tree = javalang.parse.parse(content)
    except Exception:
        return []

    package_name = tree.package.name if tree.package else ""
    lines = content.splitlines()
    methods: list[MethodInfo] = []

    def append_executable(path_nodes: tuple[Any, ...], node: Any) -> None:
        if not hasattr(node, "body") or node.body is None:
            return

        if isinstance(node, javalang.tree.MethodDeclaration) and (
            node.modifiers and "abstract" in node.modifiers
        ):
            return

        # Walk the path to find the enclosing class(es)
        class_names = []
        for p in path_nodes:
            if isinstance(
                p,
                (
                    javalang.tree.ClassDeclaration,
                    javalang.tree.InterfaceDeclaration,
                    javalang.tree.EnumDeclaration,
                ),
            ):
                class_names.append(p.name)
        class_name = ".".join(class_names) if class_names else "Unknown"

        # Build FQN
        fqn_prefix = f"{package_name}.{class_name}" if package_name else class_name

        # Parameter types
        param_types = []
        for param in node.parameters or []:
            try:
                ptype = _type_to_string(param.type)
            except AttributeError:
                ptype = "Object"
            param_types.append(ptype)

        method_name = node.name
        declaration_line = node.position.line if node.position else -1
        if declaration_line == -1:
            return

        start_line = _expand_method_start_line(lines, declaration_line)

        # Find end line via brace matching
        end_line = _find_method_end(lines, declaration_line - 1)

        # Extract method content (join lines, collapse whitespace)
        method_lines = lines[declaration_line - 1 : end_line]
        method_content = " ".join(line.strip() for line in method_lines)

        # Method ID: package.Class.method(P1,P2).startLine.endLine
        sig = f"{fqn_prefix}.{method_name}({','.join(param_types)})"
        method_id = f"{sig}.{start_line}.{end_line}"

        methods.append(
            MethodInfo(
                method_id=method_id,
                class_fqn=fqn_prefix,
                method_name=method_name,
                param_types=param_types,
                start_line=start_line,
                end_line=end_line,
                content=method_content,
                file_path=str(java_file_path),
            )
        )

    for path_nodes, node in tree.filter(javalang.tree.MethodDeclaration):
        append_executable(path_nodes, node)

    for path_nodes, node in tree.filter(javalang.tree.ConstructorDeclaration):
        append_executable(path_nodes, node)

    return methods


def _expand_method_start_line(lines: list[str], declaration_line: int) -> int:
    """Expand method start to include leading doc comments/annotations.

    FlexFL method boundaries appear to include immediately preceding Javadoc
    blocks, while extraction content should still begin at the declaration line.
    """
    idx = declaration_line - 1  # 0-based

    while idx > 0:
        prev = idx - 1
        stripped = lines[prev].strip()

        # Include contiguous annotations.
        if stripped.startswith("@"):
            idx = prev
            continue

        # Include contiguous block/javadoc comments directly above declaration.
        if stripped.endswith("*/") or stripped.startswith("/*") or stripped.startswith("*"):
            block_start = prev
            while block_start >= 0:
                if "/*" in lines[block_start]:
                    idx = block_start
                    break
                block_start -= 1
            if block_start >= 0:
                continue

        break

    return idx + 1  # back to 1-based


def _find_method_end(lines: list[str], start_idx: int) -> int:
    """Find the closing brace of a method starting at ``start_idx`` (0-based).

    Returns 1-based line number of the closing brace.
    """
    brace_count = 0
    found_open = False
    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Count braces outside of string literals (rough heuristic)
        for ch in line:
            if ch == "{":
                brace_count += 1
                found_open = True
            elif ch == "}":
                brace_count -= 1
        if found_open and brace_count == 0:
            return i + 1  # 1-based
    return len(lines)  # fallback: rest of file


# ---------------------------------------------------------------------------
# Statement extraction (for Blues)
# ---------------------------------------------------------------------------


def extract_statements_from_java(
    java_file_path: str | Path,
    source_root: str | Path | None = None,
) -> list[StatementInfo]:
    """Extract statement-level entries from a Java source file.

    Each non-blank, non-comment source line is treated as a "statement".
    This is a practical approximation of Blues' 57-AST-node-type extraction,
    since we don't have Indri/Eclipse JDT available in Python.

    Statement IDs use the GZoltar convention:
        fully.qualified.ClassName#lineNumber

    Args:
        java_file_path: Path to a .java file.
        source_root: If given, used to derive the package-qualified class name
                     from the file path. Otherwise, parsed from the source.
    """
    java_file_path = Path(java_file_path)
    try:
        content = java_file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    lines = content.splitlines()
    class_fqn = _derive_class_fqn(java_file_path, lines, source_root)

    statements: list[StatementInfo] = []
    in_block_comment = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track block comments
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue

        # Skip empty, single-line comments, annotations, pure braces
        if (
            not stripped
            or stripped.startswith("//")
            or stripped.startswith("*")
            or stripped.startswith("@")
            or stripped in ("{", "}", "};", ");")
            or stripped.startswith("package ")
            or stripped.startswith("import ")
        ):
            continue

        stmt_id = f"{class_fqn}#{i}"
        statements.append(
            StatementInfo(
                stmt_id=stmt_id,
                content=stripped,
                line_number=i,
                class_fqn=class_fqn,
                file_path=str(java_file_path),
            )
        )

    return statements


def _derive_class_fqn(
    file_path: Path,
    lines: list[str],
    source_root: str | Path | None,
) -> str:
    """Derive the fully-qualified class name from file path or source.

    Strategy:
      1. If source_root is given, compute from relative path:
         source_root/org/foo/Bar.java -> org.foo.Bar
      2. Otherwise, parse the 'package' declaration and use the filename.
    """
    if source_root is not None:
        source_root = Path(source_root)
        try:
            rel = file_path.resolve().relative_to(source_root.resolve())
            # org/foo/Bar.java -> org.foo.Bar
            return str(rel.with_suffix("")).replace(os.sep, ".")
        except ValueError:
            pass  # Fall through to parsing

    # Parse package from source
    package_name = ""
    for line in lines[:30]:  # package is always near the top
        stripped = line.strip()
        if stripped.startswith("package "):
            package_name = stripped.split()[1].rstrip(";")
            break

    class_name = file_path.stem  # filename without .java
    return f"{package_name}.{class_name}" if package_name else class_name


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_java_files(
    root_dir: str | Path,
    exclude_tests: bool = True,
) -> list[Path]:
    """Recursively find all .java source files under root_dir.

    Args:
        root_dir: Directory to search.
        exclude_tests: If True, skip files under paths containing
                       'test' (case-insensitive) to avoid test code.
    """
    root_dir = Path(root_dir)
    java_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if exclude_tests:
            # Skip test directories
            rel = Path(dirpath).relative_to(root_dir)
            parts_lower = [p.lower() for p in rel.parts]
            if any("test" in p for p in parts_lower):
                continue
        for fname in filenames:
            if fname.endswith(".java"):
                java_files.append(Path(dirpath) / fname)
    return sorted(java_files)
