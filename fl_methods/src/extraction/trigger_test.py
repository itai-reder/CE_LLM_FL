"""Build the cleaned trigger-test artifact consumed by Agent4LR.

The Agent4LR phase expects a single combined text block per bug,
mirroring FlexFL's pre-curated ``data/input/trigger_tests/{dataset}/{bug}.txt``
files. The block contains:

1. The failing test method's source code, truncated **at the failing
   assertion** (last retained line = the source line that raised the
   exception);
2. A literal transition line: ``"The last line shown above failed with the
   following stack trace."``;
3. The cleaned stack trace: framework / reflection / build-tool frames
   stripped, plus non-SUT frames (e.g. ``java.lang.*``) dropped. The
   exception type / message header is preserved verbatim.

Neither the original nor the modular FlexFL pipeline extracts this file —
both consume FlexFL's pre-curated dataset artifacts. CEFL must build the
extractor. The algorithm here was derived by reverse-engineering FlexFL's
output against five paired CEFL/FlexFL fixtures spanning five projects
(Lang, Chart, Math, Closure, Mockito).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.extraction.d4j import D4JRepo

# Prefix tokens identifying frames to drop. Derived from the 5 FlexFL
# fixtures: junit + JVM reflection (Java 8 + Java 9+) + Ant/Gradle/TestNG.
DEFAULT_FRAMEWORK_PREFIXES: tuple[str, ...] = (
    "junit.",
    "org.junit.",
    "sun.reflect.",
    "jdk.internal.reflect.",
    "java.lang.reflect.",
    "org.apache.tools.ant.",
    "org.gradle.",
    "org.testng.",
)

TRANSITION_LINE = "The last line shown above failed with the following stack trace."


@dataclass(frozen=True)
class ParsedFailure:
    """Structured view over a raw ``trigger_tests`` / ``failing_tests`` chunk.

    Only the FIRST ``--- <fqn>::<method>`` chunk is parsed: bugs with
    multiple trigger tests follow FlexFL's convention of curating
    against the first one.

    ``fail_line`` is the source line of the failing test method's own
    frame (the bottom-most occurrence if the method appears multiple
    times — recursive test helpers).
    """

    test_class_fqn: str
    test_method: str
    fail_line: int
    raw_stack: str  # the exception header + all subsequent lines in the chunk


def _strip_module(class_fqn: str) -> str:
    """Strip Java 9+ module prefix from a frame's class FQN.

    ``"java.base/jdk.internal.reflect.NativeMethodAccessorImpl"`` →
    ``"jdk.internal.reflect.NativeMethodAccessorImpl"``. Java 8 traces
    have no module prefix; left untouched.
    """
    if "/" in class_fqn:
        return class_fqn.split("/", 1)[1]
    return class_fqn


_FRAME_PREFIX = "at "


def _parse_frame_line(line: str) -> tuple[str, str, str, int] | None:
    """Parse ``\\tat <class>.<method>(<file>:<line>)`` → tuple, or None.

    Returns ``(class_fqn_with_module, method, file_part, line_num)``.
    ``line_num`` is ``-1`` for native frames or unknown sources.
    """
    s = line.strip()
    if not s.startswith(_FRAME_PREFIX):
        return None
    rest = s[len(_FRAME_PREFIX) :]
    open_paren = rest.rfind("(")
    if open_paren < 0:
        return None
    head = rest[:open_paren]
    body = rest[open_paren + 1 :]
    if body.endswith(")"):
        body = body[:-1]
    file_part: str
    line_num: int
    if ":" in body:
        file_part, _, line_part = body.rpartition(":")
        try:
            line_num = int(line_part)
        except ValueError:
            line_num = -1
    else:
        file_part = body
        line_num = -1
    last_dot = head.rfind(".")
    if last_dot < 0:
        return None
    class_fqn = head[:last_dot]
    method = head[last_dot + 1 :]
    return class_fqn, method, file_part, line_num


def parse_failing_test(raw: str) -> ParsedFailure:
    """Parse a raw ``trigger_tests`` / ``failing_tests`` payload.

    Format::

        --- <test-class-fqn>::<test-method>
        <exception-header>
        [multi-line exception detail]
        \\tat <class>.<method>(<file>:<line>)
        ...
        [--- next-chunk-header ...]  # if multiple chunks

    Returns a :class:`ParsedFailure` for the first chunk. Raises
    :class:`ValueError` if no ``---`` header is present or the test
    method's own stack frame can't be located (so ``fail_line`` would
    be -1).
    """
    lines = raw.splitlines()
    test_class_fqn = test_method = ""
    chunk_start = -1
    for i, line in enumerate(lines):
        if line.startswith("--- "):
            header = line[4:].strip()
            if "::" in header:
                test_class_fqn, test_method = header.split("::", 1)
                chunk_start = i + 1
                break
    if chunk_start < 0:
        raise ValueError("No '--- <fqn>::<method>' header in raw failing_tests payload")

    chunk_end = len(lines)
    for j in range(chunk_start, len(lines)):
        if lines[j].startswith("--- "):
            chunk_end = j
            break
    chunk_lines = lines[chunk_start:chunk_end]
    chunk = "\n".join(chunk_lines)

    fail_line = -1
    for line in chunk_lines:
        frame = _parse_frame_line(line)
        if frame is None:
            continue
        class_fqn, method, _, line_num = frame
        if _strip_module(class_fqn) == test_class_fqn and method == test_method:
            fail_line = line_num  # keep the LAST (deepest) occurrence

    if fail_line < 0:
        raise ValueError(
            f"Could not locate stack frame for {test_class_fqn}.{test_method} in raw trace"
        )

    return ParsedFailure(
        test_class_fqn=test_class_fqn,
        test_method=test_method,
        fail_line=fail_line,
        raw_stack=chunk,
    )


def clean_stack_trace(
    raw: str,
    *,
    sut_packages: tuple[str, ...],
    framework_prefixes: tuple[str, ...] = DEFAULT_FRAMEWORK_PREFIXES,
) -> str:
    """Produce the FlexFL-style cleaned stack trace.

    Behaviour:

    * The **first non-empty line** of ``raw`` is the exception header
      (e.g. ``"java.lang.NumberFormatException: For input string: ..."``)
      and is preserved verbatim, including any trailing whitespace.
    * **Multi-line exception detail** between the header and the first
      ``at `` frame is dropped (FlexFL's curated output never includes
      it — see Closure-1 reference).
    * Each ``at <class>.<method>(<file>:<line>)`` frame is classified:
        - declared class FQN starts with any token in
          ``framework_prefixes`` → drop;
        - declared class FQN starts with any token in ``sut_packages``
          (treated as ``"<pkg>."`` prefix match) → keep with original
          whitespace;
        - otherwise (JDK frames like ``java.lang.*``) → drop.
    * Lines that are neither the exception header nor an ``at `` frame
      after the header (blank lines, ``Caused by:`` continuations,
      indented detail) are dropped.

    The result has no trailing newline.
    """
    lines = raw.splitlines()
    if not lines:
        return ""

    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(_FRAME_PREFIX):
            break
        if line.strip():
            header_idx = i
            break
    if header_idx < 0:
        return ""

    out: list[str] = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        frame = _parse_frame_line(line)
        if frame is None:
            continue
        class_fqn = _strip_module(frame[0])
        if any(class_fqn.startswith(p) for p in framework_prefixes):
            continue
        if any(class_fqn == pkg or class_fqn.startswith(pkg + ".") for pkg in sut_packages):
            out.append(line)
        # else: JDK or other library — drop
    return "\n".join(out)


def _derive_sut_packages(test_class_fqn: str) -> tuple[str, ...]:
    """SUT prefix derivation: the test class's direct package.

    For ``"org.apache.commons.lang3.math.NumberUtilsTest"`` the SUT
    prefix is ``("org.apache.commons.lang3.math",)``. Frames in the
    same package or a sub-package are kept; ancestor packages are not
    (matches the 5-fixture observations).
    """
    if "." in test_class_fqn:
        return (test_class_fqn.rsplit(".", 1)[0],)
    return ()


_DECL_PRIMARY_PATTERN = re.compile(
    r"^\s*(?:public|protected|private)\s+"
    r"(?:static\s+|final\s+|synchronized\s+|abstract\s+)*"
    r"\w+(?:\s*<[^>]*>)?\s+"  # return type, with optional generics
    r"{name}\s*\("  # method name + (
)


def extract_test_method_source(
    *,
    test_file: Path,
    test_method: str,
    fail_line: int,
) -> str:
    """Slice the test method body from declaration through ``fail_line``.

    Two-pass locator:

    1. Match lines against ``"<access> [modifiers] <return_type>
       <test_method>("`` and take the latest match before ``fail_line``.
       Handles the common Java test method shape (the case for all 5
       FlexFL fixtures).
    2. If the primary pattern doesn't match, fall back to a looser
       ``"<test_method>("`` search that excludes call-site uses (lines
       where the match is preceded by ``.`` or ``=``).

    The slice is ``lines[decl_idx:fail_line]`` (i.e. inclusive of
    ``fail_line``), joined by ``\\n``. **No trailing newline appended**
    and **no synthetic close-brace** added — matches FlexFL's curated
    format which deliberately leaves the body unbalanced as a visual
    cue that execution stopped at the failing line.
    """
    text = test_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if fail_line < 1 or fail_line > len(lines):
        raise ValueError(
            f"fail_line={fail_line} is out of range for {test_file} ({len(lines)} lines)"
        )

    primary = re.compile(_DECL_PRIMARY_PATTERN.pattern.format(name=re.escape(test_method)))
    name_re = re.compile(rf"\b{re.escape(test_method)}\s*\(")

    decl_idx = -1
    for i in range(fail_line):  # 0-based; fail_line is 1-based
        if primary.search(lines[i]):
            decl_idx = i

    if decl_idx < 0:
        for i in range(fail_line):
            m = name_re.search(lines[i])
            if m is None:
                continue
            prefix = lines[i][: m.start()].rstrip()
            if prefix.endswith(".") or prefix.endswith("="):
                continue
            decl_idx = i

    if decl_idx < 0:
        raise ValueError(
            f"Could not locate declaration of {test_method!r} in {test_file} before line {fail_line}"
        )

    return "\n".join(lines[decl_idx:fail_line])


def build_trigger_test_clean(
    *,
    raw_failing_tests: str,
    test_file: Path,
    sut_packages: tuple[str, ...] | None = None,
) -> str:
    """Compose the full ``trigger_test_clean.txt`` body.

    Sequence:

    1. :func:`parse_failing_test` → :class:`ParsedFailure`.
    2. Derive ``sut_packages`` (when not given) from the test class's
       package via :func:`_derive_sut_packages`.
    3. :func:`extract_test_method_source` → source slice.
    4. :data:`TRANSITION_LINE`.
    5. :func:`clean_stack_trace` → cleaned trace.

    Joined with ``\\n``, no trailing newline. Result is byte-identical
    to FlexFL's curated artifact for the 5 paired fixtures
    (``tests/fixtures/trigger_test/``).
    """
    parsed = parse_failing_test(raw_failing_tests)
    pkgs = sut_packages if sut_packages is not None else _derive_sut_packages(parsed.test_class_fqn)
    src = extract_test_method_source(
        test_file=test_file, test_method=parsed.test_method, fail_line=parsed.fail_line
    )
    cleaned = clean_stack_trace(parsed.raw_stack, sut_packages=pkgs)
    return f"{src}\n{TRANSITION_LINE}\n{cleaned}"


def save_trigger_test_clean(
    repo: D4JRepo,
    *,
    skip_existing: bool = True,
) -> Path:
    """Build ``trigger_test_clean.txt`` next to the raw ``trigger_tests``.

    Reads ``<processed>/trigger_tests``, locates the test class file via
    ``repo.get_src_tests_dir()`` plus the parsed test class FQN, calls
    :func:`build_trigger_test_clean`, and writes the result.

    Returns the path to the written file. Honours ``skip_existing``.
    Raises :class:`FileNotFoundError` if ``trigger_tests`` or the test
    source file is missing — callers must ``repo.checkout()`` first.
    """
    processed_dir = repo.output_dir
    raw_path = processed_dir / "trigger_tests"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw trigger_tests not found at {raw_path}. Extraction must run first."
        )
    out_path = processed_dir / "trigger_test_clean.txt"
    if skip_existing and out_path.exists():
        return out_path

    raw = raw_path.read_text(encoding="utf-8")
    parsed = parse_failing_test(raw)

    test_src_dir = repo.get_src_tests_dir()
    rel = parsed.test_class_fqn.replace(".", "/") + ".java"
    test_file = test_src_dir / rel
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test source not found at {test_file} "
            f"(test class {parsed.test_class_fqn!r}). Did you run repo.checkout()?"
        )

    body = build_trigger_test_clean(raw_failing_tests=raw, test_file=test_file)
    out_path.write_text(body, encoding="utf-8")
    return out_path
