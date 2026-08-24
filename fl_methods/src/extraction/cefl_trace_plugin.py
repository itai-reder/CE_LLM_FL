"""Live trace-capture pytest plugin — the Python analogue of Defects4J's ``Formatter.java``.

D4J's JUnit ``Formatter`` writes, for every failing test, ``--- <Class>::<method>`` followed by
``Throwable.printStackTrace`` — the *full live stack trace of the real exception*, frame by frame,
independent of any reporter's verbosity. BugsInPy has no equivalent: its ``bug_buggy.txt`` is just
captured ``pytest``/``unittest`` console text, which is reporter-formatted and lossy.

This plugin restores parity. Loaded into the live FauxPy run (``python -m pytest ... -p
cefl_trace_plugin``), it hooks :func:`pytest_exception_interact` — which fires for both test
failures *and* collection/import errors — and appends one D4J-format block per event to the file
named by the ``CEFL_TRACE_OUT`` environment variable::

    --- <nodeid>
    Traceback (most recent call last):
      File "<module-relative path>", line <N>, in <func>
        <code line>
      ...
    <ExcType>: <message>

Frames come from the real traceback object (``traceback.extract_tb``), so the chain is complete
(test -> SUT -> raising line) and uniform across pytest- and unittest-style tests (pytest is the
universal runner). Each frame's filename is made **module-relative** by stripping the longest
matching ``sys.path``/cwd entry — the same resolution Python uses to find the module — so SUT
frames (including ones living inside a site-packages ``*.egg``) come out as e.g.
``ansible/galaxy/collection.py`` and stdlib/framework frames as ``unittest/case.py``, ready for
the cleaner's module-ownership filter.

The plugin is **standalone** (no ``src.*`` imports) and deliberately written in **Python 3.5-safe
syntax** — no ``from __future__ import annotations`` (3.7+), no ``X | Y`` / ``list[...]`` annotations
(3.9/3.10), no PEP 526 variable annotations (3.6), no f-strings (3.6) — because it loads into the
bug's *own* conda env, which can be as old as Python 3.5/3.6 across the BugsInPy corpus; a syntax
error here would abort the whole FauxPy run rather than degrade. Every hook is also defensively
wrapped: a capture failure must never break the coverage run it piggybacks on. When
``CEFL_TRACE_OUT`` is unset the hooks no-op.
"""

# NOTE: the Python 3.5-safe syntax here (typing.Optional/List, str.format) is intentional — see the
# module docstring. The `UP` (pyupgrade) lint family is disabled for this file in pyproject.toml so
# it is not auto-modernised back to 3.7+/3.9+/3.10+ constructs.

import contextlib
import os
import sys
import traceback
from typing import Any, List, Optional

_ENV_OUT = "CEFL_TRACE_OUT"
_TRACEBACK_HEADER = "Traceback (most recent call last):"


def _out_path() -> Optional[str]:
    """The capture target, or ``None`` (plugin disabled) when the env var is unset/empty."""
    path = os.environ.get(_ENV_OUT)
    return path or None


def _syspath_prefixes() -> List[str]:
    """Absolute ``sys.path`` entries plus cwd, longest first (for module-relative stripping)."""
    raw = list(sys.path)
    with contextlib.suppress(OSError):
        raw.append(os.getcwd())
    prefixes = []  # type: List[str]
    for entry in raw:
        if not entry:
            continue
        try:
            ap = os.path.abspath(entry)
        except OSError:
            continue
        prefixes.append(ap.rstrip(os.sep) + os.sep)
    # Longest prefix first so the most specific (e.g. an .egg under site-packages) wins.
    return sorted(set(prefixes), key=len, reverse=True)


def _module_relative(filename: str) -> str:
    """Make *filename* relative to its owning ``sys.path`` entry -> canonical module path.

    ``.../site-packages/ansible_base-X.egg/ansible/galaxy/collection.py`` -> ``ansible/galaxy/collection.py``;
    ``<checkout>/test/units/galaxy/test_collection.py`` -> ``test/units/galaxy/test_collection.py``.
    Non-file frames (``<frozen importlib._bootstrap>``) and paths under no entry are returned as-is.
    """
    if not filename or filename.startswith("<"):
        return filename
    try:
        absname = os.path.abspath(filename)
    except OSError:
        return filename
    for prefix in _syspath_prefixes():
        if absname.startswith(prefix):
            return absname[len(prefix) :].replace(os.sep, "/")
    return filename.replace(os.sep, "/")


def _exception_header(exc_type: Optional[type], exc_value: Optional[BaseException]) -> str:
    """The bare ``<ExcType>: <message>`` line (last line of ``format_exception_only``)."""
    try:
        formatted = traceback.format_exception_only(exc_type, exc_value)
    except Exception:  # never let header rendering break capture
        return exc_type.__name__ if exc_type else "Exception"
    lines = "".join(formatted).strip().splitlines()
    return lines[-1] if lines else ""


def _render_block(nodeid: str, exc_type: Any, exc_value: Any, tb: Any) -> str:
    """Render one ``--- nodeid`` D4J-format block from a real traceback object."""
    out = ["--- " + nodeid, _TRACEBACK_HEADER]
    # extract_tb items are (filename, lineno, name, line) — index, not attr, for old-Python compat.
    for frame in traceback.extract_tb(tb):
        out.append(
            '  File "{0}", line {1}, in {2}'.format(_module_relative(frame[0]), frame[1], frame[2])
        )
        code = frame[3]
        if code:
            out.append("    " + code.strip())
    out.append(_exception_header(exc_type, exc_value))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    """Truncate the capture file once per session so a re-run starts clean."""
    out = _out_path()
    if not out:
        return
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w"):
            pass
    except OSError:
        pass


def pytest_exception_interact(node: Any, call: Any, report: Any) -> None:
    """Append the failing test/collection's full live traceback in D4J format."""
    out = _out_path()
    if not out:
        return
    try:
        excinfo = call.excinfo
        if excinfo is None:
            return
        nodeid = getattr(node, "nodeid", "") or getattr(report, "nodeid", "") or "<unknown>"
        block = _render_block(nodeid, excinfo.type, excinfo.value, excinfo.tb)
        with open(out, "a") as handle:
            handle.write(block)
    except Exception as exc:  # capture must never break the coverage run
        node_id = getattr(node, "nodeid", "?")
        sys.stderr.write("cefl_trace_plugin: capture failed for {0}: {1}\n".format(node_id, exc))
