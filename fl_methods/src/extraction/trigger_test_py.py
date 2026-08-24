"""Build the cleaned BugsInPy failing-test artifact (``trigger_test_clean.txt``).

The Python analogue of :mod:`src.extraction.trigger_test` (the Java/JUnit cleaner). It produces
the **same three-part artifact** D4J's ``trigger_test_clean.txt`` does — consumed by Agent4LR (and
Agent4SR):

1. the failing test method's **source slice**, decorators/``def`` → the failing line (truncated
   there, no synthetic close, read **from the checkout**: native indentation, blank lines kept);
2. the verbatim transition literal
   ``"The last line shown above failed with the following stack trace."``;
3. a single ``Traceback (most recent call last):`` block — framework / third-party frames dropped,
   SUT + test frames kept, the bare exception header preserved.

**Source of the trace.** Unlike D4J — whose ``failing_tests``/``trigger_tests`` is a live
``printStackTrace`` capture — BugsInPy ships only ``bug_buggy.txt`` (lossy reporter console text).
CEFL therefore captures the trace itself, *live*, via :mod:`src.extraction.cefl_trace_plugin`
(the ``Formatter.java`` analogue) attached to the FauxPy run; the plugin emits a structured
``trigger_trace.txt`` (``--- <nodeid>`` + a real ``Traceback`` block, frames already made
module-relative). This module consumes that — there is **no reporter-text parsing**: a deterministic
canonical-trace parse, then the same structure-driven cleaning the D4J path uses.

**Mirrorable-vs-blank.** A single guard decides it: the failing frame must lie inside the trigger
test's own ``def`` span (resolved via :mod:`src.common.python_parser`). Collection/import errors
frame the test file at a module-level line (outside the def), captured-passes produce no failing
block, not-collected runs have no sliceable test — so all route to a **blank** artifact + a
``logging.warning``, never a crash.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.python_parser import extract_methods_from_python
from src.extraction.trigger_test import TRANSITION_LINE

if TYPE_CHECKING:
    from src.extraction.bugsinpy import BugsInPyRepo

logger = logging.getLogger(__name__)

# --- canonical trigger_trace.txt structure ---------------------------------
_BLOCK_HEADER = "--- "
_TRACEBACK_HEADER = "Traceback (most recent call last):"
_FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')

# Defensive path normalisation. The plugin already emits module-relative paths; these strip any
# absolute prefix that slips through (e.g. a frame under no sys.path entry), keeping the cleaner
# robust and fixture-compatible.
_CONDA_ENV = re.compile(r"/opt/conda/envs/[0-9a-f]{32}/lib/python[0-9.]+/site-packages/")
_BIP_TEMP = re.compile(r'[^\s"]*/BugsInPy/temp/projects/[^/\s"]+/')
_EGG = re.compile(r"[^\s/\"]+\.egg/")

# Top-level modules that are never SUT (stdlib test machinery / loaders).
_FRAMEWORK_TOPLEVEL = frozenset(
    {"_pytest", "pytest", "pluggy", "unittest", "runpy", "importlib", "nose", "py"}
)


@dataclass
class _Frame:
    """One stack frame: normalised path, 1-based line, function, stripped code line."""

    path: str
    line: int
    func: str
    code: str


# ---------------------------------------------------------------------------
# Path / name helpers
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Strip conda-env / ``*.egg/`` / BugsInPy-temp prefixes → repo/module-relative path."""
    path = _CONDA_ENV.sub("", path)
    path = _BIP_TEMP.sub("", path)
    path = _EGG.sub("", path)
    return path.lstrip("/")


def _top_level_package(rel_path: str) -> str:
    """First path component of a ``foo/bar/baz.py`` relative path (the top-level package)."""
    rel_path = rel_path.replace("\\", "/").lstrip("./")
    head = rel_path.split("/", 1)[0]
    return head[:-3] if head.endswith(".py") else head


def _method_name(qualname: str) -> str:
    """``Class.method`` / ``Class::method`` / ``path.py::func[param]`` → the bare method name.

    Strips a trailing pytest parametrization suffix (``test_x[case1]`` → ``test_x``).
    """
    name = re.split(r"[.:]", qualname)[-1]
    return name.split("[", 1)[0]


# ---------------------------------------------------------------------------
# Canonical trace parsing
# ---------------------------------------------------------------------------


def _parse_one_block(body: list[str]) -> tuple[list[_Frame], str]:
    """Parse one block's body (after its ``--- nodeid`` header) into frames + exception header."""
    frames: list[_Frame] = []
    header = ""
    i, n = 0, len(body)
    while i < n:
        line = body[i]
        m = _FRAME.match(line)
        if m:
            code = ""
            if i + 1 < n and body[i + 1].startswith((" ", "\t")) and not _FRAME.match(body[i + 1]):
                code = body[i + 1].strip()
                i += 1
            frames.append(
                _Frame(
                    path=_normalize_path(m.group("path")),
                    line=int(m.group("line")),
                    func=m.group("func").strip(),
                    code=code,
                )
            )
        elif line.strip() and line.strip() != _TRACEBACK_HEADER:
            header = line.strip()  # last non-frame, non-empty line = the exception header
        i += 1
    return frames, header


def _parse_trace_blocks(trace_text: str) -> list[tuple[str, list[_Frame], str]]:
    """Split ``trigger_trace.txt`` into ``(nodeid, frames, exception_header)`` blocks."""
    lines = trace_text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(_BLOCK_HEADER)]
    blocks: list[tuple[str, list[_Frame], str]] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        nodeid = lines[start][len(_BLOCK_HEADER) :].strip()
        frames, header = _parse_one_block(lines[start + 1 : end])
        blocks.append((nodeid, frames, header))
    return blocks


def _select_block(
    blocks: list[tuple[str, list[_Frame], str]], test_file: Path, method: str
) -> tuple[str, list[_Frame], str] | None:
    """Pick the block for the trigger: prefer file+method match, then file, then the first block."""
    base = test_file.name
    for blk in blocks:
        node_path = blk[0].split("::", 1)[0]
        if Path(node_path).name == base and (not method or _method_name(blk[0]) == method):
            return blk
    for blk in blocks:  # file-level / class-level trigger: any block in the test file
        if Path(blk[0].split("::", 1)[0]).name == base:
            return blk
    return blocks[0] if blocks else None


# ---------------------------------------------------------------------------
# Source slice (from the checkout, via the ast parser)
# ---------------------------------------------------------------------------


def _method_spans(test_file: Path, method: str) -> list[tuple[int, int]]:
    """All ``(decl_start, end_line)`` spans of methods named *method* (decorators folded in)."""
    return [
        (mi.start_line, mi.end_line)
        for mi in extract_methods_from_python(test_file)
        if mi.method_name == method
    ]


def _slice_source(test_file: Path, decl_start: int, fail_line: int) -> str | None:
    """Slice ``lines[decl_start-1:fail_line]`` from the checkout file (verbatim, inclusive)."""
    try:
        text = test_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    lines = text.splitlines()
    if decl_start < 1 or fail_line > len(lines) or decl_start > fail_line:
        return None
    return "\n".join(lines[decl_start - 1 : fail_line])


# ---------------------------------------------------------------------------
# Frame selection + cleaning + rendering
# ---------------------------------------------------------------------------


def _find_fail_frame(
    frames: list[_Frame], test_relpath: str, spans: list[tuple[int, int]]
) -> _Frame | None:
    """Deepest frame in the trigger test file whose line lands inside a trigger-method span.

    Line-in-span (the spec's guard), not ``func == method`` — the failing frame is often a nested
    function/callback defined *inside* the test body (e.g. a decorated route handler).
    """
    base = Path(test_relpath).name
    match: _Frame | None = None
    for fr in frames:  # frames are outer→inner; keep the LAST (deepest) match
        if Path(fr.path).name == base and any(lo <= fr.line <= hi for lo, hi in spans):
            match = fr
    return match


def _is_sut_frame(fr: _Frame, sut_packages: frozenset[str], test_relpath: str) -> bool:
    """Keep test-file frames and frames whose top-level package is a SUT package."""
    if Path(fr.path).name == Path(test_relpath).name:
        return True
    top = _top_level_package(fr.path)
    if top in _FRAMEWORK_TOPLEVEL:
        return False
    return top in sut_packages


def _render_traceback(frames: list[_Frame], header: str) -> str:
    """Compose the synthetic ``Traceback (most recent call last):`` block."""
    out = [_TRACEBACK_HEADER]
    for fr in frames:
        out.append(f'  File "{fr.path}", line {fr.line}, in {fr.func}')
        if fr.code:
            out.append(f"    {fr.code}")
    out.append(header)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_python_trigger_clean(
    *,
    trigger_trace: str,
    test_file: Path,
    test_qualname: str,
    sut_packages: frozenset[str],
) -> str:
    """Build the cleaned trigger artifact, or ``""`` when the failure is not mirrorable.

    *trigger_trace* is the live-captured ``trigger_trace.txt`` (see
    :mod:`src.extraction.cefl_trace_plugin`). Returns the three-part body (no trailing newline);
    an empty string means a guard failed (no matching/failing block, no in-test fail line,
    unsliceable test, or no exception header) — the caller writes a blank file.
    """
    if not trigger_trace.strip():
        return ""
    blocks = _parse_trace_blocks(trigger_trace)
    if not blocks:
        return ""

    method = _method_name(test_qualname)
    block = _select_block(blocks, test_file, method)
    if block is None:
        return ""
    nodeid, frames, header = block

    # Class-/file-level trigger: the trigger names a class or file (no method def by that name —
    # the run targets the whole class/file), so recover the actually-failing method from the
    # selected block's nodeid.
    spans = _method_spans(test_file, method)
    if not spans:
        recovered = _method_name(nodeid)
        if recovered and _method_spans(test_file, recovered):
            method = recovered
            spans = _method_spans(test_file, method)
    if not frames or not header or not spans:
        return ""

    # Guard: the deepest test-file frame landing inside a span of the trigger method (resolves
    # same-named methods across classes; nested in-test callbacks still qualify).
    test_relpath = test_file.name  # frames are matched by basename (paths are module-relative)
    fail_frame = _find_fail_frame(frames, test_relpath, spans)
    if fail_frame is None:
        return ""
    enclosing = [s for s in spans if s[0] <= fail_frame.line <= s[1]]
    span = min(enclosing, key=lambda s: s[1] - s[0])  # smallest enclosing span

    # Source slice from the checkout (the test frame's code = the sliced fail line).
    slice_src = _slice_source(test_file, span[0], fail_frame.line)
    if not slice_src:
        return ""
    sliced_lines = slice_src.splitlines()
    fail_code = sliced_lines[-1].strip() if sliced_lines else fail_frame.code

    # Clean: keep test + SUT frames. For test-file frames, render the **repo-relative** path from
    # the nodeid (``test/units/.../test_collection.py``) rather than the frame's sys.path-relative
    # path, which can drop a leading ``test/`` when that dir is not an importable package (D4J
    # convention is repo-relative). The test frame's code is the sliced fail line.
    node_path = nodeid.split("::", 1)[0]
    use_node_path = bool(node_path) and Path(node_path).name == Path(test_relpath).name
    kept = [fr for fr in frames if _is_sut_frame(fr, sut_packages, test_relpath)]
    for fr in kept:
        if Path(fr.path).name == Path(test_relpath).name:
            if use_node_path:
                fr.path = node_path
            if fr.line == fail_frame.line:
                fr.code = fail_code
    if not kept:
        return ""

    return f"{slice_src}\n{TRANSITION_LINE}\n{_render_traceback(kept, header)}"


def _read_sut_packages(method_signatures_csv: Path) -> frozenset[str]:
    """Top-level SUT packages from ``method_signatures.csv`` (corpus-id module prefixes)."""
    if not method_signatures_csv.exists():
        return frozenset()
    pkgs: set[str] = set()
    for raw in method_signatures_csv.read_text(encoding="utf-8").splitlines()[1:]:
        corpus_id = raw.split(";", 1)[0]
        module = corpus_id.split("$", 1)[0]
        if module:
            pkgs.add(module.split(".", 1)[0])
    return frozenset(pkgs)


def save_python_trigger_clean(repo: BugsInPyRepo, *, skip_existing: bool = True) -> Path:
    """Build ``trigger_test_clean.txt`` for a BugsInPy bug from its live-captured trace + checkout.

    Reads ``trigger_trace.txt`` (the FauxPy-run capture; see :mod:`src.extraction.cefl_trace_plugin`)
    from :attr:`repo.output_dir`, resolves the trigger test file/qualname from ``bug.info`` + the
    trigger list, and the SUT packages from ``method_signatures.csv``. A non-mirrorable failure (or
    a missing/blank trace) yields a **blank** file + a ``logging.warning`` (never raises). Honours
    ``skip_existing``. Requires the checkout on disk (the ``repo_setup`` step) and the trace
    (produced by the ``gzoltar``/FauxPy step).
    """
    out_path = repo.output_dir / "trigger_test_clean.txt"
    if skip_existing and out_path.exists():
        return out_path

    prefix = f"{repo.project}-{repo.bug_id}"
    trace_path = repo.output_dir / "trigger_trace.txt"
    trigger_trace = (
        trace_path.read_text(encoding="utf-8", errors="ignore") if trace_path.exists() else ""
    )

    triggers = repo.get_trigger_tests()
    if triggers:
        # Resolve the test FILE from the trigger's module — authoritative, and robust to a
        # ``;``-joined multi-file ``bug.info`` test_file (e.g. pandas ``a.py;b.py``).
        trigger = triggers[0]
        module = trigger.split("::", 1)[0].split("$", 1)[0]
        test_rel = module.replace(".", "/") + ".py"
        qualname = (trigger.split("$", 1)[-1] if "$" in trigger else trigger).replace("::", ".")
    else:
        # File-level trigger (``pytest <file>`` with no method): take the first ``bug.info`` test
        # file; the failing method is recovered from the trace block in build.
        test_rel = (repo._bug_info_value("test_file") or "").split(";")[0].split()[0]
        qualname = test_rel[:-3].replace("/", ".") if test_rel.endswith(".py") else ""

    body = ""
    if not test_rel:
        logger.warning(
            "%s: blank trigger_test_clean.txt (no test_file/trigger in bug.info)", prefix
        )
    elif not trigger_trace.strip():
        logger.warning(
            "%s: blank trigger_test_clean.txt (no trigger_trace.txt — FauxPy run produced no "
            "captured failure)",
            prefix,
        )
    else:
        sut_packages = _read_sut_packages(repo.output_dir / "method_signatures.csv")
        body = build_python_trigger_clean(
            trigger_trace=trigger_trace,
            test_file=repo.get_src_tests_dir() / test_rel,
            test_qualname=qualname,
            sut_packages=sut_packages,
        )
        if not body:
            logger.warning(
                "%s: blank trigger_test_clean.txt (failure not mirrorable — no in-test fail line "
                "/ unsliceable test / no exception header)",
                prefix,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body + "\n" if body else "", encoding="utf-8")
    return out_path
