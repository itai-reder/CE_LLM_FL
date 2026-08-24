"""Byte-equality tests for ``src.extraction.trigger_test``.

Each fixture under ``tests/fixtures/trigger_test/<Bug>/`` holds three
files paired off against FlexFL's pre-curated dataset artifact:

* ``raw_trigger_tests.txt`` — CEFL-side raw output (copied verbatim
  from ``data/D4J/processed/<P>/<B>/trigger_tests``).
* ``<SimpleTestClass>.java`` — a synthetic test source file built so
  that the failing test method occupies the same line range as in the
  original D4J source. The portion that should land in
  ``trigger_test_clean.txt`` is byte-identical to FlexFL's curated
  source-portion; the surrounding lines are padding plus an unrelated
  method that must NOT leak into the output.
* ``expected_clean.txt`` — FlexFL's curated artifact (copied verbatim
  from ``data/input/trigger_tests/Defects4J/<Bug>.txt`` in the
  unmodified FlexFL repo).

The 5 fixture bugs span five different D4J projects (Lang, Chart, Math,
Closure, Mockito) to cover the variation in stack-trace structures,
JUnit versions, and exception-detail styles seen in practice. Adding a
new bug to ``_BUG_FIXTURES`` (and dropping its three files into the
fixture dir) is the documented way to broaden coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extraction.trigger_test import (
    TRANSITION_LINE,
    _derive_sut_packages,
    build_trigger_test_clean,
    clean_stack_trace,
    parse_failing_test,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "trigger_test"

# (bug, simple_test_class) — drives all parameterised tests below.
_BUG_FIXTURES: list[tuple[str, str]] = [
    ("Lang-1", "NumberUtilsTest"),
    ("Chart-1", "AbstractCategoryItemRendererTests"),
    ("Math-1", "BigFractionTest"),
    ("Closure-1", "CommandLineRunnerTest"),
    ("Mockito-1", "InvocationMatcherTest"),
]


def _load(bug: str, simple_class: str) -> tuple[str, str, Path]:
    """Return ``(raw, expected, test_file_path)`` for one fixture."""
    d = _FIXTURE_ROOT / bug
    raw = (d / "raw_trigger_tests.txt").read_text(encoding="utf-8")
    expected = (d / "expected_clean.txt").read_text(encoding="utf-8")
    test_file = d / f"{simple_class}.java"
    return raw, expected, test_file


@pytest.mark.parametrize(("bug", "simple_class"), _BUG_FIXTURES)
def test_parse_failing_test_extracts_test_metadata(bug: str, simple_class: str) -> None:
    raw, _, _ = _load(bug, simple_class)
    parsed = parse_failing_test(raw)
    assert parsed.test_class_fqn.endswith("." + simple_class), (
        f"test class FQN should end with .{simple_class}; got {parsed.test_class_fqn}"
    )
    assert parsed.test_method, "test_method should not be empty"
    assert parsed.fail_line > 0, "fail_line should be a positive 1-based line number"
    assert parsed.raw_stack, "raw_stack should preserve the chunk text"


@pytest.mark.parametrize(("bug", "simple_class"), _BUG_FIXTURES)
def test_clean_stack_trace_matches_expected(bug: str, simple_class: str) -> None:
    raw, expected, _ = _load(bug, simple_class)
    parsed = parse_failing_test(raw)
    pkgs = _derive_sut_packages(parsed.test_class_fqn)
    got = clean_stack_trace(parsed.raw_stack, sut_packages=pkgs)
    sep = "\n" + TRANSITION_LINE + "\n"
    assert sep in expected, f"expected_clean.txt for {bug} should contain the transition separator"
    expected_trace = expected.split(sep, 1)[1]
    assert got == expected_trace, (
        f"clean_stack_trace diverged from FlexFL fixture for {bug}.\n"
        f"--- expected ---\n{expected_trace}\n--- got ---\n{got}"
    )


@pytest.mark.parametrize(("bug", "simple_class"), _BUG_FIXTURES)
def test_build_trigger_test_clean_byte_equal_to_fixture(bug: str, simple_class: str) -> None:
    raw, expected, test_file = _load(bug, simple_class)
    got = build_trigger_test_clean(raw_failing_tests=raw, test_file=test_file)
    assert got == expected, (
        f"build_trigger_test_clean diverged from FlexFL fixture for {bug}. "
        f"Lengths: got={len(got)}, expected={len(expected)}."
    )


def test_parse_failing_test_rejects_payload_without_header() -> None:
    with pytest.raises(ValueError, match="No '--- "):
        parse_failing_test("not a trigger_tests file\n")


def test_parse_failing_test_picks_first_chunk_when_multiple_present() -> None:
    raw, _, _ = _load("Mockito-1", "InvocationMatcherTest")
    assert raw.count("--- ") >= 2, "Mockito-1 fixture must have multiple chunks"
    parsed = parse_failing_test(raw)
    assert parsed.test_class_fqn == ("org.mockito.internal.invocation.InvocationMatcherTest")


def test_extract_test_method_source_rejects_out_of_range_fail_line(tmp_path: Path) -> None:
    from src.extraction.trigger_test import extract_test_method_source

    test_file = tmp_path / "X.java"
    test_file.write_text("class X { public void m() {} }\n")
    with pytest.raises(ValueError, match="out of range"):
        extract_test_method_source(test_file=test_file, test_method="m", fail_line=100)


def test_extract_test_method_source_skips_call_site_fallback(tmp_path: Path) -> None:
    """Primary pattern fails → fallback ignores call-site uses preceded by ``.``."""
    from src.extraction.trigger_test import extract_test_method_source

    text = (
        "class X {\n"  # line 1
        "    void helper() {}\n"  # line 2
        "    void run() {\n"  # line 3
        "        obj.helper();\n"  # line 4 — call site, must be ignored
        "        boom();\n"  # line 5
        "    }\n"  # line 6
        "}\n"  # line 7
    )
    test_file = tmp_path / "X.java"
    test_file.write_text(text)
    got = extract_test_method_source(test_file=test_file, test_method="helper", fail_line=2)
    assert got == "    void helper() {}", got
