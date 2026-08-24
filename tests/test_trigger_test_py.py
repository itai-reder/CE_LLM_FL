"""Tests for the BugsInPy failing-test cleaner (src.extraction.trigger_test_py).

The cleaner consumes the **live-captured** ``trigger_trace.txt`` (a ``--- nodeid`` header + a real
``Traceback`` block per failing test, frames already module-relative — emitted by
:mod:`src.extraction.cefl_trace_plugin`) and mirrors D4J's three-part ``trigger_test_clean.txt``
(source slice → transition line → cleaned ``Traceback``). A single guard — the failing frame inside
the trigger test's ``def`` span — decides mirrorable-vs-blank, with no per-pattern categorisation.

The trace is provided here as canonical fixtures/inline strings; the source slice is provided by a
synthetic test file padded to the capture's line numbers (the production path reads it from the
checkout).
"""

from __future__ import annotations

from pathlib import Path

from src.extraction.trigger_test import TRANSITION_LINE
from src.extraction.trigger_test_py import (
    build_python_trigger_clean,
    save_python_trigger_clean,
)

FIXTURES = Path(__file__).parent / "fixtures" / "trigger_test_py"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _synthetic_test_file(
    tmp_path: Path,
    *,
    name: str,
    slice_lines: list[str],
    fail_line: int,
    class_header: str | None = None,
) -> Path:
    """Write a test file (basename must match the trace's frame) placing ``slice_lines`` so the
    last line lands on ``fail_line``."""
    decl_start = fail_line - len(slice_lines) + 1
    pad_count = decl_start - 1
    head: list[str] = []
    if class_header is not None:
        head.append(class_header)
        pad_count -= 1
    body = head + ["    # pad"] * pad_count + slice_lines + ["        pass"]
    path = tmp_path / name
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


class TestBuildPythonTriggerClean:
    def test_empty_input_is_blank(self, tmp_path: Path) -> None:
        assert (
            build_python_trigger_clean(
                trigger_trace="",
                test_file=tmp_path / "x.py",
                test_qualname="t",
                sut_packages=frozenset(),
            )
            == ""
        )

    def test_ansible_1_byte_exact(self, tmp_path: Path) -> None:
        """pytest mirror — framework frame dropped, SUT frames kept — byte-exact."""
        expected = _read("ansible_1_expected.txt")
        slice_lines = expected.split("\n" + TRANSITION_LINE + "\n", 1)[0].splitlines()
        test_file = _synthetic_test_file(
            tmp_path, name="test_collection.py", slice_lines=slice_lines, fail_line=1169
        )
        out = build_python_trigger_clean(
            trigger_trace=_read("ansible_1_trigger_trace.txt"),
            test_file=test_file,
            test_qualname="test_verify_collections_no_version",
            sut_packages=frozenset({"ansible"}),
        )
        assert out + "\n" == expected

    def test_unittest_style_is_uniform_mirror(self, tmp_path: Path) -> None:
        """A trace from a unittest-style test comes out identically (uniform canonical format)."""
        trace = (
            "--- test/test_utils.py::TestUtil::test_match_str\n"
            "Traceback (most recent call last):\n"
            '  File "unittest/case.py", line 59, in testPartExecutor\n'
            "    yield\n"
            '  File "test/test_utils.py", line 1076, in test_match_str\n'
            "    self.assertFalse(match_str('is_live', {'is_live': False}))\n"
            '  File "unittest/case.py", line 678, in assertFalse\n'
            "    raise self.failureException(msg)\n"
            "AssertionError: True is not false\n"
        )
        slice_lines = [
            "    def test_match_str(self):",
            "        self.assertFalse(match_str('is_live', {'is_live': False}))",
        ]
        test_file = _synthetic_test_file(
            tmp_path,
            name="test_utils.py",
            slice_lines=slice_lines,
            fail_line=1076,
            class_header="class TestUtil:",
        )
        out = build_python_trigger_clean(
            trigger_trace=trace,
            test_file=test_file,
            test_qualname="TestUtil.test_match_str",
            sut_packages=frozenset({"youtube_dl"}),
        )
        assert "    def test_match_str(self):" in out  # source slice present
        assert TRANSITION_LINE in out
        assert "Traceback (most recent call last):" in out
        assert 'File "test/test_utils.py", line 1076, in test_match_str' in out
        assert out.rstrip().endswith("AssertionError: True is not false")
        assert "unittest/case.py" not in out  # framework frames dropped

    def test_egg_and_conda_prefixes_stripped(self, tmp_path: Path) -> None:
        """SUT frames living in a site-packages ``.egg`` are normalised to module-relative."""
        expected = _read("ansible_1_expected.txt")
        slice_lines = expected.split("\n" + TRANSITION_LINE + "\n", 1)[0].splitlines()
        conda = "/opt/conda/envs/0123456789abcdef0123456789abcdef/lib/python3.6/site-packages"
        egg = "ansible_base-2.11.0.dev0-py3.6.egg"
        trace = (
            "--- test/units/galaxy/test_collection.py::test_verify_collections_no_version\n"
            "Traceback (most recent call last):\n"
            '  File "test/units/galaxy/test_collection.py", line 1169, in '
            "test_verify_collections_no_version\n"
            "    collection.verify_collections(collections, './', local_collection.api, "
            "False, False)\n"
            f'  File "{conda}/{egg}/ansible/galaxy/collection.py", line 679, in '
            "verify_collections\n"
            "    allow_pre_release=allow_pre_release)\n"
            "TypeError: 'GalaxyAPI' object is not iterable\n"
        )
        out = build_python_trigger_clean(
            trigger_trace=trace,
            test_file=_synthetic_test_file(
                tmp_path, name="test_collection.py", slice_lines=slice_lines, fail_line=1169
            ),
            test_qualname="test_verify_collections_no_version",
            sut_packages=frozenset({"ansible"}),
        )
        assert "/opt/conda/envs/" not in out  # conda-env prefix gone
        assert ".egg/" not in out  # egg prefix gone
        assert 'File "ansible/galaxy/collection.py", line 679, in verify_collections' in out

    def test_collection_error_is_blank(self, tmp_path: Path) -> None:
        """An import/collection error frames the test file outside any def span → blank."""
        trace = (
            "--- test/keras/test_callbacks.py\n"
            "Traceback (most recent call last):\n"
            '  File "test/keras/test_callbacks.py", line 5, in <module>\n'
            "    from keras.missing import thing\n"
            '  File "keras/__init__.py", line 3, in <module>\n'
            "    raise ImportError('no thing')\n"
            "ImportError: no thing\n"
        )
        # The test method lives far from the module-level import line (5).
        test_file = _synthetic_test_file(
            tmp_path,
            name="test_callbacks.py",
            slice_lines=["    def test_early_stopping(self):", "        assert True"],
            fail_line=200,
            class_header="class TestCallbacks:",
        )
        out = build_python_trigger_clean(
            trigger_trace=trace,
            test_file=test_file,
            test_qualname="test_early_stopping",
            sut_packages=frozenset({"keras"}),
        )
        assert out == ""

    def test_no_matching_block_is_blank(self, tmp_path: Path) -> None:
        """A captured-pass run yields no failing block for the trigger → blank."""
        out = build_python_trigger_clean(
            trigger_trace="--- other/test_thing.py::test_other\nTraceback (most recent call last):\n"
            '  File "other/test_thing.py", line 3, in test_other\n    assert False\n'
            "AssertionError\n",
            test_file=tmp_path / "missing.py",
            test_qualname="test_something",
            sut_packages=frozenset({"black"}),
        )
        assert out == ""

    def test_fail_line_outside_def_span_is_blank(self, tmp_path: Path) -> None:
        """A test-file frame outside the trigger test's def span routes to blank."""
        test_file = tmp_path / "test_mod.py"
        test_file.write_text(
            "def test_verify_collections_no_version():\n    pass\n", encoding="utf-8"
        )
        trace = (
            "--- test/test_mod.py::test_verify_collections_no_version\n"
            "Traceback (most recent call last):\n"
            '  File "test/test_mod.py", line 1169, in test_verify_collections_no_version\n'
            "    something()\n"
            "TypeError: boom\n"
        )
        out = build_python_trigger_clean(
            trigger_trace=trace,
            test_file=test_file,
            test_qualname="test_verify_collections_no_version",
            sut_packages=frozenset({"ansible"}),
        )
        assert out == ""


class _StubRepo:
    """Minimal BugsInPyRepo surface for save_python_trigger_clean."""

    def __init__(
        self, tmp_path: Path, *, trigger_trace: str, test_file_rel: str, trigger: str
    ) -> None:
        self.project = "proj"
        self.bug_id = 1
        self.output_dir = tmp_path / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trigger_trace.txt").write_text(trigger_trace, encoding="utf-8")
        self._src_dir = tmp_path / "src"
        self._test_file_rel = test_file_rel
        self._trigger = trigger

    def _bug_info_value(self, key: str) -> str | None:
        return self._test_file_rel if key == "test_file" else None

    def get_trigger_tests(self) -> list[str]:
        return [self._trigger]

    def get_src_tests_dir(self) -> Path:
        return self._src_dir


class TestSavePythonTriggerClean:
    def test_writes_mirror_with_trailing_newline(self, tmp_path: Path) -> None:
        expected = _read("ansible_1_expected.txt")
        slice_lines = expected.split("\n" + TRANSITION_LINE + "\n", 1)[0].splitlines()
        decl_start = 1169 - len(slice_lines) + 1
        rel = "test/units/galaxy/test_collection.py"
        test_file = tmp_path / "src" / rel
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "\n".join(["# pad"] * (decl_start - 1) + slice_lines + ["        pass"]) + "\n",
            encoding="utf-8",
        )
        repo = _StubRepo(
            tmp_path,
            trigger_trace=_read("ansible_1_trigger_trace.txt"),
            test_file_rel=rel,
            trigger="test.units.galaxy.test_collection::test_verify_collections_no_version",
        )
        (repo.output_dir / "method_signatures.csv").write_text(
            "corpus_id;path;startLine;endLine\nansible.galaxy.collection$from_name(a);x;1;2\n",
            encoding="utf-8",
        )
        out_path = save_python_trigger_clean(repo, skip_existing=False)  # type: ignore[arg-type]
        assert out_path.read_text(encoding="utf-8") == expected  # incl. trailing newline

    def test_blank_when_not_mirrorable(self, tmp_path: Path) -> None:
        repo = _StubRepo(
            tmp_path,
            trigger_trace="",  # FauxPy run captured no failing trace
            test_file_rel="tests/keras/test_callbacks.py",
            trigger="tests.keras.test_callbacks::test_x",
        )
        out_path = save_python_trigger_clean(repo, skip_existing=False)  # type: ignore[arg-type]
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == ""  # blank file, no crash
