"""Tests for the live trace-capture pytest plugin (src.extraction.cefl_trace_plugin).

The plugin is the Python analogue of Defects4J's ``Formatter.java``: attached to a live pytest run
(``-p cefl_trace_plugin``) it writes, for each failing test / collection error, a ``--- nodeid``
header + a real ``Traceback`` block to ``CEFL_TRACE_OUT``, with frames made module-relative.

These tests run an *inner* pytest in a subprocess (mirroring the in-container FauxPy invocation: the
plugin is copied next to the tests and loaded by bare module name with its dir on ``PYTHONPATH``),
then assert the emitted ``trigger_trace.txt`` shape.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = REPO_ROOT / "fl_methods" / "src" / "extraction" / "cefl_trace_plugin.py"


def _run_inner_pytest(work_dir: Path, test_file: Path) -> Path:
    """Run pytest on *test_file* with the plugin loaded; return the capture-file path."""
    shutil.copy2(PLUGIN_SRC, work_dir / "cefl_trace_plugin.py")
    out_file = work_dir / "trigger_trace.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["CEFL_TRACE_OUT"] = str(out_file)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-p",
            "cefl_trace_plugin",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    return out_file


def test_failing_test_is_captured(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "def helper():\n    raise ValueError('boom')\n\n\ndef test_x():\n    helper()\n",
        encoding="utf-8",
    )
    out = _run_inner_pytest(tmp_path, test_file)
    text = out.read_text(encoding="utf-8")

    lines = text.splitlines()
    assert lines[0].startswith("--- ")
    assert "test_sample.py::test_x" in lines[0]
    assert "Traceback (most recent call last):" in text
    # frame paths are module-relative (no absolute tmp prefix), test + helper frames present
    assert 'File "test_sample.py", line 6, in test_x' in text
    assert "in helper" in text
    assert str(tmp_path) not in text  # no absolute paths leaked
    assert text.rstrip().endswith("ValueError: boom")


def test_collection_error_is_captured(tmp_path: Path) -> None:
    test_file = tmp_path / "test_broken.py"
    test_file.write_text(
        "import this_module_truly_does_not_exist_xyz\n\n\ndef test_y():\n    pass\n",
        encoding="utf-8",
    )
    out = _run_inner_pytest(tmp_path, test_file)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("--- ")  # collection errors are captured too
    assert "Traceback (most recent call last):" in text
    assert "ModuleNotFoundError" in text or "ImportError" in text


def test_disabled_without_env(tmp_path: Path) -> None:
    """With CEFL_TRACE_OUT unset the plugin no-ops (no file written, run unaffected)."""
    test_file = tmp_path / "test_pass.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    shutil.copy2(PLUGIN_SRC, tmp_path / "cefl_trace_plugin.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("CEFL_TRACE_OUT", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-p", "cefl_trace_plugin", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert not (tmp_path / "trigger_trace.txt").exists()
