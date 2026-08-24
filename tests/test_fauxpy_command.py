"""Regression guards for the FauxPy in-container pytest collection targets.

Two failures previously produced empty coverage across whole projects:

1. A ``;``-separated ``test_file`` (fastapi/11, fastapi/16) injected raw into ``bash -lc``, where
   ``;`` is a command separator — pytest ran on only the first file (without the FauxPy flags) and
   the trailing path was executed as a bogus command, producing no report.
2. ``bug.info``'s ``test_file`` lists non-test *data fixtures* alongside the real test module
   (black: ``tests/data/tupleassign.py;tests/test_black.py``, ``tests/python2.py``). Handing a
   fixture to pytest aborts collection (``SyntaxError`` on a Python-2 sample, relative-import
   error) -> 0 tests -> empty coverage. The collection targets must come from the failing-test
   node ids (the real test modules, matching ``run_test.sh``), not the raw ``test_file``.
"""

from __future__ import annotations

import pytest

from src.extraction.bip_fauxpy_config import (
    FAUXPY_PROJECT_CONFIG,
    env_hook_commands,
    exclude_arg,
    fixture_copies_for,
    host_shell_commands,
)
from src.extraction.fauxpy import _pytest_collection_targets


@pytest.mark.parametrize(
    ("failing_nodeids", "test_file", "expected"),
    [
        # Single trigger -> its module only.
        (["tests/test_black.py::BlackTestCase::test_fmtonoff4"], "", "tests/test_black.py"),
        # black-7: test_file lists a data fixture first; it must be dropped, only the test module kept.
        (
            ["tests/test_black.py::BlackTestCase::test_tuple_assign"],
            "tests/data/tupleassign.py;tests/test_black.py",
            "tests/test_black.py",
        ),
        # black-23: tests/python2.py (literal py2) must not reach pytest.
        (
            ["tests/test_black.py::BlackTestCase::test_python2"],
            "tests/python2.py;tests/test_black.py",
            "tests/test_black.py",
        ),
        # Multiple triggers in the same module -> deduped to one target.
        (
            [
                "tests/test_black.py::BlackTestCase::test_a",
                "tests/test_black.py::BlackTestCase::test_b",
            ],
            "tests/test_black.py",
            "tests/test_black.py",
        ),
        # Triggers spanning two real modules -> both collected, order preserved.
        (
            ["tests/test_x.py::test_a", "tests/test_y.py::test_b"],
            "tests/test_x.py;tests/test_y.py",
            "tests/test_x.py tests/test_y.py",
        ),
        # Fallback: no node ids -> ;-split test_file (fastapi multi-file).
        (
            [],
            "tests/test_union_body.py;tests/test_union_inherited_body.py",
            "tests/test_union_body.py tests/test_union_inherited_body.py",
        ),
    ],
)
def test_collection_targets(failing_nodeids: list[str], test_file: str, expected: str) -> None:
    targets = _pytest_collection_targets(failing_nodeids, test_file)
    assert targets == expected
    assert ";" not in targets  # no shell command-separator survives into the bash script


def test_collection_targets_excludes_data_fixtures() -> None:
    """The whole point: a non-test fixture in test_file never becomes a pytest target."""
    targets = _pytest_collection_targets(
        ["tests/test_black.py::BlackTestCase::test_comments7"],
        "tests/data/comments7.py;tests/test_black.py",
    )
    assert "tests/data/comments7.py" not in targets
    assert targets == "tests/test_black.py"


# ---------------------------------------------------------------------------
# Per-project --exclude + pre-FauxPy hook command rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("project", "expected"),
    [
        ("black", "[env,tests]"),
        ("pandas", "[pandas/tests]"),
        ("spacy", "[spacy/tests,env]"),
        ("fastapi", "[]"),  # no exclude -> empty bracket list
    ],
)
def test_exclude_arg(project: str, expected: str) -> None:
    assert exclude_arg(FAUXPY_PROJECT_CONFIG[project]) == expected
    assert ";" not in exclude_arg(FAUXPY_PROJECT_CONFIG[project])  # no shell separator survives


def test_exclude_arg_none_is_empty_brackets() -> None:
    assert exclude_arg(None) == "[]"


@pytest.mark.parametrize("project", ["luigi", "sanic"])
def test_env_hooks_uninstall_pytest_sugar(project: str) -> None:
    # Both projects uninstall pytest-sugar first; luigi additionally installs its deps after.
    cmds = env_hook_commands(FAUXPY_PROJECT_CONFIG[project], "ENV")
    assert cmds[0] == "conda run -n ENV pip uninstall -y pytest-sugar || true"


def test_env_hooks_empty_for_plain_project() -> None:
    assert env_hook_commands(FAUXPY_PROJECT_CONFIG["youtube-dl"], "ENV") == []


def test_env_hooks_black_generates_version_file() -> None:
    """black is never installed; generate its setuptools_scm _black_version.py before pytest."""
    assert env_hook_commands(FAUXPY_PROJECT_CONFIG["black"], "ENV") == [
        "conda run -n ENV python setup.py --version || true"
    ]


def test_host_hooks_comment_pytest_ini() -> None:
    assert host_shell_commands(FAUXPY_PROJECT_CONFIG["httpie"]) == [
        "[ -f pytest.ini ] && sed -i '\\|--tb=native|{/^[[:space:]]*#/!s/^/# /}' pytest.ini || true"
    ]
    assert host_shell_commands(FAUXPY_PROJECT_CONFIG["keras"]) == [
        "[ -f pytest.ini ] && sed -i '\\|-n 2|{/^[[:space:]]*#/!s/^/# /}' pytest.ini || true"
    ]


def test_host_hooks_remove_paths() -> None:
    assert host_shell_commands(FAUXPY_PROJECT_CONFIG["fastapi"]) == ["rm -rf tests/test_tutorial"]


def test_host_hooks_tqdm_rename() -> None:
    assert host_shell_commands(FAUXPY_PROJECT_CONFIG["tqdm"]) == [
        'for f in tqdm/tests/test*; do n="${f/tests_/test_}"; [ "$n" != "$f" ] && cp "$f" "$n"; done'
    ]


def test_host_hooks_exclude_fixture_copies() -> None:
    """Fixture copies are handled by the runner (host staging), not host_shell_commands.

    thefuck has both a ``replace_in_files`` hook (the get_marker port, which *does* belong in
    host_shell_commands) and ``copy_fixtures`` (which does not) — so its output is exactly the
    replace command, with no ``cp`` for the fixture conftest.
    """
    assert host_shell_commands(FAUXPY_PROJECT_CONFIG["thefuck"]) == [
        "[ -f tests/conftest.py ] && sed -i 's|get_marker(|get_closest_marker(|g' tests/conftest.py || true"
    ]
    assert host_shell_commands(FAUXPY_PROJECT_CONFIG["black"]) == []


def test_fixture_copies_are_per_bug() -> None:
    thefuck = FAUXPY_PROJECT_CONFIG["thefuck"]
    applied = fixture_copies_for(thefuck, 4)
    assert [fc.dest for fc in applied] == ["conftest.py"]
    # The original tests/conftest.py must be removed first (both define --enable-functional).
    assert applied[0].remove_before == ("tests/conftest.py",)
    assert fixture_copies_for(thefuck, 5) == []  # bug 5 has no fixture -> no swap
    assert fixture_copies_for(FAUXPY_PROJECT_CONFIG["black"], 1) == []
