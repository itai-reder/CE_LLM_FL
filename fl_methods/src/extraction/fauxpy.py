"""FauxPy coverage bridge: run pytest-FauxPy for a BugsInPy bug and reduce its
SQLite output to the canonical CEFL coverage schema.

Strategy (FauxPy DB bridge):

    run pytest-FauxPy (SBFL / statement) in the bug's conda env
      -> FauxPyReport_*/{fauxpy.db, Scores_*.csv}  (tier-1 raw, isolated under FauxPy/raw/)
      -> read fauxpy.db (TestCase outcomes, ExecutionTrace matrix, Score Ochiai)
      -> emit, under FauxPy/coverage + FauxPy/reports (config-resolved per benchmark):
           spectra.csv / matrix.txt / tests.csv      (same schema as GZoltar's)
           ochiai.ranking.csv                         (FauxPy's precomputed Ochiai)

Only entities (``<path>::<line>``) that map to a corpus method are kept, so every
spectra/ochiai id is in the canonical ``<module>$<qualname>(params):line`` form and
reduces to ``(class_fqn_dotted, line)`` for the statement→method join.
Module-level lines (no enclosing method) are excluded — they do not aggregate to
a method entity.

``fauxpy.db`` schema (reverse-engineered on youtube-dl/2):

    TestCase(Rowid, TestName, Type['passed'|'failed'], Target[0|1])
    ExecutionTrace(Rowid, TestName, Entity)            # long-form coverage matrix
    Score(Rowid, Entity, Ef, Ep, Nf, Np, Tarantula, Ochiai, Dstar)
    TestName = '<path>::<defline>::<Class>.<method>'   (module-level: '<path>::<defline>::<func>')
    Entity   = '<abs_src_path>::<line>'
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import (
    MATRIX_FILE,
    OCHIAI_RANKING_FILE,
    SPECTRA_FILE,
    TESTS_FILE,
    get_coverage_dir,
    get_fauxpy_raw_dir,
    get_ochiai_ranking_dir,
)
from src.common.method_entity import MethodEntity, load_method_entities
from src.extraction.bip_fauxpy_config import (
    FAUXPY_PROJECT_CONFIG,
    FixtureCopy,
    env_hook_commands,
    exclude_arg,
    fixture_copies_for,
    host_shell_commands,
)
from src.extraction.bugsinpy import _path_to_module, _to_bip_container_path, run_bip
from src.extraction.trigger_test_py import save_python_trigger_clean

if TYPE_CHECKING:
    from src.extraction.bugsinpy import BugsInPyRepo

# Live trace-capture plugin (the Formatter.java analogue) — copied into the bug's mounted
# processed dir and loaded into the FauxPy pytest run via ``-p cefl_trace_plugin``.
_TRACE_PLUGIN_SRC = Path(__file__).with_name("cefl_trace_plugin.py")
_TRACE_PLUGIN_MODULE = "cefl_trace_plugin"
_TRACE_FILE = "trigger_trace.txt"

# pytest-FauxPy requires ``coverage>=6.2``, which dropped support for Python <3.6. Bugs whose
# per-bug conda env is older than this (only cookiecutter/4 @ 3.5.6 in the BugsInPy corpus) cannot
# run FauxPy at all, so the coverage step is skipped-and-warned rather than hard-failed.
_FAUXPY_MIN_PYTHON = (3, 6)


def fauxpy_supported(repo: BugsInPyRepo) -> bool:
    """True when the bug's conda-env Python is new enough for pytest-FauxPy (>= 3.6).

    Reads ``python_version`` from ``bug.info``; an unparseable/absent value returns ``True`` (attempt
    rather than silently skip on a parse miss).
    """
    m = re.match(r"\s*(\d+)\.(\d+)", repo._bug_info_value("python_version") or "")
    if not m:
        return True
    return (int(m.group(1)), int(m.group(2))) >= _FAUXPY_MIN_PYTHON


logger = logging.getLogger(__name__)

_DATASET = "bugsinpy"


# ---------------------------------------------------------------------------
# Test-id / path canonicalisation
# ---------------------------------------------------------------------------


def _canonical_test_name(fauxpy_test_name: str) -> str:
    """Convert a FauxPy ``TestName`` to the canonical CEFL test id.

    ``test/foo.py::79::Cls.method`` -> ``test.foo$Cls#method``;
    ``test/foo.py::79::func`` -> ``test.foo#func``.
    """
    parts = fauxpy_test_name.split("::")
    module = _path_to_module(parts[0])
    qual = parts[-1]
    if "." in qual:
        cls, method = qual.rsplit(".", 1)
        return f"{module}${cls}#{method}"
    return f"{module}#{qual}"


def _to_pytest_nodeid(trigger_id: str) -> str:
    """Convert a canonical trigger id to a pytest node id for ``--failing-list``.

    ``test.foo$Cls::method`` -> ``test/foo.py::Cls::method``;
    ``test.foo::func`` -> ``test/foo.py::func``.
    """
    if "$" in trigger_id:
        module_part, rest = trigger_id.split("$", 1)
    else:
        module_part, _, func = trigger_id.partition("::")
        rest = func
    path = module_part.replace(".", "/") + ".py"
    return f"{path}::{rest}"


def _build_path_line_index(
    entities: list[MethodEntity],
) -> dict[tuple[str, int], list[MethodEntity]]:
    """Build a ``(src-relative path, line) -> [MethodEntity]`` index."""
    index: dict[tuple[str, int], list[MethodEntity]] = defaultdict(list)
    for e in entities:
        for line in range(e.start_line, e.end_line + 1):
            index[(e.path, line)].append(e)
    return index


def _rel_to_src(abs_path: str, container_src: str) -> str | None:
    """Return *abs_path* relative to the container source root, or ``None``."""
    prefix = container_src.rstrip("/") + "/"
    if abs_path.startswith(prefix):
        return abs_path[len(prefix) :]
    return None


def _fauxpy_src_arg(repo: BugsInPyRepo) -> str:
    """Choose FauxPy's ``--src``: the SUT package path relative to the source tree.

    Combines the import root (``get_src_class_dir``) with the top-level package of
    the modified files, so e.g. ``youtube_dl`` (flat) and ``lib/ansible`` (lib layout)
    both localise just the SUT package. Falls back to the import root, then ``.``.
    """
    import_root = repo.get_src_class_dir()
    try:
        ir_rel = import_root.relative_to(repo.source_tree_dir)
    except ValueError:
        return "."

    modified = repo.get_modified_classes()
    if modified:
        mod = Path(modified[0])
        try:
            mod_rel = mod.relative_to(ir_rel) if ir_rel != Path(".") else mod
        except ValueError:
            mod_rel = mod
        if mod_rel.parts:
            package = mod_rel.parts[0]
            src = ir_rel / package if ir_rel != Path(".") else Path(package)
            return src.as_posix()
    return ir_rel.as_posix() if ir_rel != Path(".") else "."


# ---------------------------------------------------------------------------
# Container run
# ---------------------------------------------------------------------------


# Requirement files commonly carrying a project's runtime + test dependencies.
# BugsInPy's per-bug requirements / setup.sh frequently omit test deps (and
# `setup.py install` under-installs install_requires), so the FauxPy env-prep
# installs whichever of these the source tree ships before collecting tests.
_REQUIREMENT_FILES = (
    "requirements.txt",
    "test_requirements.txt",
    "test-requirements.txt",
    "requirements-test.txt",
    "requirements_test.txt",
    "requirements-dev.txt",
    "dev-requirements.txt",
)


def _pytest_collection_targets(failing_nodeids: list[str], test_file: str) -> str:
    """Space-separated pytest collection targets: the test modules holding the triggering tests.

    Derive the targets from the failing-test node ids — whose file part is the real test module,
    exactly what BugsInPy's ``run_test.sh`` runs (e.g. ``python -m unittest tests.test_black...``).
    Deliberately do **not** use the raw ``bug.info`` ``test_file`` field: for many projects it also
    lists non-test *data fixtures* (black's ``tests/data/*.py`` formatting samples, ``tests/python2.py``)
    alongside the test module. Handing a fixture to pytest makes its import fail (a ``SyntaxError`` on
    a Python-2 sample, a relative-import error, ...), and a single collection error aborts the whole
    session -> 0 tests collected -> empty coverage. (It also matters that the fixture's collection
    error would otherwise pollute ``trigger_trace.txt`` instead of the real test failure.)

    ``test_file`` is used only as a ``;``-split fallback when no node ids are available (e.g. fastapi
    multi-file: ``tests/a.py;tests/b.py``) — a literal ``;`` inside ``bash -lc`` is a command
    separator, so it must be collapsed to space-separated args there too.
    """
    targets: list[str] = []
    seen: set[str] = set()
    for nodeid in failing_nodeids:
        path = nodeid.split("::", 1)[0].strip()
        if path and path not in seen:
            seen.add(path)
            targets.append(path)
    if targets:
        return " ".join(targets)
    return " ".join(part.strip() for part in test_file.split(";") if part.strip())


def _stage_fixture(repo: BugsInPyRepo, fc: FixtureCopy) -> str | None:
    """Copy a per-bug fixture file into the mounted checkout; return its container path.

    The authors' fixtures (thefuck conftest, cookiecutter test_requirements) live outside the
    ``/workspace`` mount, so they cannot be ``cp``'d in-container directly. Stage them host-side
    under the (host-writable, mounted) checkout dir so the emitted in-container ``cp`` can reach
    them. Returns ``None`` (skipping the copy) when the reference fixture is absent.
    """
    src = fc.source_for(repo.bug_id)
    if not src.exists():
        logger.warning(
            "FauxPy fixture missing for %s-%s, skipping: %s", repo.project, repo.bug_id, src
        )
        return None
    staged = repo.repo_dir / "_fauxpy_fixtures" / fc.dest
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, staged)
    return _to_bip_container_path(staged)


def _run_fauxpy(repo: BugsInPyRepo) -> None:
    """Run pytest-FauxPy (SBFL/statement) and copy the report into FauxPy/raw/."""
    env = repo.conda_env_name()
    cfg = FAUXPY_PROJECT_CONFIG.get(repo.project)
    src_arg = cfg.src if cfg else _fauxpy_src_arg(repo)
    exclude = exclude_arg(cfg)
    test_file = repo._bug_info_value("test_file") or ""
    if not test_file:
        raise RuntimeError(f"No test_file in bug.info for {repo.project}-{repo.bug_id}.")

    failing = [_to_pytest_nodeid(t) for t in repo.get_trigger_tests()]
    failing_list = "[" + ",".join(failing) + "]"
    test_files = _pytest_collection_targets(failing, test_file)

    # Best-effort install of the project's runtime + test requirements so the test
    # module imports cleanly under FauxPy (see _REQUIREMENT_FILES).
    req_files = [name for name in _REQUIREMENT_FILES if (repo.source_tree_dir / name).exists()]
    dep_installs = "".join(
        f"conda run -n {env} pip install -q -r {name} || true ; " for name in req_files
    )

    raw_dir = get_fauxpy_raw_dir(repo.project, repo.bug_id)
    raw_dir.mkdir(parents=True, exist_ok=True)  # host-owned parent for the root-owned copy

    src_tree_c = _to_bip_container_path(repo.source_tree_dir)
    repo_dir_c = _to_bip_container_path(repo.repo_dir)
    raw_dir_c = _to_bip_container_path(raw_dir)

    # Live trace capture: copy the cefl_trace plugin into the (gitignored) checkout dir so it is
    # importable in-container (``-p cefl_trace_plugin`` via PYTHONPATH) without polluting the
    # tracked processed/ tree — it is removed with the checkout. The capture itself
    # (trigger_trace.txt, a tracked artifact) is written into the processed dir via CEFL_TRACE_OUT.
    out_dir = repo.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_TRACE_PLUGIN_SRC, repo.repo_dir / f"{_TRACE_PLUGIN_MODULE}.py")
    plugin_dir_c = _to_bip_container_path(repo.repo_dir)
    trace_out_c = _to_bip_container_path(out_dir / _TRACE_FILE)

    # PYTHONPATH carries only the trace-plugin dir (so `-p cefl_trace_plugin` imports) plus any
    # per-project extra roots (e.g. ansible's lib/). The SUT source is deliberately NOT added: the
    # per-bug setup.sh installs the package into the env, and forcing raw source ahead of that
    # install shadows setuptools_scm-generated modules (black's _black_version, matplotlib's
    # _version) -> ImportError. Mirrors the authors (cd into the tree, rely on the install + cwd).
    extra_pp = [
        _to_bip_container_path(repo.source_tree_dir / p)
        for p in (cfg.extra_pythonpath if cfg else ())
    ]
    pythonpath = ":".join([*extra_pp, plugin_dir_c])

    # Per-project pre-FauxPy workarounds ported from faux-in-py.sh: env mutations (pytest-sugar
    # uninstall), source-tree mutations (pytest.ini edits, file removals, tqdm rename), and per-bug
    # fixture copies (thefuck conftest, cookiecutter test_requirements). All run in-container from
    # cwd = the source tree; fixture content is staged host-side into the mounted checkout first.
    hook_cmds: list[str] = []
    if cfg:
        hook_cmds.extend(env_hook_commands(cfg, env))
        hook_cmds.extend(host_shell_commands(cfg))
        for fc in fixture_copies_for(cfg, repo.bug_id):
            staged_c = _stage_fixture(repo, fc)
            if staged_c is not None:
                hook_cmds.extend(f"rm -f {path}" for path in fc.remove_before)
                hook_cmds.append(f"cp {staged_c} {fc.dest}")
    hook_block = "".join(f"{cmd} ; " for cmd in hook_cmds)

    # Clear any stale FauxPyReport_* (from a prior run) so exactly one fresh report
    # lands in FauxPy/raw/. FauxPy writes the report under the repo dir (the parent
    # of the source tree); copy it into FauxPy/raw/, then remove the originals.
    script = (
        f"cd {src_tree_c} && "
        f"rm -rf {raw_dir_c}/FauxPyReport* ; rm -f {trace_out_c} ; "
        f"find {repo_dir_c} -maxdepth 2 -type d -name 'FauxPyReport*' -exec rm -rf {{}} + ; "
        # --use-pep517 forces an isolated PEP 517 build for FauxPy: pip builds it in a fresh env
        # with a self-consistent setuptools instead of the bug's conda env, whose setuptools is
        # often broken by the SUT's own deps (e.g. fastapi drags importlib_metadata down to 1.6.1,
        # which the env's setuptools 75 cannot import -> 'no attribute EntryPoints' -> egg_info
        # fails -> the &&-guarded pytest never runs -> "No FauxPy report DB"). Isolation sidesteps
        # the env entirely without perturbing the SUT's runtime dependency versions.
        f"conda run -n {env} pip install -q --use-pep517 /opt/pytest-FauxPy && "
        # setuptools>=71 re-vendored typeguard, whose plugin probe calls importlib.metadata
        # entry_points(group=...) — unsupported on Python 3.8, which aborts pytest's pytest11
        # entry-point loading before any test runs (no report DB). Freshly-created conda envs pull
        # setuptools 75; pin below 71 so the FauxPy pytest run loads (no-op on older reused envs).
        f"conda run -n {env} pip install -q 'setuptools<71' || true ; "
        f"{dep_installs}"
        f"{hook_block}"
        f"conda run -n {env} env PYTHONPATH={pythonpath} "
        f"CEFL_TRACE_OUT={trace_out_c} python -m pytest {test_files} "
        f"-p {_TRACE_PLUGIN_MODULE} "
        f"--src {src_arg} --exclude '{exclude}' --granularity statement --family sbfl "
        f"--failing-list '{failing_list}' ; "
        f"find {repo_dir_c} -maxdepth 2 -type d -name 'FauxPyReport*' "
        f"-exec cp -rp {{}} {raw_dir_c}/ \\; ; "
        f"find {repo_dir_c} -maxdepth 2 -type d -name 'FauxPyReport*' -exec rm -rf {{}} +"
    )
    logger.info("Running FauxPy SBFL for %s-%s (env %s)", repo.project, repo.bug_id, env)
    # FauxPy exits non-zero because the trigger test fails; that is expected. The in-container
    # pytest/FauxPy stdout+stderr is the only place the *root cause* of an empty/failed report
    # surfaces (egg_info failures, import errors, 0-collected, plugin-load aborts), so persist it
    # host-side into the already-host-owned raw dir for retroactive analysis (`check=False` and the
    # cefl_trace plugin are untouched; run_bip already captured the streams).
    result = run_bip(["bash", "-lc", script], check=False)
    (raw_dir / "fauxpy_run.log").write_text(
        f"$ {script}\n\n"
        f"----- exit code: {result.returncode} -----\n"
        f"----- STDOUT -----\n{result.stdout or ''}\n"
        f"----- STDERR -----\n{result.stderr or ''}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# DB -> canonical schema bridge
# ---------------------------------------------------------------------------


def _find_report_db(repo: BugsInPyRepo) -> Path | None:
    """Return the ``fauxpy.db`` inside FauxPy/raw/, or ``None`` if absent."""
    raw_dir = get_fauxpy_raw_dir(repo.project, repo.bug_id)
    # Names are timestamped; prefer the most recent report that has a DB.
    for report in sorted(raw_dir.glob("FauxPyReport*"), reverse=True):
        db = report / "fauxpy.db"
        if db.exists():
            return db
    return None


def bridge_fauxpy_report(repo: BugsInPyRepo, *, skip_existing: bool = True) -> None:
    """Reduce the raw FauxPy report to the canonical coverage schema.

    Writes spectra/matrix/tests under ``FauxPy/coverage`` and ``ochiai.ranking.csv``
    under ``FauxPy/reports``, plus ``all_tests.txt`` / ``relevant_tests.txt`` (the
    suite FauxPy ran) under the processed dir.
    """
    coverage_dir = get_coverage_dir(repo.project, repo.bug_id, dataset=_DATASET)
    ranking_dir = get_ochiai_ranking_dir(repo.project, repo.bug_id, dataset=_DATASET)
    spectra_path = coverage_dir / SPECTRA_FILE
    matrix_path = coverage_dir / MATRIX_FILE
    tests_path = coverage_dir / TESTS_FILE

    if skip_existing and spectra_path.exists() and matrix_path.exists() and tests_path.exists():
        logger.info(
            "FauxPy coverage already bridged for %s-%s, skipping.", repo.project, repo.bug_id
        )
        return

    db = _find_report_db(repo)
    if db is None:
        raise FileNotFoundError(
            f"No FauxPy report DB under {get_fauxpy_raw_dir(repo.project, repo.bug_id)}; "
            "run the FauxPy step first."
        )

    try:
        entities = load_method_entities(repo.output_dir)
    except FileNotFoundError:
        logger.warning(
            "method_signatures.csv missing in %s; coverage will be empty.", repo.output_dir
        )
        entities = []
    path_index = _build_path_line_index(entities)
    container_src = _to_bip_container_path(repo.get_src_class_dir())

    con = sqlite3.connect(str(db))
    try:
        test_rows = con.execute("SELECT TestName, Type FROM TestCase ORDER BY Rowid").fetchall()
        score_rows = con.execute("SELECT Entity, Ochiai FROM Score").fetchall()
        trace_rows = con.execute("SELECT TestName, Entity FROM ExecutionTrace").fetchall()
    finally:
        con.close()

    # Map each Score entity that falls inside a corpus method -> spectra id.
    spectra_id_of: dict[str, str] = {}
    ochiai_of: dict[str, float] = {}
    for entity, ochiai in score_rows:
        path, sep, line_s = entity.rpartition("::")
        if not sep or not line_s.isdigit():
            continue
        rel = _rel_to_src(path, container_src)
        if rel is None:
            continue
        candidates = path_index.get((rel, int(line_s)))
        if not candidates:
            continue  # module-level / unmapped line: excluded (no method entity)
        chosen = min(candidates, key=lambda e: e.end_line - e.start_line)
        spectra_id_of[entity] = f"{chosen.corpus_id}:{line_s}"
        ochiai_of[entity] = float(ochiai)

    # Deterministic column order: by (relative path, line).
    def _sort_key(ent: str) -> tuple[str, int]:
        path, _, line_s = ent.rpartition("::")
        return (path, int(line_s))

    columns = sorted(spectra_id_of, key=_sort_key)
    col_of = {ent: i for i, ent in enumerate(columns)}

    covered: dict[str, set[int]] = defaultdict(set)
    for test_name, entity in trace_rows:
        col = col_of.get(entity)
        if col is not None:
            covered[test_name].add(col)

    tests = [
        (_canonical_test_name(tn), "FAIL" if ty == "failed" else "PASS", tn) for tn, ty in test_rows
    ]

    coverage_dir.mkdir(parents=True, exist_ok=True)
    ranking_dir.mkdir(parents=True, exist_ok=True)

    with spectra_path.open("w", encoding="utf-8") as fh:
        fh.write("name\n")
        for ent in columns:
            fh.write(spectra_id_of[ent] + "\n")

    n_cols = len(columns)
    with matrix_path.open("w", encoding="utf-8") as fh:
        for _name, outcome, raw_name in tests:
            cov = covered.get(raw_name, set())
            bits = " ".join("1" if i in cov else "0" for i in range(n_cols))
            marker = "-" if outcome == "FAIL" else "+"
            fh.write(f"{bits} {marker}\n" if bits else f"{marker}\n")

    with tests_path.open("w", encoding="utf-8") as fh:
        fh.write("name,outcome,runtime,stacktrace\n")
        for name, outcome, _raw in tests:
            fh.write(f"{name},{outcome},0,\n")

    with (ranking_dir / OCHIAI_RANKING_FILE).open("w", encoding="utf-8") as fh:
        fh.write("name;suspiciousness_value\n")
        for ent in sorted(spectra_id_of, key=lambda e: (-ochiai_of[e], spectra_id_of[e])):
            fh.write(f"{spectra_id_of[ent]};{ochiai_of[ent]}\n")

    test_names = [name for name, _o, _r in tests]
    (repo.output_dir / "all_tests.txt").write_text(
        "\n".join(test_names) + ("\n" if test_names else ""), encoding="utf-8"
    )
    (repo.output_dir / "relevant_tests.txt").write_text(
        "\n".join(test_names) + ("\n" if test_names else ""), encoding="utf-8"
    )

    logger.info(
        "FauxPy bridge for %s-%s: %d statements, %d tests -> %s",
        repo.project,
        repo.bug_id,
        n_cols,
        len(tests),
        coverage_dir,
    )


def run_fauxpy_pipeline(repo: BugsInPyRepo, *, skip_existing: bool = True) -> None:
    """Produce §5.4 coverage for a BugsInPy bug: run FauxPy then bridge its output.

    The FauxPy pytest run also captures the live failing-test traceback (``trigger_trace.txt``,
    via the ``cefl_trace`` plugin), from which ``trigger_test_clean.txt`` is built — so the
    cleaned trigger artifact is produced here, not in the ``tests`` step.

    When the bug's env Python is too old for FauxPy (:func:`fauxpy_supported` is ``False``), the
    coverage run is skipped-and-warned: no coverage / ``all_tests`` / ``relevant_tests`` are
    produced and ``trigger_test_clean.txt`` is written blank (the caller also skips the
    coverage-dependent ``faults_first`` and relaxes validation).
    """
    if not fauxpy_supported(repo):
        logger.warning(
            "%s-%s: skipping FauxPy — Python %r is below the FauxPy floor %d.%d (pytest-FauxPy "
            "needs coverage>=6.2, unavailable for <3.6); coverage/all_tests/relevant_tests/"
            "faults_first absent and trigger_test_clean.txt blank for this bug.",
            repo.project,
            repo.bug_id,
            repo._bug_info_value("python_version"),
            _FAUXPY_MIN_PYTHON[0],
            _FAUXPY_MIN_PYTHON[1],
        )
        save_python_trigger_clean(repo, skip_existing=skip_existing)
        return

    coverage_dir = get_coverage_dir(repo.project, repo.bug_id, dataset=_DATASET)
    required = [coverage_dir / SPECTRA_FILE, coverage_dir / MATRIX_FILE, coverage_dir / TESTS_FILE]
    if skip_existing and all(p.exists() for p in required):
        logger.info(
            "FauxPy coverage already present for %s-%s, skipping.", repo.project, repo.bug_id
        )
    else:
        if not skip_existing or _find_report_db(repo) is None:
            _run_fauxpy(repo)
        bridge_fauxpy_report(repo, skip_existing=skip_existing)

    # Build the cleaned trigger artifact from the live-captured trace (cefl_trace plugin).
    save_python_trigger_clean(repo, skip_existing=skip_existing)
