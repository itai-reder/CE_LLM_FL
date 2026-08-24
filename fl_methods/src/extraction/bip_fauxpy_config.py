"""Per-project FauxPy invocation config for BugsInPy, mirroring the authors' reference.

The original FauxPy replication package (``utils/docker/bugsinpy/fauxpy-experiments``) pins a
*constant per-project* ``--src`` / ``--exclude`` and applies per-project pre-FauxPy workarounds.
This module encodes both so our extraction faithfully reproduces the authors' setup instead of
deriving ``--src`` heuristically and skipping ``--exclude`` entirely.

Sources of truth (read-only, in-repo):
  - ``.../bash_script_generator/info/subject_info.csv`` -> ``src`` (TARGET_DIR), ``exclude`` (EXCLUDE)
  - ``.../faux-in-py.sh``                               -> per-project env/host hooks
  - ``.../bash_script_generator/data_files/fixes/subjects/`` -> fixture files (conftest, requirements)

13 projects have authoritative values from ``subject_info.csv``; the 4 BugsInPy projects absent
from the reference (ansible, matplotlib, scrapy, PySnooper) get values *derived* from their package
layout (commented per entry). ``--src`` / ``--exclude`` are resolved by FauxPy relative to the pytest
cwd, which :func:`src.extraction.fauxpy._run_fauxpy` sets to the bug's source tree.

Hooks split by where they must run:
  - ``FauxPyEnvHooks`` mutate the conda env (e.g. ``pip uninstall pytest-sugar``) -> run in-container.
  - ``FauxPyHostHooks`` mutate the checked-out source tree (pytest.ini edits, file removals, fixture
    copies, the tqdm test-module rename). The checkout is root-owned, so these also run in-container;
    fixture *content* is staged host-side into the mounted checkout first (see ``fauxpy.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import BIP_WRAPPER

# Parallelism for C-extension builds (pandas/spacy). Default 4 is a good throughput/contention
# balance across 5 lanes on a 16-core host. pandas' tslibs occasionally hits a parallel-build race
# ("undefined symbol: days_per_month_table") that produces an inconsistent .so; set
# CEFL_BIP_BUILD_JOBS=1 to re-run such stragglers with a race-free serial build.
_BUILD_JOBS = os.environ.get("CEFL_BIP_BUILD_JOBS", "4")

# Root of the authors' per-bug fixture files (thefuck conftest.py, cookiecutter test_requirements.txt).
FAUXPY_FIXTURES_ROOT = (
    BIP_WRAPPER.parent
    / "fauxpy-experiments"
    / "bash_script_generator"
    / "data_files"
    / "fixes"
    / "subjects"
)

# Authors' per-bug CSV (TARGET_DIR/EXCLUDE per project) — consumed by the config cross-check test.
SUBJECT_INFO_CSV = (
    BIP_WRAPPER.parent
    / "fauxpy-experiments"
    / "bash_script_generator"
    / "info"
    / "subject_info.csv"
)


def _fixture_bug_ids(project: str) -> frozenset[int]:
    """Bug ids that have an on-disk fixture dir ``<project>/B<id>/`` under :data:`FAUXPY_FIXTURES_ROOT`.

    Derived from disk (not a hardcoded list) so the swap set stays in sync with the shipped fixtures —
    e.g. thefuck B9 has a conftest fixture even though ``faux-in-py.sh``'s inline list omits it.
    """
    base = FAUXPY_FIXTURES_ROOT / project
    ids: set[int] = set()
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith("B") and child.name[1:].isdigit():
                ids.add(int(child.name[1:]))
    return frozenset(ids)


@dataclass(frozen=True)
class FixtureCopy:
    """A per-bug file copied from the reference clone into the bug's checkout before FauxPy runs."""

    src_template: str  # relative to FAUXPY_FIXTURES_ROOT, ``{bug}``-formatted, e.g. "thefuck/B{bug}/conftest.py"
    dest: str  # checkout-relative destination, e.g. "conftest.py"
    bugs: frozenset[int] | None = None  # None = applies to every bug; else only these ids
    remove_before: tuple[
        str, ...
    ] = ()  # checkout-relative paths to ``rm -f`` first (per-bug, coupled)

    def applies_to(self, bug_id: int) -> bool:
        return self.bugs is None or bug_id in self.bugs

    def source_for(self, bug_id: int) -> Path:
        return FAUXPY_FIXTURES_ROOT / self.src_template.format(bug=bug_id)


@dataclass(frozen=True)
class FauxPyHostHooks:
    """Mutations applied to the checked-out source tree before FauxPy runs."""

    comment_pytest_ini: tuple[str, ...] = ()  # substrings of pytest.ini lines to comment out
    comment_file_lines: tuple[
        tuple[str, str], ...
    ] = ()  # (rel_path, line-substring) to comment out
    replace_in_files: tuple[tuple[str, str, str], ...] = ()  # (rel_path, literal old, literal new)
    remove_paths: tuple[str, ...] = ()  # checkout-relative paths to ``rm -rf``
    rename_tests_glob: bool = False  # copy ``<src>/tests/tests_*`` -> ``test_*`` (tqdm)
    copy_fixtures: tuple[FixtureCopy, ...] = ()
    write_files: tuple[tuple[str, str], ...] = ()  # (checkout-rel path, content) to author verbatim


@dataclass(frozen=True)
class FauxPyEnvHooks:
    """Mutations / build steps applied via the conda env before FauxPy runs."""

    uninstall: tuple[str, ...] = ()  # packages to ``pip uninstall -y`` (e.g. pytest-sugar)
    run: tuple[
        str, ...
    ] = ()  # commands run via ``conda run -n <env>`` from cwd (e.g. a build step)


def fast_build(build_cmd: str) -> str:
    """Wrap a C-extension build command so compilation is cheap, for use in ``FauxPyEnvHooks.run``.

    The SUT only needs to be *importable* for coverage — not fast — and the same project's many
    bugs recompile near-identical sources. So two compounding, one-time-overhead speedups:

    * ``-O0`` (via ``CFLAGS``/``CXXFLAGS``): skips the expensive optimiser; the dominant cost of a
      from-scratch pandas/matplotlib/spacy build. No infrastructure — gcc is always present.
    * ``ccache`` (via the ``/usr/lib/ccache`` masquerade dir prepended to ``PATH``, a no-op when
      ccache is not installed): most translation units are byte-identical across a project's bug
      versions, so after warmup repeat compiles are cache hits. The cache lives on the shared
      ``/workspace`` mount (``data/BIP/.ccache``), so it persists across runs and all parallel lanes.

    Emitted as a ``bash -c`` so the env exports happen *inside* ``conda run`` (where ``$PATH``
    already has the env's bin) — prepending ccache there keeps ``python``/``cython`` resolving to
    the env. Degrades safely: missing ccache -> real compiler at ``-O0``; ``-O0`` honoured wherever
    the build respects ``CFLAGS`` (last ``-O`` wins over the sysconfig default).
    """
    return (
        "bash -c 'export PATH=/usr/lib/ccache:$PATH; "
        "export CCACHE_DIR=/workspace/.ccache CCACHE_MAXSIZE=8G; "
        "export CFLAGS=-O0 CXXFLAGS=-O0; "
        f"{build_cmd}'"
    )


@dataclass(frozen=True)
class FauxPyProjectConfig:
    """Per-project FauxPy invocation parameters + pre-run workarounds."""

    src: str  # --src, relative to the source tree (pytest cwd)
    exclude: tuple[str, ...] = ()  # --exclude entries, relative to --src
    extra_pythonpath: tuple[str, ...] = ()  # source-tree-relative dirs to prepend to PYTHONPATH
    host: FauxPyHostHooks = field(default_factory=FauxPyHostHooks)
    env: FauxPyEnvHooks = field(default_factory=FauxPyEnvHooks)


# Keyed by the exact BugsInPy project name (note casing: "PySnooper", "youtube-dl").
FAUXPY_PROJECT_CONFIG: dict[str, FauxPyProjectConfig] = {
    # --- Authoritative (subject_info.csv + faux-in-py.sh) -------------------------------------
    # black is never installed (its setup.sh only `touch`es a file) and `black.py` does
    # `from _black_version import version`; setuptools_scm writes `_black_version.py` only at build
    # time. Generate it in-source via `setup.py --version` (the checkout keeps `.git`) so the test
    # module imports — the authors implicitly rely on the package being importable from source.
    "black": FauxPyProjectConfig(
        src=".",
        exclude=("env", "tests"),
        env=FauxPyEnvHooks(run=("python setup.py --version",)),
    ),
    "cookiecutter": FauxPyProjectConfig(
        src="cookiecutter",
        # bug 3's conftest imports the ``past`` module (from the ``future`` package), which neither
        # setup.py nor the pinned requirements install -> `ModuleNotFoundError: No module named 'past'`
        # aborts conftest loading. Install ``future`` (ships both ``future`` and ``past``); additive
        # and harmless for the other cookiecutter bugs.
        env=FauxPyEnvHooks(run=("pip install -q future",)),
        host=FauxPyHostHooks(
            # B4 ships a generated test_requirements.txt (B4 is Python 3.5 -> skipped; parity only).
            copy_fixtures=(
                FixtureCopy(
                    "cookiecutter/B{bug}/test_requirements.txt",
                    "test_requirements.txt",
                    bugs=_fixture_bug_ids("cookiecutter"),
                ),
            ),
        ),
    ),
    "fastapi": FauxPyProjectConfig(
        src="fastapi",
        # fastapi declares deps in pyproject.toml and ships no ``requirements.txt``, so the pipeline's
        # ``_REQUIREMENT_FILES`` install misses the frozen pins BugsInPy writes to
        # ``bugsinpy_requirements.txt`` (e.g. bug 16: pydantic==0.18.2 / starlette==0.10.1 /
        # fastapi==0.55.1). Without them, installing fastapi pulls newer pydantic/starlette whose APIs
        # drifted (``cannot import name 'Schema' from pydantic``, ``'FastAPI' object has no attribute
        # 'exception_handlers'``, starlette WebSocket changes) -> collection error -> empty spectra.
        # ``--no-deps``: the pins are a frozen ``pip freeze`` snapshot whose exact versions pip's
        # modern resolver rejects as mutually inconsistent (fastapi==0.55.1 vs starlette==0.10.1 ->
        # ResolutionImpossible); installing without re-resolution downgrades the drifted packages to
        # the bug's intended versions while leaving the already-installed transitive deps intact.
        env=FauxPyEnvHooks(run=("pip install -q --no-deps -r bugsinpy_requirements.txt",)),
        # tests/test_tutorial has collection-time errors that abort the whole session.
        host=FauxPyHostHooks(remove_paths=("tests/test_tutorial",)),
    ),
    "httpie": FauxPyProjectConfig(
        src="httpie",
        # FauxPy's st-family is incompatible with --tb=native; comment it out of pytest.ini.
        host=FauxPyHostHooks(comment_pytest_ini=("--tb=native",)),
    ),
    "keras": FauxPyProjectConfig(
        src=".",
        exclude=("env", "tests"),
        # Keras needs its exact dep pins, which setup.py's ranges don't enforce: bugs collect-error on
        # drifted deps -- keras/9 `'EntryPoints' object has no attribute 'get'` (markdown reads
        # `entry_points().get(...)`, the dict API a too-new importlib-metadata dropped) and keras/21-23
        # `module 'keras_applications' has no attribute 'set_keras_submodules'` (an API later
        # keras-applications removed). The right versions differ per bug (bug 9: Keras 2.2.4 /
        # keras-applications 1.0.8; bugs 21-23: Keras 2.2.2 / keras-applications 1.0.7), so grep each
        # bug's own pins for just the two culprits out of bugsinpy_requirements.txt (installing the
        # whole file aborts atomically -- its numpy==1.19.0rc2 is yanked from PyPI) and reinstall them
        # --no-deps. Runs from the checkout cwd; ``|| true`` (added by env_hook_commands) tolerates a
        # bug whose requirements pin neither.
        env=FauxPyEnvHooks(
            run=(
                "pip install -q --no-deps $(grep -iE "
                "'^(keras-applications|importlib-metadata)==' bugsinpy_requirements.txt | tr '\\n' ' ')",
            ),
        ),
        # FauxPy is incompatible with pytest-xdist's "-n 2"; comment it out of pytest.ini.
        host=FauxPyHostHooks(comment_pytest_ini=("-n 2",)),
    ),
    # luigi: bugs 1,3,6,23 fail with missing tornado / psutil. `python setup.py install` pulls
    # luigi's core install_requires but NOT tornado (luigi.server's tornado is an optional extra,
    # and bug 1/23's trigger is the server test). Re-apply the bug's setup.sh (the pinned test deps)
    # and add tornado explicitly (luigi 2.8 era => tornado<6; 5.1.1 installs on py3.8).
    "luigi": FauxPyProjectConfig(
        src=".",
        exclude=("env", "test"),
        env=FauxPyEnvHooks(
            uninstall=("pytest-sugar",),
            run=(
                "python setup.py install",
                "pip install -q 'tornado>=4.0,<6' mock==4.0.2 psutil==5.7.0 nose==1.3.7",
            ),
        ),
    ),
    # pandas imports from the source dir (cwd), so its Cython C-extensions must exist in the
    # checkout: else `import pandas` raises "C extension ... not built ... run 'python setup.py
    # build_ext --inplace --force'". bugsinpy-compile runs the bug's setup.sh but without checking
    # its exit code, so a single silent failure leaves the build incomplete by FauxPy time. Re-run
    # the bug's own setup.sh recipe as an env hook (cwd = source tree, after the dep pass). This is
    # the 162/169-majority recipe in projects/pandas/bugs/*/setup.sh; `-j 0` parallelises the build
    # (all cores) and a fresh checkout builds every extension regardless of --force. cython/numpy
    # are already satisfied (pinned in the bug requirements, installed at compile) so the unpinned
    # installs are idempotent no-ops that just guarantee the build toolchain is present.
    "pandas": FauxPyProjectConfig(
        src="pandas",
        exclude=("pandas/tests",),
        env=FauxPyEnvHooks(
            run=(
                "pip install -q cython numpy",
                fast_build(f"python setup.py build_ext --inplace -j {_BUILD_JOBS}"),
            ),
        ),
    ),
    # sanic: bugs 4,5 fail with missing requests_async. Unresolved — sanic's [test] extra pins
    # `requests-async==0.5.0`, which has been removed from PyPI (only <=0.2.4 remain), so the test
    # deps can't be installed from the index. Would need a vendored/pinned-index workaround.
    "sanic": FauxPyProjectConfig(
        src=".",
        exclude=("env", "tests"),
        env=FauxPyEnvHooks(uninstall=("pytest-sugar",)),
    ),
    # spacy: its requirements (thinc==7.4.0, cymem/preshed/murmurhash/blis) never install, so
    # `import thinc` / `spacy.pipeline.pipes` fail. The default `pip install -r requirements.txt`
    # pass dies under PEP517 build isolation: the sdists' build-system pulls `cython>=3.1`, which
    # has no distribution for the env's Python (3.7/3.8<3.9) -> "No matching distribution". Fix by
    # pre-installing an era-appropriate cython then installing with `--no-build-isolation` (uses the
    # env's cython 0.29 instead of fetching a modern one), then building spacy's own C-extensions
    # (fixes `spacy.pipeline.pipes`). Mirrors spacy/bugs/*/setup.sh, drift-hardened. fast_build
    # routes the source builds through ccache at -O0.
    "spacy": FauxPyProjectConfig(
        src=".",
        exclude=("spacy/tests", "env"),
        env=FauxPyEnvHooks(
            run=(
                'pip install -q "cython<3.0" numpy',
                "pip install -q --only-binary=:all: -r requirements.txt",
                fast_build("python setup.py build_ext --inplace"),
            ),
        ),
    ),
    "thefuck": FauxPyProjectConfig(
        src="thefuck",
        # Old thefuck conftest uses pytest's removed get_marker(); the authors replace the whole
        # tests/conftest.py with a fixed root conftest.py (rm tests/conftest.py, then cp). Both define
        # --enable-functional, so the original must be removed first or pytest_addoption double-registers.
        # The authors only shipped that fixture for a subset of bugs; the rest (5,10,11,12,16,18,21)
        # kept the old `request.node.get_marker('functional')` -> AttributeError under FauxPy's newer
        # pytest -> whole session errors. get_marker/get_closest_marker are boolean-equivalent here, so
        # patch the call in-place for every thefuck bug (no-op for fixture bugs, whose tests/conftest.py
        # is removed and replaced below).
        host=FauxPyHostHooks(
            replace_in_files=(("tests/conftest.py", "get_marker(", "get_closest_marker("),),
            copy_fixtures=(
                FixtureCopy(
                    "thefuck/B{bug}/conftest.py",
                    "conftest.py",
                    bugs=_fixture_bug_ids("thefuck"),
                    remove_before=("tests/conftest.py",),
                ),
            ),
        ),
    ),
    "tornado": FauxPyProjectConfig(src="tornado", exclude=("tornado/test",)),
    "tqdm": FauxPyProjectConfig(
        src="tqdm",
        exclude=("tqdm/tests",),
        # pytest won't collect modules named tests_*; copy them to test_* so they're collected.
        host=FauxPyHostHooks(rename_tests_glob=True),
    ),
    "youtube-dl": FauxPyProjectConfig(src="youtube_dl"),
    # --- Derived (no reference; from package layout) -----------------------------------------
    # ansible: lib/ layout, package at lib/ansible, tests at test/. setup.py install puts ansible in
    # site-packages, but keep lib on PYTHONPATH so `import ansible` resolves regardless.
    "ansible": FauxPyProjectConfig(src="lib/ansible", exclude=("test",), extra_pythonpath=("lib",)),
    # matplotlib: lib/ layout, in-package tests. Every bug ships an EMPTY requirements.txt and
    # (29/30) an empty setup.sh, so bugsinpy-compile installs nothing -> the FauxPy env has only
    # python+pytest -> `import matplotlib` raises ModuleNotFoundError: numpy. Re-apply the authors'
    # recipe (matplotlib/bugs/1/setup.sh) as an env hook for ALL bugs: `pip install Cython` then an
    # editable install, which pulls numpy (+ the rest of install_requires) into the env and builds
    # the C-extensions in-place under lib/matplotlib (where --src points). pip's PEP517 build
    # isolation installs the build-time numpy; matplotlib's setupext downloads+builds freetype when
    # the system lacks it (the container has no freetype/libpng -dev). fast_build routes the compile
    # through ccache at -O0 (env vars propagate into pip's build subprocess).
    "matplotlib": FauxPyProjectConfig(
        src="lib/matplotlib",
        exclude=("lib/matplotlib/tests",),
        env=FauxPyEnvHooks(
            # matplotlib bugs share one conda env (empty requirements + same python => same hash),
            # so a prior bug's editable install lingers (a matplotlib-nspkg.pth pointing at a deleted
            # checkout) and makes the next bug's rebuild skip the freetype download -> "FreeType
            # version 2.3 or higher is required". Uninstall first for a clean per-bug install, and
            # force MPLLOCALFREETYPE=1 so matplotlib always downloads+builds its own freetype (the
            # container has no system freetype -dev).
            uninstall=("matplotlib",),
            run=(
                "pip install -q Cython",
                fast_build("MPLLOCALFREETYPE=1 python -m pip install -e ."),
            ),
        ),
        # matplotlib's conftest sets ("filterwarnings", "error"), which turns coverage's benign
        # "module-not-measured" (mpl_toolkits) warning into a session-aborting INTERNALERROR ->
        # empty coverage. Comment that single line out so the FauxPy run completes.
        host=FauxPyHostHooks(
            comment_file_lines=(("lib/matplotlib/testing/conftest.py", "filterwarnings.*error"),),
        ),
    ),
    # scrapy: flat package at scrapy/, tests at tests/; not installed (cwd suffices for import).
    # scrapy's root conftest.py has a class-scoped ``reactor_pytest`` fixture that reads the
    # ``--reactor`` pytest option; that option is registered by the pytest-twisted plugin (scrapy's
    # pytest.ini sets ``twisted = 1``). BugsInPy's own run_test.sh drives the triggers via
    # ``python -m unittest`` (which never loads conftest), so its frozen requirements.txt pins pytest
    # + Twisted but *omits* pytest-twisted. Under FauxPy (pytest) the conftest loads and every test in
    # the module errors with ``ValueError: no option named '--reactor'`` -> 0 collected -> empty
    # spectra -> real_failure. Install pytest-twisted so ``--reactor`` is registered (default
    # "default"); pinned <1.14 for compatibility with the pinned pytest 5.4.2 / Twisted 20.3.0.
    "scrapy": FauxPyProjectConfig(
        src="scrapy",
        exclude=("tests",),
        # Two independent pytest-only breakages (BugsInPy drives the triggers via unittest, which
        # loads neither conftest nor a coverage tracer, so it never hits either):
        #   1. scrapy's root conftest.py has a ``reactor_pytest`` fixture reading the ``--reactor``
        #      option, which is registered by the pytest-twisted plugin (pytest.ini sets
        #      ``twisted = 1``). The frozen requirements pin pytest + Twisted but omit pytest-twisted
        #      -> every test errors ``no option named '--reactor'`` -> 0 collected. Install it.
        #   2. With pytest-twisted loaded the tests run inside the Twisted reactor *greenlet*, but
        #      FauxPy starts coverage per-test (``coverage.Coverage().start()``) on the main greenlet
        #      *after* the reactor greenlet already exists, so its trace fn never reaches that
        #      greenlet -> ``No data was collected`` -> empty spectra (plain ``coverage run``, which
        #      starts before any greenlet, is unaffected). ``coverage.Coverage()`` reads ``.coveragerc``
        #      from the pytest cwd, so authoring one with ``concurrency = greenlet`` makes FauxPy
        #      trace the reactor greenlet. (greenlet ships as a pytest-twisted dependency.)
        env=FauxPyEnvHooks(run=("pip install -q 'pytest-twisted<1.14'",)),
        host=FauxPyHostHooks(write_files=((".coveragerc", "[run]\nconcurrency = greenlet\n"),)),
    ),
    # PySnooper: flat package at pysnooper/, tests at tests/; scoping --src drops setup.py/misc/ noise.
    "PySnooper": FauxPyProjectConfig(src="pysnooper", exclude=("tests",)),
}


# ---------------------------------------------------------------------------
# Shell-command rendering (pure; the runner splices these into its bash script)
# ---------------------------------------------------------------------------


def exclude_arg(cfg: FauxPyProjectConfig | None) -> str:
    """FauxPy ``--exclude`` value: a bracketed, comma-joined list (``[]`` when empty)."""
    entries = cfg.exclude if cfg else ()
    return "[" + ",".join(entries) + "]"


def env_hook_commands(cfg: FauxPyProjectConfig, env: str) -> list[str]:
    """In-container commands run in the conda env before pytest (uninstalls, then build steps)."""
    cmds = [f"conda run -n {env} pip uninstall -y {pkg} || true" for pkg in cfg.env.uninstall]
    cmds += [f"conda run -n {env} {cmd} || true" for cmd in cfg.env.run]
    return cmds


def _comment_pytest_ini_cmd(pattern: str) -> str:
    """Comment out (prefix ``# ``) any *uncommented* ``pytest.ini`` line containing ``pattern``.

    ``pattern`` is used as a sed BRE address (delimiter ``|``); the known patterns
    (``--tb=native``, ``-n 2``) contain no regex metacharacters.
    """
    return f"[ -f pytest.ini ] && sed -i '\\|{pattern}|{{/^[[:space:]]*#/!s/^/# /}}' pytest.ini || true"


def _replace_in_file_cmd(path: str, old: str, new: str) -> str:
    """In-place literal string replacement in a checkout file (cwd = source tree) via ``sed``.

    ``old``/``new`` are used as sed ``s`` operands with a ``|`` delimiter, so they must not contain
    ``|`` (the known use — thefuck's ``get_marker(`` -> ``get_closest_marker(`` in tests/conftest.py,
    porting the pre-pytest-4 marker API — has none). Guarded by ``[ -f ]`` and ``|| true`` so a
    missing file is a no-op rather than a failure.
    """
    return f"[ -f {path} ] && sed -i 's|{old}|{new}|g' {path} || true"


def _comment_file_line_cmd(path: str, pattern: str) -> str:
    """Comment out (prefix ``# ``) any *uncommented* line in ``path`` containing ``pattern``.

    ``pattern`` is a sed BRE address (delimiter ``|``). Used to neutralise in-source pytest config
    set programmatically (e.g. matplotlib's conftest ``("filterwarnings", "error")``, which turns
    coverage's benign ``module-not-measured`` warning into a session-aborting INTERNALERROR).
    """
    return f"[ -f {path} ] && sed -i '\\|{pattern}|{{/^[[:space:]]*#/!s/^/# /}}' {path} || true"


# Copy tqdm's ``tests_*`` modules to ``test_*`` so pytest collects them (cwd = source tree).
_TQDM_RENAME_CMD = (
    'for f in tqdm/tests/test*; do n="${f/tests_/test_}"; [ "$n" != "$f" ] && cp "$f" "$n"; done'
)


def _write_file_cmd(path: str, content: str) -> str:
    """Author a small file verbatim in the checkout (cwd = source tree) via ``printf '%b'``.

    Used for config the SUT doesn't ship but FauxPy's coverage needs — e.g. scrapy's
    ``.coveragerc`` with ``concurrency = greenlet`` (coverage.Coverage() reads it from cwd).
    Content is expected to be plain ASCII config (no single quotes); newlines are encoded as
    ``\\n`` for ``%b``. Kept minimal on purpose — not a general templating mechanism.
    """
    escaped = content.replace("\\", "\\\\").replace("\n", "\\n")
    return f"printf '%b' '{escaped}' > {path}"


def host_shell_commands(cfg: FauxPyProjectConfig) -> list[str]:
    """In-container commands that mutate the checked-out source tree, excluding fixture copies.

    Run from cwd = the source tree, so all paths are source-tree-relative. Fixture copies are
    handled by the runner (they require host-side staging) — see :func:`fixture_copies_for`.
    """
    cmds: list[str] = []
    for pattern in cfg.host.comment_pytest_ini:
        cmds.append(_comment_pytest_ini_cmd(pattern))
    for path, pattern in cfg.host.comment_file_lines:
        cmds.append(_comment_file_line_cmd(path, pattern))
    for path, old, new in cfg.host.replace_in_files:
        cmds.append(_replace_in_file_cmd(path, old, new))
    for path in cfg.host.remove_paths:
        cmds.append(f"rm -rf {path}")
    if cfg.host.rename_tests_glob:
        cmds.append(_TQDM_RENAME_CMD)
    for path, content in cfg.host.write_files:
        cmds.append(_write_file_cmd(path, content))
    return cmds


def fixture_copies_for(cfg: FauxPyProjectConfig, bug_id: int) -> list[FixtureCopy]:
    """Fixture copies that apply to ``bug_id`` (empty when none / project has no fixtures)."""
    return [fc for fc in cfg.host.copy_fixtures if fc.applies_to(bug_id)]
