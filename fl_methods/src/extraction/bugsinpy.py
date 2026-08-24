"""BugsInPy repository operations: checkout, compile, remove.

The BugsInPy analogue of :mod:`src.extraction.d4j`. All container work goes through
the generic, conda-init-aware wrapper ``utils/docker/bugsinpy/cmd-docker-bip.sh``
(see :data:`src.common.config.BIP_WRAPPER`) via :func:`run_bip`.

The lifecycle methods (``checkout`` / ``compile`` / ``remove_repo``) work against the
container. Properties are computed on demand from checkout metadata (``bug.info`` /
``run_test.sh`` / ``bug_patch.txt``) — there is no ``defects4j export`` analogue — and
test enumeration is produced by the FauxPy coverage step, so ``export_property`` and
the test-enumeration methods raise ``NotImplementedError`` by design.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import shutil
import subprocess
from pathlib import Path

from src.common.config import (
    BIP_CONTAINER_WORKSPACE,
    BIP_INDEX_CONTAINER,
    BIP_INDEX_CSV,
    BIP_WRAPPER,
    get_benchmark_fixed_root,
    get_benchmark_processed_root,
    get_benchmark_repos_root,
    get_bip_workspace_root,
    get_results_meta_dir,
)

logger = logging.getLogger(__name__)

_BENCHMARK = "bugsinpy"

# bugsinpy-compile prints this when the per-bug conda env is missing (and then exits
# with status 0 — see compile() for why we cannot rely on the return code). We parse
# the env-name hash and python version out of it.
_CONDA_CREATE_RE = re.compile(r"conda create -n (\S+) -y python=([0-9][0-9.]*)")


# ---------------------------------------------------------------------------
# Container interaction
# ---------------------------------------------------------------------------


def run_bip(
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command inside the BugsInPy container via ``cmd-docker-bip.sh``.

    The wrapper ensures the container is up, sources conda, maps the workspace, and
    runs as root. ``args`` is the full command vector (e.g.
    ``["bugsinpy-checkout", "-p", ...]``). No timeout is imposed — a cold
    ``conda create`` legitimately takes minutes.
    """
    logger.info("BIP: %s", " ".join(args))
    result = subprocess.run(
        [str(BIP_WRAPPER), *args],
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"BugsInPy command failed (rc={result.returncode}): {' '.join(args)}\n{stderr}"
        )
    return result


def _to_bip_container_path(host_path: str | Path) -> str:
    """Translate a host path under the BugsInPy workspace to a ``/workspace/...`` path.

    Raises ``ValueError`` if *host_path* is not under the workspace — passing such a
    path to the root container would silently fail deep inside framework output.
    """
    host = Path(host_path).resolve()
    workspace = get_bip_workspace_root().resolve()
    try:
        rel = host.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Path {host} is not under the BugsInPy workspace {workspace}; "
            "cannot translate it to a container path."
        ) from None
    rel_str = rel.as_posix()
    if rel_str == ".":
        return BIP_CONTAINER_WORKSPACE
    return f"{BIP_CONTAINER_WORKSPACE}/{rel_str}"


def _parse_conda_create(text: str | None) -> tuple[str | None, str | None]:
    """Extract ``(env_name, python_version)`` from a bugsinpy-compile message."""
    if not text:
        return None, None
    match = _CONDA_CREATE_RE.search(text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


# ---------------------------------------------------------------------------
# Bug-metadata parsing (run_test.sh / bug_patch.txt -> canonical IDs)
# ---------------------------------------------------------------------------


def _path_to_module(rel_path: str) -> str:
    """Convert a source-relative file path to a dotted module path.

    ``test/test_InfoExtractor.py`` -> ``test.test_InfoExtractor``;
    ``pkg/__init__.py`` -> ``pkg``.
    """
    p = rel_path.strip().strip("/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = [seg for seg in p.split("/") if seg]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _modified_files(patch_text: str | None, patchfile_info: str | None) -> list[str]:
    """Return changed ``.py`` files as source-relative paths.

    Prefers the unified diff's ``diff --git a/<p> b/<p>`` headers; falls back to
    the semicolon-separated ``bugsinpy_patchfile.info`` list.
    """
    files: list[str] = []
    if patch_text:
        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                match = re.match(r"diff --git a/(.+?) b/(.+)$", line.strip())
                if match:
                    path = match.group(2).strip()
                    if path.endswith(".py") and path not in files:
                        files.append(path)
    if not files and patchfile_info:
        for raw in patchfile_info.replace("\n", ";").split(";"):
            path = raw.strip()
            if path.endswith(".py") and path not in files:
                files.append(path)
    return files


def _make_trigger_id(module: str, rest: list[str]) -> str | None:
    """Build a trigger ID from a module and a residual ``[Class..., method]`` chain.

    ``(mod, [Cls, m])`` -> ``mod$Cls::m``; ``(mod, [m])`` (module-level test) ->
    ``mod::m``; empty residual -> ``None`` (no method-level test specified).
    """
    rest = [r for r in rest if r]
    if not rest:
        return None
    if len(rest) == 1:
        return f"{module}::{rest[0]}"
    cls = ".".join(rest[:-1])
    return f"{module}${cls}::{rest[-1]}"


def _parse_trigger_tests(run_test_text: str, test_file: str) -> list[str]:
    """Parse ``run_test.sh`` into trigger IDs (``<module>$<Class>::<method>``).

    Handles both ``python -m unittest <module>.<Class>.<method>`` (anchored on the
    ``bug.info`` ``test_file`` to split module from class/method) and pytest node
    IDs (``path.py::Class::method``).
    """
    # ``bug.info`` ``test_file`` may list several files joined by ``;`` (e.g. pandas
    # ``a.py;b.py``); each yields candidate dotted modules to anchor unittest tokens on. For a
    # package ``__init__.py`` test module, keep BOTH the collapsed form (``pkg``) and the raw
    # form (``pkg.__init__``) the unittest token actually uses; try the most specific first.
    test_modules: list[str] = []
    for f in re.split(r"[;\s]+", test_file):
        if not f.endswith(".py"):
            continue
        for cand in (f[:-3].replace("/", "."), _path_to_module(f)):
            if cand and cand not in test_modules:
                test_modules.append(cand)
    test_modules.sort(key=len, reverse=True)
    triggers: list[str] = []
    # ``;`` also chains multiple runner invocations on one line (e.g. matplotlib's
    # ``pytest a::test_x;pytest a::test_y``); treat it as a token separator like whitespace.
    for token in run_test_text.replace(";", " ").split():
        ident: str | None = None
        if "::" in token:  # pytest node id: <path>.py::Class::method
            parts = token.split("::")
            path = parts[0]
            module = _path_to_module(path) if path.endswith(".py") else path
            ident = _make_trigger_id(module, parts[1:])
        else:  # unittest dotted id: <module>.<Class>.<method> — anchor on a candidate module
            tm = next((m for m in test_modules if token == m or token.startswith(m + ".")), None)
            if tm:
                suffix = token[len(tm) :].lstrip(".")
                ident = _make_trigger_id(tm, suffix.split(".") if suffix else [])
        if ident and ident not in triggers:
            triggers.append(ident)
    return triggers


# ---------------------------------------------------------------------------
# BugsInPyRepo
# ---------------------------------------------------------------------------


class BugsInPyRepo:
    """Manages a single BugsInPy bug checkout and its extracted artifacts.

    Parameters
    ----------
    project:
        BugsInPy project/repo name (e.g. ``"youtube-dl"``, ``"PySnooper"``).
    bug_id:
        Bug number.
    buggy:
        If True (default), work with the buggy version; otherwise the fixed version.

    Layout note
    -----------
    ``bugsinpy-checkout -w <repo_dir>`` ``git clone``s into ``<repo_dir>/<project>/``
    and writes the bug metadata (``bugsinpy_bug.info`` etc.) and the compile sentinel
    (``bugsinpy_compile_flag``) there. So the source tree and lifecycle markers live at
    :attr:`source_tree_dir` = ``repo_dir / project``, one level below ``repo_dir``.
    """

    def __init__(self, project: str, bug_id: int, *, buggy: bool = True) -> None:
        self.project = project
        self.bug_id = bug_id
        self.buggy = buggy

        if buggy:
            self.version_flag = f"{bug_id}b"
            self.repo_dir = get_benchmark_repos_root(_BENCHMARK) / project / str(bug_id)
        else:
            self.version_flag = f"{bug_id}f"
            self.repo_dir = get_benchmark_fixed_root(_BENCHMARK) / project / str(bug_id)

        self.output_dir = get_benchmark_processed_root(_BENCHMARK) / project / str(bug_id)

    @property
    def source_tree_dir(self) -> Path:
        """The cloned source tree (``repo_dir / project``) — see class docstring."""
        return self.repo_dir / self.project

    @property
    def _checkout_version(self) -> str:
        """bugsinpy-checkout ``-v`` value (0 = buggy, 1 = fixed)."""
        return "0" if self.buggy else "1"

    # ------------------------------------------------------------------
    # Checkout & compile
    # ------------------------------------------------------------------

    def is_checked_out(self) -> bool:
        """Return True if the repo is already checked out."""
        return (self.source_tree_dir / "bugsinpy_bug.info").exists()

    def checkout(self, *, skip_existing: bool = True) -> None:
        """Checkout this project/version under :attr:`repo_dir`."""
        if skip_existing and self.is_checked_out():
            logger.info(
                "%s-%s already checked out at %s, skipping.",
                self.project,
                self.version_flag,
                self.source_tree_dir,
            )
            return

        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # `git clone` (inside bugsinpy-checkout) refuses a non-empty target, so a
        # stale/partial clone must be cleared first. It is root-owned (the container
        # runs as root), so the removal must go THROUGH the container.
        if self.source_tree_dir.exists():
            logger.warning(
                "Removing stale checkout at %s before re-checkout.", self.source_tree_dir
            )
            run_bip(["rm", "-rf", _to_bip_container_path(self.source_tree_dir)], check=False)

        logger.info("Checking out %s-%s to %s", self.project, self.version_flag, self.repo_dir)
        run_bip(
            [
                "bugsinpy-checkout",
                "-p",
                self.project,
                "-i",
                str(self.bug_id),
                "-v",
                self._checkout_version,
                "-w",
                _to_bip_container_path(self.repo_dir),
            ]
        )

    def is_compiled(self) -> bool:
        """Return True if the BugsInPy compile sentinel is present and set."""
        flag = self.source_tree_dir / "bugsinpy_compile_flag"
        if not flag.exists():
            return False
        try:
            return flag.read_text().strip() == "1"
        except OSError:
            return False

    def compile(self, *, skip_existing: bool = True) -> None:
        """Compile the checked-out project (creating its conda env on first run).

        bugsinpy-compile's contract is awkward: when the per-bug conda env does not yet
        exist it prints a ``conda create -n <hash> ... python=<ver>`` hint and exits
        with status 0 (not an error). So success is signalled by the
        ``bugsinpy_compile_flag`` sentinel, never the return code.
        """
        if skip_existing and self.is_compiled():
            logger.info("%s-%s already compiled, skipping.", self.project, self.version_flag)
            return
        if not self.is_checked_out():
            raise RuntimeError(
                f"{self.project}-{self.version_flag} is not checked out; call checkout() first."
            )

        work = _to_bip_container_path(self.source_tree_dir)
        logger.info("Compiling %s-%s", self.project, self.version_flag)
        first = run_bip(["bugsinpy-compile", "-w", work], check=False, capture=True)

        if not self.is_compiled():
            env_name, py_version = _parse_conda_create(first.stdout)
            if env_name is None or py_version is None:
                raise RuntimeError(
                    f"bugsinpy-compile for {self.project}-{self.version_flag} produced neither a "
                    f"compile flag nor a conda-create request.\n"
                    f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"
                )
            logger.info(
                "Creating conda env %s (python=%s) for %s-%s",
                env_name,
                py_version,
                self.project,
                self.version_flag,
            )
            # The framework's printed hint omits pytest, but FauxPy / test running need
            # it, so build the create command explicitly rather than replaying the hint.
            run_bip(["conda", "create", "-n", env_name, "-y", f"python={py_version}", "pytest"])
            run_bip(["bugsinpy-compile", "-w", work], check=False, capture=True)

        if not self.is_compiled():
            raise RuntimeError(
                f"Compilation of {self.project}-{self.version_flag} did not produce "
                f"bugsinpy_compile_flag == '1' at "
                f"{self.source_tree_dir / 'bugsinpy_compile_flag'}."
            )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bug metadata access (host project clone, with checkout fallback)
    # ------------------------------------------------------------------

    def _metadata_dir(self) -> Path:
        """Host BugsInPy project-clone dir for this bug's metadata."""
        return BIP_INDEX_CSV.parent / self.project / "bugs" / str(self.bug_id)

    def _read_meta(self, checkout_name: str, host_name: str) -> str | None:
        """Return metadata file text, preferring the checkout, else the host clone."""
        for candidate in (
            self.source_tree_dir / checkout_name,
            self._metadata_dir() / host_name,
        ):
            if candidate.exists():
                return candidate.read_text(encoding="utf-8", errors="ignore")
        return None

    def _bug_info_value(self, key: str) -> str | None:
        """Return a ``key="value"`` field from ``bug.info`` (handles ``key ="v"``)."""
        text = self._read_meta("bugsinpy_bug.info", "bug.info")
        if not text:
            return None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(key):
                continue
            after = stripped[len(key) :].lstrip()
            if after.startswith("="):
                return after[1:].strip().strip('"').strip()
        return None

    # ------------------------------------------------------------------
    # Convenience accessors
    #
    # Python property semantics:
    #   - no separate compiled tree  -> bin dirs == src dirs
    #   - no JVM classpath           -> cp.test == ""  (Python uses sys.path)
    #   - "modified/relevant class"  -> a changed *module*, as a source-relative path
    #   - properties are computed on-demand from bug.info / run_test.sh / bug_patch.txt;
    #     no `defects4j export` analogue, so no D4J-named property files are written.
    # ------------------------------------------------------------------

    def get_src_class_dir(self) -> Path:
        """Return the source-classes import root (the dir on ``sys.path`` for the SUT).

        Resolution order:
          1. ``bug.info`` ``pythonpath`` if that dir exists post-compile (it is relative
             to :attr:`repo_dir`, i.e. it includes the project-dir prefix — e.g.
             ``ansible/build/lib`` → ``<repo_dir>/ansible/build/lib``);
          2. else the import root derived from the source layout — walk up from a modified
             file through package dirs (``__init__.py``) to the first non-package ancestor.
             Handles flat (``youtube_dl/``), ``src/``, and ``lib/`` (ansible) layouts;
          3. else the checkout root.
        """
        pythonpath = self._bug_info_value("pythonpath")
        if pythonpath:
            candidate = self.repo_dir / pythonpath
            if candidate.is_dir():
                return candidate
        derived = self._derive_import_root()
        return derived if derived is not None else self.source_tree_dir

    def _derive_import_root(self) -> Path | None:
        """Walk up from a modified file to the first non-package (no ``__init__.py``) dir.

        Returns ``None`` when no modified file is known. Requires the checkout on disk.
        """
        modified = self.get_modified_classes()
        if not modified:
            return None
        directory = self.source_tree_dir
        for part in modified[0].split("/")[:-1]:
            directory = directory / part
        while directory != self.source_tree_dir and directory != directory.parent:
            if not (directory / "__init__.py").exists():
                return directory
            directory = directory.parent
        return self.source_tree_dir

    def get_src_tests_dir(self) -> Path:
        """Return the test-sources root (pytest's rootdir = the cloned tree)."""
        return self.source_tree_dir

    def get_bin_class_dir(self) -> Path:
        """Python has no separate compiled tree; the bin dir is the src dir."""
        return self.get_src_class_dir()

    def get_bin_tests_dir(self) -> Path:
        """Python has no separate compiled test tree; equals the src tests dir."""
        return self.get_src_tests_dir()

    def get_cp_test(self) -> str:
        """No JVM classpath for Python (imports resolve via ``sys.path``)."""
        return ""

    def get_modified_classes(self) -> list[str]:
        """Return the bug's changed source files, as source-relative ``.py`` paths.

        Sourced from the patch (host ``bug_patch.txt`` diff headers, else the
        checkout's ``bugsinpy_patchfile.info`` semicolon list).
        """
        return _modified_files(
            self._read_meta("bug_patch.txt", "bug_patch.txt"),
            self._read_meta("bugsinpy_patchfile.info", "bugsinpy_patchfile.info"),
        )

    def get_relevant_classes(self) -> list[str]:
        """BugsInPy has no separate 'relevant' scope; equals the modified files."""
        return self.get_modified_classes()

    def get_relevant_test_classes(self) -> list[str]:
        """Return the trigger tests' enclosing test classes (``<module>$<Class>``)."""
        classes: list[str] = []
        for trigger in self.get_trigger_tests():
            head = trigger.split("::", 1)[0]
            if "$" in head and head not in classes:
                classes.append(head)
        return classes

    def get_trigger_tests(self) -> list[str]:
        """Return the failing/trigger test IDs in ``<module>$<Class>::<method>`` form.

        Parsed from ``run_test.sh`` (the exact ``unittest``/``pytest`` invocation),
        anchored on ``bug.info``'s ``test_file`` to split module from class/method.
        Module-level test functions take the ``<module>::<func>`` form.
        """
        run_test = self._read_meta("bugsinpy_run_test.sh", "run_test.sh")
        test_file = self._bug_info_value("test_file") or ""
        if not run_test:
            return []
        return _parse_trigger_tests(run_test, test_file)

    def classpath_from_class_signature(self, signature: str) -> Path:
        """Map a dotted module FQN to its ``.py`` source path under the src root."""
        relpath = signature.replace(".", "/") + ".py"
        return self.get_src_class_dir() / relpath

    def get_patch_text(self) -> str | None:
        """Return the buggy↔fixed unified diff (``bug_patch.txt``), or ``None``."""
        return self._read_meta("bug_patch.txt", "bug_patch.txt")

    def conda_env_name(self) -> str:
        """Return the per-bug conda env name created by ``bugsinpy-compile``.

        Deterministic md5 of ``echo $python_version`` (trailing newline) followed
        by the requirements file, mirroring ``bugsinpy-compile``:
        ``md5sum <(echo $bug_python_version) bugsinpy_requirements.txt``.
        """
        python_version = self._bug_info_value("python_version") or ""
        requirements = self._read_meta("bugsinpy_requirements.txt", "requirements.txt") or ""
        return hashlib.md5(f"{python_version}\n{requirements}".encode()).hexdigest()

    def export_property(self, prop: str) -> str:
        raise NotImplementedError(
            "BugsInPy properties are computed on-demand via the get_* accessors "
            f"(no `defects4j export` analogue); export_property({prop!r}) is not used."
        )

    def export_all_properties(self) -> None:
        """No-op: BugsInPy properties are derived on-demand, not exported to disk."""
        logger.debug(
            "export_all_properties is a no-op for BugsInPy (%s-%s); properties are "
            "computed from bug.info / run_test.sh / bug_patch.txt on demand.",
            self.project,
            self.version_flag,
        )

    def get_all_test_methods(self, *, skip_existing: bool = True) -> list[str]:
        """Produced by the FauxPy coverage step (the suite it runs); see fauxpy.py."""
        raise NotImplementedError(
            "BugsInPy test enumeration is produced by the FauxPy coverage step "
            "(all_tests.txt / relevant_tests.txt reflect the suite FauxPy ran)."
        )

    def get_relevant_test_methods(self, *, skip_existing: bool = True) -> list[str]:
        """Produced by the FauxPy coverage step; see :meth:`get_all_test_methods`."""
        raise NotImplementedError(
            "BugsInPy test enumeration is produced by the FauxPy coverage step "
            "(all_tests.txt / relevant_tests.txt reflect the suite FauxPy ran)."
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_repo(self) -> None:
        """Remove the checked-out repo directory (through the container)."""
        if not self.repo_dir.exists():
            return
        # Checkout artifacts are root-owned (the container runs as root), so delete
        # through the container; host-side shutil.rmtree would raise PermissionError.
        try:
            run_bip(["rm", "-rf", _to_bip_container_path(self.repo_dir)], check=False)
        except ValueError:
            logger.exception(
                "Refusing to remove %s: not under the BugsInPy workspace.", self.repo_dir
            )
            return
        # Best-effort host cleanup of an empty parent (may itself be root-owned).
        try:
            if self.repo_dir.exists():
                shutil.rmtree(self.repo_dir, ignore_errors=True)
            parent = self.repo_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
        logger.info("Removed %s", self.repo_dir)


# ---------------------------------------------------------------------------
# Project/version discovery (off bugsinpy-index.csv)
# ---------------------------------------------------------------------------


def _read_index_text() -> str:
    """Return the BugsInPy index CSV text.

    Prefers the benchmark clone, falls back to the snapshot committed under
    ``results/BIP/_meta/`` (so project/bug enumeration works without the clone),
    and finally reads it out of the container.
    """
    if BIP_INDEX_CSV.exists():
        return BIP_INDEX_CSV.read_text()
    snapshot = get_results_meta_dir(_BENCHMARK) / "bugsinpy-index.csv"
    if snapshot.exists():
        return snapshot.read_text()
    return run_bip(["cat", BIP_INDEX_CONTAINER], capture=True).stdout


def _index_rows() -> list[tuple[str, str]]:
    """Parse the index into ``(repo, bugid)`` rows."""
    reader = csv.DictReader(io.StringIO(_read_index_text()))
    rows: list[tuple[str, str]] = []
    for row in reader:
        repo = (row.get("repo") or "").strip()
        bugid = (row.get("bugid") or "").strip()
        if repo and bugid:
            rows.append((repo, bugid))
    return rows


def get_bip_pids() -> list[str]:
    """Return all BugsInPy project identifiers, sorted."""
    return sorted({repo for repo, _ in _index_rows()})


def get_bip_bids(project: str) -> list[int]:
    """Return all bug IDs for *project*, sorted numerically."""
    bids: set[int] = set()
    for repo, bugid in _index_rows():
        if repo == project:
            try:
                bids.add(int(bugid))
            except ValueError:
                continue
    return sorted(bids)
