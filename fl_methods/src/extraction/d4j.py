"""Defects4J repository operations: checkout, compile, export properties, test methods.

This is the lean extraction-only version of D4JRepo — no ComponentGraph,
no pickle, no AST/JPype machinery.  All commands run inside the Docker
container via :mod:`src.extraction.docker`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.common.config import (
    D4J_CONTAINER_DIR,
    get_benchmark_fixed_root,
    get_benchmark_processed_root,
    get_benchmark_repos_root,
)
from src.extraction.docker import ensure_project_dir_layout_writable, run_defects4j

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Properties exported from Defects4J
# ---------------------------------------------------------------------------

EXPORT_PROPERTIES = (
    "classes.modified",
    "classes.relevant",
    "cp.test",
    "dir.bin.classes",
    "dir.bin.tests",
    "dir.src.classes",
    "dir.src.tests",
    "tests.relevant",
    "tests.trigger",
)


def write_parsed_failing_tests(output_dir: Path) -> bool:
    """Parse raw ``failing_tests`` into ``failing_tests.txt``.

    Returns ``False`` when the raw ``failing_tests`` artifact is missing.
    Otherwise writes the parsed file and returns whether it exists afterward.
    """
    raw = output_dir / "failing_tests"
    out = output_dir / "failing_tests.txt"
    if not raw.exists():
        return False
    tests: list[str] = []
    for line in raw.read_text().splitlines():
        if line.startswith("--- "):
            tests.append(line[4:].strip())
    out.write_text("\n".join(tests) + ("\n" if tests else ""))
    return out.exists()


# ---------------------------------------------------------------------------
# D4JRepo
# ---------------------------------------------------------------------------


class D4JRepo:
    """Manages a single Defects4J bug checkout and its extracted artifacts.

    Parameters
    ----------
    project:
        Defects4J project identifier (e.g. ``"Chart"``).
    bug_id:
        Bug/version number.
    buggy:
        If True (default), work with the buggy version; otherwise the fixed version.
    """

    def __init__(self, project: str, bug_id: int, *, buggy: bool = True) -> None:
        self.project = project
        self.bug_id = bug_id
        self.buggy = buggy

        if buggy:
            self.version_flag = f"{bug_id}b"
            self.repo_dir = get_benchmark_repos_root() / project / str(bug_id)
        else:
            self.version_flag = f"{bug_id}f"
            self.repo_dir = get_benchmark_fixed_root() / project / str(bug_id)

        self.output_dir = get_benchmark_processed_root() / project / str(bug_id)

    # ------------------------------------------------------------------
    # Checkout & compile
    # ------------------------------------------------------------------

    def is_checked_out(self) -> bool:
        """Return True if the repo is already checked out."""
        return (self.repo_dir / "defects4j.build.properties").exists()

    def checkout(self, *, skip_existing: bool = True) -> None:
        """Checkout this project/version into :attr:`repo_dir`."""
        if skip_existing and self.is_checked_out():
            logger.info(
                "%s-%s already checked out at %s, skipping.",
                self.project,
                self.version_flag,
                self.repo_dir,
            )
            return

        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Checking out %s-%s to %s",
            self.project,
            self.version_flag,
            self.repo_dir,
        )
        command = [
            "checkout",
            "-p",
            self.project,
            "-v",
            self.version_flag,
            "-w",
            str(self.repo_dir),
        ]

        try:
            run_defects4j(command)
        except RuntimeError as exc:
            message = str(exc)
            needs_layout_fix = (
                self.project == "Chart"
                and "dir-layout.csv" in message
                and "Permission denied" in message
            )
            if not needs_layout_fix:
                raise

            logger.warning(
                "Checkout for %s-%s failed due to dir-layout permission; retrying after fix.",
                self.project,
                self.version_flag,
            )
            ensure_project_dir_layout_writable(self.project)

            if self.repo_dir.exists():
                shutil.rmtree(self.repo_dir)
            self.repo_dir.mkdir(parents=True, exist_ok=True)

            run_defects4j(command)

    def is_compiled(self) -> bool:
        """Return True if the compiled classes directory exists."""
        try:
            bin_dir = self._read_property("dir.bin.classes")
        except FileNotFoundError:
            return False
        return (self.repo_dir / bin_dir).exists()

    def compile(self, *, skip_existing: bool = True) -> None:
        """Compile the checked-out project."""
        if skip_existing and self.is_compiled():
            logger.info(
                "%s-%s already compiled, skipping.",
                self.project,
                self.version_flag,
            )
            return

        logger.info("Compiling %s-%s", self.project, self.version_flag)
        run_defects4j(["compile", "-w", str(self.repo_dir)])

    # ------------------------------------------------------------------
    # Property export
    # ------------------------------------------------------------------

    def export_property(self, prop: str) -> str:
        """Export a Defects4J property and cache to file.

        Returns the property value as a string (newline-separated for
        multi-valued properties).
        """
        prop_file = self.output_dir / prop
        if prop_file.exists():
            return prop_file.read_text().strip()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting %s for %s-%d", prop, self.project, self.bug_id)
        run_defects4j(
            [
                "export",
                "-p",
                prop,
                "-w",
                str(self.repo_dir),
                "-o",
                str(prop_file),
            ]
        )
        return prop_file.read_text().strip()

    def export_all_properties(self) -> None:
        """Export all standard properties."""
        for prop in EXPORT_PROPERTIES:
            self.export_property(prop)

    def _read_property(self, prop: str) -> str:
        """Read a property from the cached file, raising if not yet exported."""
        prop_file = self.output_dir / prop
        if not prop_file.exists():
            raise FileNotFoundError(
                f"Property file {prop_file} does not exist. Run export_property('{prop}') first."
            )
        return prop_file.read_text().strip()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_bin_class_dir(self) -> Path:
        """Return absolute path to compiled production classes."""
        rel = self.export_property("dir.bin.classes")
        return self.repo_dir / rel

    def get_src_class_dir(self) -> Path:
        """Return absolute path to source classes."""
        rel = self.export_property("dir.src.classes")
        return self.repo_dir / rel

    def get_src_tests_dir(self) -> Path:
        """Return absolute path to test sources."""
        rel = self.export_property("dir.src.tests")
        return self.repo_dir / rel

    def get_bin_tests_dir(self) -> Path:
        """Return absolute path to compiled test classes."""
        rel = self.export_property("dir.bin.tests")
        return self.repo_dir / rel

    def get_cp_test(self) -> str:
        """Return the test classpath string."""
        return self.export_property("cp.test")

    def get_modified_classes(self) -> list[str]:
        """Return list of modified class FQCNs."""
        return self.export_property("classes.modified").splitlines()

    def get_relevant_classes(self) -> list[str]:
        """Return list of relevant class FQCNs."""
        return self.export_property("classes.relevant").splitlines()

    def get_relevant_test_classes(self) -> list[str]:
        """Return list of relevant test class FQCNs."""
        return self.export_property("tests.relevant").splitlines()

    def get_trigger_tests(self) -> list[str]:
        """Return list of trigger test IDs (``Class::method`` format)."""
        return self.export_property("tests.trigger").splitlines()

    def classpath_from_class_signature(self, signature: str) -> Path:
        """Convert a FQCN like ``org.example.MyClass`` to its ``.java`` source path."""
        relpath = signature.replace(".", "/") + ".java"
        return self.get_src_class_dir() / relpath

    # ------------------------------------------------------------------
    # Test method enumeration
    # ------------------------------------------------------------------

    def get_all_test_methods(self, *, skip_existing: bool = True) -> list[str]:
        """Run ``defects4j test`` and return normalised test method IDs.

        Format: ``TestClass#methodName`` (class part may be fully qualified or short).

        Side effects:
            - Writes ``all_tests.txt`` to :attr:`output_dir`.
            - Copies ``failing_tests`` from the repo if present.
            - Writes parsed ``failing_tests.txt`` (one ``Class::method`` per line).
            - Copies ``trigger_tests`` from the D4J installation.
        """
        all_tests_path = self.output_dir / "all_tests.txt"
        if skip_existing and all_tests_path.exists():
            # Backfill the parsed failing_tests.txt if missing (e.g., for bugs
            # processed before this artifact existed).
            self._write_parsed_failing_tests()
            return [
                line.strip() for line in all_tests_path.read_text().splitlines() if line.strip()
            ]

        logger.info("Running defects4j test for %s-%d", self.project, self.bug_id)
        run_defects4j(["test", "-w", str(self.repo_dir)])

        # Copy failing_tests if present
        failing_src = self.repo_dir / "failing_tests"
        if failing_src.exists():
            shutil.copy2(failing_src, self.output_dir / "failing_tests")
        self._write_parsed_failing_tests()

        # Copy trigger_tests from the D4J installation inside the container
        trigger_container = (
            f"{D4J_CONTAINER_DIR}/framework/projects/{self.project}/trigger_tests/{self.bug_id}"
        )
        trigger_dst = self.output_dir / "trigger_tests"
        if not trigger_dst.exists():
            # Cat the file out via the active container runtime
            from src.extraction.docker import docker_exec

            result = docker_exec(
                ["cat", trigger_container],
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                trigger_dst.write_text(result.stdout)

        # Parse all_tests from the repo
        all_tests_file = self.repo_dir / "all_tests"
        if not all_tests_file.exists():
            raise FileNotFoundError(
                f"Expected {all_tests_file} after 'defects4j test'. "
                "The test command may have failed."
            )

        test_methods: list[str] = []
        with all_tests_file.open() as src, all_tests_path.open("w") as dst:
            for line in src:
                line = line.strip()
                if "(" in line and ")" in line:
                    method_name = line.split("(")[0].strip()
                    class_name = line.split("(")[1].split(")")[0].strip()
                    formatted = f"{class_name}#{method_name}"
                    test_methods.append(formatted)
                    dst.write(formatted + "\n")

        return test_methods

    def get_relevant_test_methods(self, *, skip_existing: bool = True) -> list[str]:
        """Filter all test methods to keep only those in relevant test classes.

        Writes ``relevant_tests.txt`` and ``junit_tests.txt`` to :attr:`output_dir`.
        """
        relevant_path = self.output_dir / "relevant_tests.txt"
        junit_path = self.output_dir / "junit_tests.txt"

        if skip_existing and relevant_path.exists() and junit_path.exists():
            return [line.strip() for line in relevant_path.read_text().splitlines() if line.strip()]

        relevant_classes = set(self.get_relevant_test_classes())
        simple_to_fqcn: dict[str, set[str]] = {}
        for test_class in relevant_classes:
            simple_name = test_class.rsplit(".", maxsplit=1)[-1]
            simple_to_fqcn.setdefault(simple_name, set()).add(test_class)

        all_methods = self.get_all_test_methods(skip_existing=skip_existing)

        methods: set[str] = set()
        fallback_matches = 0
        ambiguous_matches = 0
        for method in all_methods:
            if "#" not in method:
                continue

            test_class, test_method = method.split("#", maxsplit=1)
            if test_class in relevant_classes:
                methods.add(method)
                continue

            test_class_simple = test_class.split(".", maxsplit=1)[0]
            fqcn_candidates = simple_to_fqcn.get(test_class_simple, set())
            if len(fqcn_candidates) == 1:
                resolved_class = next(iter(fqcn_candidates))
                methods.add(f"{resolved_class}#{test_method}")
                fallback_matches += 1
            elif len(fqcn_candidates) > 1:
                ambiguous_matches += 1

        if fallback_matches:
            logger.warning(
                (
                    "Matched %d test methods via simple-name fallback for %s-%d; "
                    "normalised to FQCN for GZoltar compatibility."
                ),
                fallback_matches,
                self.project,
                self.bug_id,
            )

        if ambiguous_matches:
            logger.warning(
                (
                    "Skipped %d test methods due to ambiguous simple-name matches "
                    "while resolving relevant tests for %s-%d."
                ),
                ambiguous_matches,
                self.project,
                self.bug_id,
            )

        sorted_methods = sorted(methods)
        with relevant_path.open("w") as rel_f, junit_path.open("w") as junit_f:
            for m in sorted_methods:
                rel_f.write(f"{m}\n")
                junit_f.write(f"JUNIT,{m}\n")

        logger.info(
            "Wrote %d relevant test methods to %s and %s",
            len(sorted_methods),
            relevant_path,
            junit_path,
        )
        return sorted_methods

    def _write_parsed_failing_tests(self) -> None:
        """Parse the raw ``failing_tests`` dump into a flat ``failing_tests.txt``.

        D4J's ``failing_tests`` is a multi-line file: each failing test starts
        with a ``--- <Class>::<method>`` marker followed by its stack trace.
        This helper extracts the marker lines into one ``Class::method`` per
        line, suitable for tracker counts and downstream tooling.

        No-op when ``failing_tests`` is missing (i.e., the bug had no failing
        tests, or this method runs before :meth:`get_all_test_methods`).
        """
        write_parsed_failing_tests(self.output_dir)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_repo(self) -> None:
        """Remove the checked-out repo directory."""
        if self.repo_dir.exists():
            shutil.rmtree(self.repo_dir)
            logger.info("Removed %s", self.repo_dir)


# ---------------------------------------------------------------------------
# Project/version discovery
# ---------------------------------------------------------------------------


def get_d4j_pids() -> list[str]:
    """Return all Defects4J project identifiers."""
    result = run_defects4j(["pids"])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_d4j_bids(project: str) -> list[int]:
    """Return all bug IDs for *project*, sorted numerically."""
    result = run_defects4j(["bids", "-p", project])
    bids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return sorted(bids)
