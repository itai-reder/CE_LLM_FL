"""GZoltar coverage collection and fault-localization report generation.

Runs GZoltar inside the Docker container to produce ``spectra.csv``,
``matrix.txt``, ``tests.csv``, and ``ochiai.ranking.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import D4J_JUNIT_CONTAINER, SFL_SUBDIR, get_src_dir
from src.common.java_parser import extract_methods_from_java, find_java_files
from src.common.method_entity import (
    method_entity_from_method_info,
    method_entity_from_python_method_info,
    write_method_entities_csv,
)
from src.common.python_parser import (
    extract_methods_from_python,
    find_python_files,
    module_path_for_python,
)
from src.core.layout import normalize_benchmark_name
from src.extraction.d4j import D4JRepo
from src.extraction.docker import (
    ensure_java_utils_in_workspace,
    run_java_in_container,
    to_container_path,
)

if TYPE_CHECKING:
    from src.extraction.bugsinpy import BugsInPyRepo

logger = logging.getLogger(__name__)


def run_corpus_method_extraction(
    repo: D4JRepo | BugsInPyRepo,
    skip_existing: bool = True,
    *,
    dataset: str = "defects4j",
) -> None:
    """Extract method entities keyed on corpus identity.

    Produces ``method_signatures.csv`` under *repo.output_dir*, with
    header ``corpus_id;path;startLine;endLine``.  The corpus identity
    (``pkg$Class.method(SimpleParams)``) is the canonical key used by
    downstream ranking and aggregation logic.  Source methods are parsed by
    :func:`src.common.java_parser.extract_methods_from_java`.
    """
    output_csv = repo.output_dir / "method_signatures.csv"
    expected_header = "corpus_id;path;startLine;endLine"
    if skip_existing and output_csv.exists():
        first_line = output_csv.read_text(encoding="utf-8").split("\n", 1)[0].strip()
        if first_line == expected_header:
            logger.debug("Skipping corpus method extraction (already exists)")
            return
        logger.info("Stale method_signatures.csv detected; regenerating")
        output_csv.unlink()

    src_dir = get_src_dir(repo.project, repo.bug_id, dataset=dataset)
    entities = []

    if normalize_benchmark_name(dataset) == "BIP":
        py_files = find_python_files(src_dir, exclude_tests=True)
        logger.info(
            "Extracting corpus method entities for %s-%s: %d Python files in %s",
            repo.project,
            repo.bug_id,
            len(py_files),
            src_dir,
        )
        for py_file in py_files:
            module = module_path_for_python(py_file, src_dir)
            for m in extract_methods_from_python(py_file, module=module, source_root=src_dir):
                entities.append(method_entity_from_python_method_info(m, module, src_root=src_dir))
    else:
        java_files = find_java_files(src_dir, exclude_tests=True)
        logger.info(
            "Extracting corpus method entities for %s-%d: %d Java files in %s",
            repo.project,
            repo.bug_id,
            len(java_files),
            src_dir,
        )
        for java_file in java_files:
            for m in extract_methods_from_java(java_file):
                entities.append(method_entity_from_method_info(m, src_root=src_dir))

    write_method_entities_csv(output_csv, entities)
    logger.info("Corpus method entities written to %s (%d methods)", output_csv, len(entities))


def run_gzoltar_coverage(repo: D4JRepo) -> None:
    """Run GZoltar coverage collection for *repo*.

    Produces ``gzoltar/coverage/coverage.ser`` under *repo.output_dir*.
    """
    agent_jar, cli_jar = ensure_java_utils_in_workspace()

    build_dir = to_container_path(repo.get_bin_class_dir())
    cp_test = to_container_path(repo.get_cp_test())

    # Ensure test list exists
    junit_tests_path = repo.output_dir / "junit_tests.txt"
    if not junit_tests_path.exists():
        repo.get_relevant_test_methods()
    test_file = to_container_path(junit_tests_path)

    # coverage.ser output location
    gzoltar_cov_dir = repo.output_dir / "gzoltar" / "coverage"
    gzoltar_cov_dir.mkdir(parents=True, exist_ok=True)
    gzoltar_ser = gzoltar_cov_dir / "coverage.ser"
    gzoltar_ser_c = to_container_path(gzoltar_ser)

    # Remove old coverage file if present
    if gzoltar_ser.exists():
        gzoltar_ser.unlink()

    # Build includes list from relevant classes
    relevant = repo.get_relevant_classes()
    includes = ":".join([cls + ":" for cls in relevant] + [cls + "$*:" for cls in relevant])

    classpath = f"{build_dir}:{D4J_JUNIT_CONTAINER}:{cp_test}:{cli_jar}"

    javaagent = (
        f"-javaagent:{agent_jar}="
        f"destfile={gzoltar_ser_c},"
        f"buildlocation={build_dir},"
        f"includes={includes},"
        f'excludes="",'
        f"inclnolocationclasses=false,"
        f"output=FILE"
    )

    command = [
        javaagent,
        "-cp",
        classpath,
        "com.gzoltar.cli.Main",
        "runTestMethods",
        "--testMethods",
        test_file,
        "--collectCoverage",
    ]

    logger.info(
        "Running GZoltar coverage for %s-%d",
        repo.project,
        repo.bug_id,
    )
    try:
        run_java_in_container(command, cwd=repo.repo_dir)
    except RuntimeError as exc:
        message = str(exc)
        if not _should_retry_without_access_fix_tests(message):
            raise

        filtered_file = _write_filtered_junit_tests(junit_tests_path)
        if filtered_file is None:
            raise

        logger.warning(
            (
                "Retrying GZoltar coverage for %s-%d after filtering AccessFixTest "
                "methods that trigger SecurityManager ClassCircularityError."
            ),
            repo.project,
            repo.bug_id,
        )

        command[command.index("--testMethods") + 1] = to_container_path(filtered_file)
        run_java_in_container(command, cwd=repo.repo_dir)

    if not gzoltar_ser.exists():
        raise RuntimeError(
            f"GZoltar coverage.ser not produced at {gzoltar_ser}. Check container logs for errors."
        )
    logger.info("GZoltar coverage written to %s", gzoltar_ser)


def _should_retry_without_access_fix_tests(error_message: str) -> bool:
    return (
        "ClassCircularityError" in error_message
        and "AccessFixTest$CauseBlockingSecurityManager" in error_message
    )


def _write_filtered_junit_tests(junit_tests_path: Path) -> Path | None:
    lines = [line for line in junit_tests_path.read_text().splitlines() if line.strip()]
    filtered_lines = [line for line in lines if "AccessFixTest" not in line]

    if len(filtered_lines) == len(lines):
        return None

    filtered_path = junit_tests_path.with_name("junit_tests.filtered.txt")
    filtered_path.write_text("\n".join(filtered_lines) + "\n")
    return filtered_path


def generate_fl_report(repo: D4JRepo) -> None:
    """Generate the GZoltar fault-localization report for *repo*.

    Produces ``sfl/sfl/txt/{spectra.csv, matrix.txt, tests.csv, ochiai.ranking.csv}``
    under *repo.output_dir*.
    """
    _agent_jar, cli_jar = ensure_java_utils_in_workspace()

    build_dir = to_container_path(repo.get_bin_class_dir())
    cp_test = to_container_path(repo.get_cp_test())

    gzoltar_ser = repo.output_dir / "gzoltar" / "coverage" / "coverage.ser"
    if not gzoltar_ser.exists():
        raise FileNotFoundError(
            f"coverage.ser not found at {gzoltar_ser}. Run run_gzoltar_coverage() first."
        )
    gzoltar_ser_c = to_container_path(gzoltar_ser)

    sfl_output_dir = repo.output_dir / "sfl"
    sfl_output_dir.mkdir(parents=True, exist_ok=True)
    sfl_output_dir_c = to_container_path(sfl_output_dir)

    classpath = f"{build_dir}:{D4J_JUNIT_CONTAINER}:{cp_test}:{cli_jar}"

    command = [
        "-cp",
        classpath,
        "com.gzoltar.cli.Main",
        "faultLocalizationReport",
        "--buildLocation",
        build_dir,
        "--granularity",
        "line",
        "--inclPublicMethods",
        "--inclStaticConstructors",
        "--inclDeprecatedMethods",
        "--dataFile",
        gzoltar_ser_c,
        "--outputDirectory",
        sfl_output_dir_c,
        "--family",
        "sfl",
        "--formula",
        "ochiai",
        "--metric",
        "entropy",
        "--formatter",
        "txt",
    ]

    logger.info(
        "Generating GZoltar FL report for %s-%d",
        repo.project,
        repo.bug_id,
    )
    run_java_in_container(command, cwd=repo.repo_dir)

    # Verify output files exist
    report_dir = repo.output_dir / SFL_SUBDIR
    required_files = ["spectra.csv", "matrix.txt", "tests.csv"]
    missing = [f for f in required_files if not (report_dir / f).exists()]
    if missing:
        raise RuntimeError(f"GZoltar report missing expected files in {report_dir}: {missing}")
    logger.info("GZoltar FL report written to %s", report_dir)


def run_gzoltar_pipeline(repo: D4JRepo, *, skip_existing: bool = True) -> None:
    """Run the full GZoltar pipeline: coverage + FL report.

    If *skip_existing* is True, skip steps whose output files already exist.
    """
    report_dir = repo.output_dir / SFL_SUBDIR
    required_files = ["spectra.csv", "matrix.txt", "tests.csv"]

    if skip_existing and all((report_dir / f).exists() for f in required_files):
        logger.info(
            "GZoltar outputs already exist for %s-%d, skipping.",
            repo.project,
            repo.bug_id,
        )
        return

    run_gzoltar_coverage(repo)
    generate_fl_report(repo)
