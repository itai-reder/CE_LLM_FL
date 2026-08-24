"""Docker helpers for running commands inside the Defects4J container.

Provides path translation between host and container filesystems,
and a thin wrapper around ``docker exec`` for running Defects4J / Java
commands inside the container.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from src.common.config import (
    CONTAINER_NAME,
    CONTAINER_WORKSPACE,
    GZOLTAR_AGENT_JAR,
    GZOLTAR_CLI_JAR,
    get_benchmark_data_dir,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path translation
# ---------------------------------------------------------------------------


def to_container_path(host_path: str | Path) -> str:
    """Translate a host path under ``data/D4J/`` to the container equivalent.

    Example::

        /home/user/git/CEFL/data/D4J/repos/Chart/1
        → /workspace/repos/Chart/1
    """
    host_str = str(Path(host_path).resolve())
    data_str = str(get_benchmark_data_dir().resolve())
    if host_str.startswith(data_str):
        rel = host_str[len(data_str) :]
        return CONTAINER_WORKSPACE + rel
    return host_str


def translate_cmd_paths(cmd: list[str]) -> list[str]:
    """Translate absolute host paths in a command list to container paths."""
    data_prefix = str(get_benchmark_data_dir().resolve())
    translated: list[str] = []
    for arg in cmd:
        if isinstance(arg, str) and arg.startswith(data_prefix):
            translated.append(to_container_path(arg))
        else:
            translated.append(str(arg))
    return translated


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

_git_safety_configured = False


def is_container_running() -> bool:
    """Return True if the Defects4J container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return CONTAINER_NAME in result.stdout.strip().splitlines()
    except FileNotFoundError:
        return False


def _ensure_git_safety() -> None:
    """Configure ``safe.directory = *`` inside the container (once)."""
    global _git_safety_configured
    if _git_safety_configured:
        return
    result = subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "0:0",
            CONTAINER_NAME,
            "bash",
            "-lc",
            "git config --system --replace-all safe.directory '*' "
            "|| git config --system --add safe.directory '*'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to configure git safe.directory in container: {result.stderr.strip()}"
        )
    _git_safety_configured = True


def _host_user_flag() -> str:
    """Return ``uid:gid`` for the current host user."""
    return f"{os.getuid()}:{os.getgid()}"


def ensure_project_dir_layout_writable(project: str) -> None:
    """Ensure Defects4J ``dir-layout.csv`` is writable inside the container.

    Some project checkout hooks append to this file. If the file is owned by
    root and the command runs as a non-root UID, checkout may fail.
    """
    layout_path = f"/defects4j/framework/projects/{project}/dir-layout.csv"
    result = docker_exec(
        ["chmod", "a+w", layout_path],
        user="0:0",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to make {layout_path} writable in container: {result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


def docker_exec(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    user: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run *command* inside the Defects4J container via ``docker exec``.

    Parameters
    ----------
    command:
        The command + arguments to execute (already container-path translated).
    cwd:
        Working directory inside the container.  If a host path is given it is
        translated automatically.
    user:
        ``-u`` flag value (e.g. ``"0:0"``).  Defaults to current host user.
    check:
        If True, raise ``RuntimeError`` on non-zero exit code.
    """
    _ensure_git_safety()

    full_cmd: list[str] = ["docker", "exec"]
    u = user or _host_user_flag()
    full_cmd += ["-u", u]

    if cwd is not None:
        container_cwd = to_container_path(cwd)
        full_cmd += ["-w", container_cwd]

    full_cmd += [
        CONTAINER_NAME,
        "env",
        "GIT_CONFIG_COUNT=1",
        "GIT_CONFIG_KEY_0=safe.directory",
        "GIT_CONFIG_VALUE_0=*",
    ]
    full_cmd += command

    logger.debug("CMD: %s", " ".join(full_cmd))
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"docker exec failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(full_cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


def run_defects4j(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    user: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``defects4j <args>`` inside the container.

    Host-side absolute paths in *args* are translated automatically.
    """
    translated = translate_cmd_paths(args)
    return docker_exec(["defects4j", *translated], cwd=cwd, user=user, check=check)


def run_java_in_container(
    java_args: list[str],
    *,
    cwd: str | Path | None = None,
    user: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``java <java_args>`` inside the container.

    Host-side absolute paths in *java_args* are translated automatically.
    """
    translated = translate_cmd_paths(java_args)
    return docker_exec(["java", *translated], cwd=cwd, user=user, check=check)


# ---------------------------------------------------------------------------
# GZoltar JAR helpers
# ---------------------------------------------------------------------------


def ensure_java_utils_in_workspace() -> tuple[str, str]:
    """Copy GZoltar JARs into ``data/D4J/deps/`` and return container paths.

    Returns
    -------
    (gz_agent_path, gz_cli_path)
    """
    import shutil

    deps_host = get_benchmark_data_dir() / "deps"
    deps_host.mkdir(parents=True, exist_ok=True)
    container_deps = CONTAINER_WORKSPACE + "/deps"

    for jar in (GZOLTAR_AGENT_JAR, GZOLTAR_CLI_JAR):
        if not jar.exists():
            raise FileNotFoundError(
                f"JAR not found: {jar}\nPlace the JARs in utils/java/ (see utils/build_gzoltar.sh)."
            )
        dst = deps_host / jar.name
        if not dst.exists() or jar.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(jar, dst)

    agent_path = f"{container_deps}/{GZOLTAR_AGENT_JAR.name}"
    cli_path = f"{container_deps}/{GZOLTAR_CLI_JAR.name}"
    return agent_path, cli_path


D4J_JUNIT_CONTAINER = "/defects4j/framework/projects/lib/junit-4.12-hamcrest-1.3.jar"
