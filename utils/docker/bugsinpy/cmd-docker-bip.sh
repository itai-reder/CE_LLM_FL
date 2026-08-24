#!/usr/bin/env bash
# cmd-docker-bip.sh — Run an arbitrary command inside the BugsInPy container.
#
# Unlike cmd-docker4j.sh (which is a thin shim around `defects4j ...`), this
# wrapper is a generic passthrough — the full command line is the argument
# vector. This lets a single wrapper drive bugsinpy-*, pytest, conda, pip,
# bash one-liners, and the FauxPy CLI without separate scripts per tool.
#
# Usage:
#   utils/docker/bugsinpy/cmd-docker-bip.sh bugsinpy-info -p youtube-dl
#   utils/docker/bugsinpy/cmd-docker-bip.sh bugsinpy-checkout -p youtube-dl -v 0 -i 2 -w /workspace/repos
#   utils/docker/bugsinpy/cmd-docker-bip.sh bugsinpy-compile
#   utils/docker/bugsinpy/cmd-docker-bip.sh bugsinpy-test
#   utils/docker/bugsinpy/cmd-docker-bip.sh conda env list
#   utils/docker/bugsinpy/cmd-docker-bip.sh pytest tests/
#
# If invoked from inside data/BIP/ (the host workspace, overridable via
# CEFL_BIP_WORKSPACE), the container CWD is mapped accordingly so relative
# paths work.
#
# Path-preservation note: the inner shell is `bash -c` (non-login), with
# /opt/conda/etc/profile.d/conda.sh sourced explicitly. A login shell
# (`bash -lc`) re-runs conda's profile script which clobbers PATH and drops
# /BugsInPy/framework/bin, breaking `bugsinpy-*` lookups.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CEFL_ROOT="$(cd "${DOCKER_DIR}/../../.." && pwd)"
# CEFL_BIP_CONTAINER lets a parallel driver pin each worker lane to its own container
# (each container = its own conda namespace, so per-bug `conda create`/`pip install` cannot
# race across lanes). Default unchanged for all single-container usage.
CONTAINER_NAME="${CEFL_BIP_CONTAINER:-bugsinpy-cefl-container}"
WORKSPACE_DIR="${CEFL_BIP_WORKSPACE:-${CEFL_ROOT}/data/BIP}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is not installed or not in PATH" >&2
    exit 1
  fi
}

require_cmd docker

if [[ $# -eq 0 ]]; then
  echo "Error: no command supplied" >&2
  echo "Usage: $(basename "$0") <command> [args...]" >&2
  exit 2
fi

# Ensure container is running
"${DOCKER_DIR}/start-docker-bip.sh" >/dev/null

# Map host CWD to container CWD
host_cwd="$(pwd -P)"
workspace_real="$(cd "${WORKSPACE_DIR}" && pwd -P)"

if [[ "${host_cwd}" == "${workspace_real}" ]]; then
  container_cwd="/workspace"
elif [[ "${host_cwd}" == "${workspace_real}"/* ]]; then
  rel="${host_cwd#${workspace_real}}"
  container_cwd="/workspace${rel}"
else
  container_cwd="/workspace"
fi

# Escape arguments for safe shell transport
escaped_args=()
for arg in "$@"; do
  escaped_args+=("$(printf '%q' "${arg}")")
done

cmd="source /opt/conda/etc/profile.d/conda.sh && cd $(printf '%q' "${container_cwd}") && ${escaped_args[*]}"

docker_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  docker_flags=(-it)
fi

docker exec "${docker_flags[@]}" "${CONTAINER_NAME}" bash -c "${cmd}"
