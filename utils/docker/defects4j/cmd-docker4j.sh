#!/usr/bin/env bash
# cmd-docker4j.sh — Run a defects4j command inside the container.
#
# Usage:
#   utils/docker/defects4j/cmd-docker4j.sh info -p Lang
#   utils/docker/defects4j/cmd-docker4j.sh checkout -p Lang -v 1b -w /workspace/repos/Lang/1
#
# If invoked from inside data/D4J/ (the host workspace), the container CWD is
# mapped accordingly so relative paths work.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CEFL_ROOT="$(cd "${DOCKER_DIR}/../../.." && pwd)"
CONTAINER_NAME="${CEFL_D4J_CONTAINER:-defects4j-cefl-container}"
WORKSPACE_DIR="${CEFL_D4J_WORKSPACE:-${CEFL_ROOT}/data/D4J}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is not installed or not in PATH" >&2
    exit 1
  fi
}

require_cmd docker

# Ensure container is running
"${DOCKER_DIR}/start-docker4j.sh" >/dev/null

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

if [[ ${#escaped_args[@]} -eq 0 ]]; then
  cmd="cd $(printf '%q' "${container_cwd}") && defects4j"
else
  cmd="cd $(printf '%q' "${container_cwd}") && defects4j ${escaped_args[*]}"
fi

docker_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  docker_flags=(-it)
fi

docker exec "${docker_flags[@]}" "${CONTAINER_NAME}" bash -lc "${cmd}"
