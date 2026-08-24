#!/usr/bin/env bash
# setup-docker4j.sh — Clone Defects4J, build the Docker image, and create the container.
#
# Usage:
#   utils/docker/defects4j/setup-docker4j.sh              # first-time setup
#   utils/docker/defects4j/setup-docker4j.sh --rebuild    # tear down and rebuild from scratch
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CEFL_ROOT="$(cd "${DOCKER_DIR}/../../.." && pwd)"

IMAGE_TAG="defects4j:cefl"
CONTAINER_NAME="${CEFL_D4J_CONTAINER:-defects4j-cefl-container}"
WORKSPACE_DIR="${CEFL_D4J_WORKSPACE:-${CEFL_ROOT}/data/D4J}"
GZOLTAR_DIR="${CEFL_ROOT}/utils/java"
DOCKERFILE_PATH="${DOCKER_DIR}/Dockerfile"
D4J_CLONE_DIR="${DOCKER_DIR}/defects4j"
REBUILD="${1:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is not installed or not in PATH" >&2
    exit 1
  fi
}

require_cmd docker
require_cmd git

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
  echo "Error: Dockerfile not found at ${DOCKERFILE_PATH}" >&2
  exit 1
fi

# ---------- Clone Defects4J if not present ----------

if [[ ! -d "${D4J_CLONE_DIR}" ]]; then
  echo "Cloning Defects4J into ${D4J_CLONE_DIR}"
  git clone https://github.com/rjust/defects4j.git "${D4J_CLONE_DIR}"
else
  echo "Defects4J already cloned at ${D4J_CLONE_DIR}"
fi

# ---------- Create host directories ----------

mkdir -p "${WORKSPACE_DIR}"
mkdir -p "${GZOLTAR_DIR}"

# ---------- Handle --rebuild ----------

if [[ "${REBUILD}" == "--rebuild" ]]; then
  echo "Rebuild requested: removing old image/container if present"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker image rm "${IMAGE_TAG}" >/dev/null 2>&1 || true
fi

# ---------- Build image ----------

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "Building ${IMAGE_TAG} from ${DOCKERFILE_PATH}"
  docker build -f "${DOCKERFILE_PATH}" -t "${IMAGE_TAG}" "${DOCKER_DIR}"
else
  echo "Image ${IMAGE_TAG} already exists"
fi

# ---------- Create / start container ----------

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
  else
    echo "Starting existing container ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
else
  echo "Creating and starting container ${CONTAINER_NAME}"
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -e TZ=America/Los_Angeles \
    -e JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF-8 \
    -e ANT_OPTS=-Dfile.encoding=UTF-8 \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    -v "${WORKSPACE_DIR}:/workspace" \
    -v "${GZOLTAR_DIR}:/gzoltar" \
    "${IMAGE_TAG}" \
    /bin/bash -lc "tail -f /dev/null" >/dev/null
fi

echo "Ready: ${CONTAINER_NAME} (${IMAGE_TAG})"
