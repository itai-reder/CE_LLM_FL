#!/usr/bin/env bash
# setup-docker-bip.sh — Clone BugsInPy + fauxpy-experiments, build the Docker image, and create the container.
#
# Usage:
#   utils/docker/bugsinpy/setup-docker-bip.sh              # first-time setup
#   utils/docker/bugsinpy/setup-docker-bip.sh --rebuild    # tear down and rebuild from scratch
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CEFL_ROOT="$(cd "${DOCKER_DIR}/../../.." && pwd)"

IMAGE_TAG="bugsinpy:cefl"
# CEFL_BIP_CONTAINER lets a parallel driver create one container per worker lane from the
# shared image (each lane = its own conda namespace). Default unchanged for single-container use.
CONTAINER_NAME="${CEFL_BIP_CONTAINER:-bugsinpy-cefl-container}"
WORKSPACE_DIR="${CEFL_BIP_WORKSPACE:-${CEFL_ROOT}/data/BIP}"
DOCKERFILE_PATH="${DOCKER_DIR}/Dockerfile"
BIP_CLONE_DIR="${DOCKER_DIR}/BugsInPy"
FAUXPY_EXP_CLONE_DIR="${DOCKER_DIR}/fauxpy-experiments"
BIP_REPO_URL="https://github.com/reproducing-research-projects/BugsInPy"
FAUXPY_EXP_REPO_URL="https://github.com/atom-sw/fauxpy-experiments"
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

# ---------- Clone BugsInPy if not present ----------

if [[ ! -d "${BIP_CLONE_DIR}" ]]; then
  echo "Cloning BugsInPy into ${BIP_CLONE_DIR}"
  git clone "${BIP_REPO_URL}" "${BIP_CLONE_DIR}"
else
  echo "BugsInPy already cloned at ${BIP_CLONE_DIR}"
fi

# ---------- Clone fauxpy-experiments (for the pytest-FauxPy snapshot) if not present ----------

if [[ ! -d "${FAUXPY_EXP_CLONE_DIR}" ]]; then
  echo "Cloning fauxpy-experiments into ${FAUXPY_EXP_CLONE_DIR}"
  git clone "${FAUXPY_EXP_REPO_URL}" "${FAUXPY_EXP_CLONE_DIR}"
else
  echo "fauxpy-experiments already cloned at ${FAUXPY_EXP_CLONE_DIR}"
fi

if [[ ! -d "${FAUXPY_EXP_CLONE_DIR}/pytest-FauxPy" ]]; then
  echo "Error: ${FAUXPY_EXP_CLONE_DIR}/pytest-FauxPy not found; the fauxpy-experiments clone appears incomplete" >&2
  exit 1
fi

# ---------- Create host workspace ----------

mkdir -p "${WORKSPACE_DIR}"

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
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    -v "${WORKSPACE_DIR}:/workspace" \
    "${IMAGE_TAG}" \
    /bin/bash -lc "tail -f /dev/null" >/dev/null
fi

echo "Ready: ${CONTAINER_NAME} (${IMAGE_TAG})"
