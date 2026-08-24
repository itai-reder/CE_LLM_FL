#!/usr/bin/env bash
# start-docker-bip.sh — Ensure the BugsInPy container is running.
# If the container does not exist yet, delegates to setup-docker-bip.sh.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# See cmd-docker-bip.sh for CEFL_BIP_CONTAINER (per-lane container isolation). Default unchanged.
CONTAINER_NAME="${CEFL_BIP_CONTAINER:-bugsinpy-cefl-container}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is not installed or not in PATH" >&2
    exit 1
  fi
}

require_cmd docker

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} is already running"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Starting existing container ${CONTAINER_NAME}"
  docker start "${CONTAINER_NAME}" >/dev/null
  echo "Container ${CONTAINER_NAME} started"
  exit 0
fi

echo "Container ${CONTAINER_NAME} missing; running setup first"
"${DOCKER_DIR}/setup-docker-bip.sh"
