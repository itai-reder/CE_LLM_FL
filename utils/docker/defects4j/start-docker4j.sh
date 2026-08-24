#!/usr/bin/env bash
# start-docker4j.sh — Ensure the Defects4J container is running.
# If the container does not exist yet, delegates to setup-docker4j.sh.
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CEFL_D4J_CONTAINER:-defects4j-cefl-container}"

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
"${DOCKER_DIR}/setup-docker4j.sh"
