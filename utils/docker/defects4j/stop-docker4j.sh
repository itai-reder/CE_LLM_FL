#!/usr/bin/env bash
# stop-docker4j.sh — Stop and remove the Defects4J container.
set -euo pipefail

CONTAINER_NAME="${CEFL_D4J_CONTAINER:-defects4j-cefl-container}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is not installed or not in PATH" >&2
    exit 1
  fi
}

require_cmd docker

if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} does not exist. Nothing to stop."
  exit 0
fi

echo "Stopping and removing ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null
echo "Done"
