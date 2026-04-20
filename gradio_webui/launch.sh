#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/qjh/miniconda3/envs/medicalgpt/bin/python}"
SERVER_NAME="${SERVER_NAME:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-7860}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" \
  --server-name "${SERVER_NAME}" \
  --server-port "${SERVER_PORT}" \
  "$@"
