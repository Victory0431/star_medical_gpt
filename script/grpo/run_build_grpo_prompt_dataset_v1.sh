#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/qjh/miniconda3/envs/medicalgpt/bin/python}"

TRAIN_SIZE="${TRAIN_SIZE:-3000}"
VALID_SIZE="${VALID_SIZE:-300}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/script/grpo/build_grpo_prompt_dataset.py" \
  --train-size "${TRAIN_SIZE}" \
  --valid-size "${VALID_SIZE}" \
  --seed "${SEED}"
