#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/qjh/miniconda3/envs/medicalgpt/bin/python}"

TRAIN_SIZE="${TRAIN_SIZE:-3000}"
VALID_SIZE="${VALID_SIZE:-300}"
SEED="${SEED:-42}"
PRESET="${PRESET:-v1_emergency_context}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/alignment/grpo/v1_emergency_context}"
OUTPUT_NAME="${OUTPUT_NAME:-medical_grpo_prompt_v1_emergency_context}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/script/grpo/build_grpo_prompt_dataset.py" \
  --preset "${PRESET}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-name "${OUTPUT_NAME}" \
  --train-size "${TRAIN_SIZE}" \
  --valid-size "${VALID_SIZE}" \
  --seed "${SEED}"
