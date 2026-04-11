#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/qjh/miniconda3/envs/medicalgpt/bin/python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SPLIT="${SPLIT:-train}"
MODEL="${MODEL:-gpt-5.2}"
INPUT_FILE="${INPUT_FILE:-${PROJECT_ROOT}/data/alignment/raw/dpo/medical_pairwise_${SPLIT}.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/alignment/reconstructed/dpo_v2}"
OUTPUT_NAME="${OUTPUT_NAME:-medical_pairwise_${SPLIT}_v2}"
BASE_URL="${OPENAI_BASE_URL:-}"
API_KEY="${OPENAI_API_KEY:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
MAX_RETRIES="${MAX_RETRIES:-8}"
MIN_REQUEST_INTERVAL="${MIN_REQUEST_INTERVAL:-0.2}"
PROGRESS_LOG_INTERVAL="${PROGRESS_LOG_INTERVAL:-10}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
OVERWRITE="${OVERWRITE:-0}"

RUN_LOG_DIR="${PROJECT_ROOT}/outputs/logs/dpo_reconstruct"
RUN_LOG_PATH="${RUN_LOG_DIR}/${TIMESTAMP}_${OUTPUT_NAME}_${SPLIT}_serial.log"

mkdir -p "${RUN_LOG_DIR}"

if [[ -z "${BASE_URL}" ]]; then
  echo "OPENAI_BASE_URL is required" >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  echo "OPENAI_API_KEY is required" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/script/alignment/reconstruct_dpo_dataset.py"
  --input-file "${INPUT_FILE}"
  --split "${SPLIT}"
  --output-name "${OUTPUT_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --model "${MODEL}"
  --strict-serial
  --timeout-seconds "${TIMEOUT_SECONDS}"
  --max-retries "${MAX_RETRIES}"
  --min-request-interval "${MIN_REQUEST_INTERVAL}"
  --progress-log-interval "${PROGRESS_LOG_INTERVAL}"
)

if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

{
  echo "timestamp=${TIMESTAMP}"
  echo "split=${SPLIT}"
  echo "input_file=${INPUT_FILE}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "output_name=${OUTPUT_NAME}"
  echo "model=${MODEL}"
  echo "strict_serial=true"
  echo "min_request_interval=${MIN_REQUEST_INTERVAL}"
  echo "progress_log_interval=${PROGRESS_LOG_INTERVAL}"
  echo "max_samples=${MAX_SAMPLES}"
  echo "overwrite=${OVERWRITE}"
  echo "run_log_path=${RUN_LOG_PATH}"
} | while IFS= read -r line; do
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
done | tee -a "${RUN_LOG_PATH}"

export OPENAI_BASE_URL="${BASE_URL}"
export OPENAI_API_KEY="${API_KEY}"

"${CMD[@]}" 2>&1 | while IFS= read -r line; do
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
done | tee -a "${RUN_LOG_PATH}"
