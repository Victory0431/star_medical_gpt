#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/qjh/llm_learning/my_medical_gpt"
PYTHON_BIN="/home/qjh/miniconda3/envs/medicalgpt/bin/python"
MODEL_PATH="${MODEL_PATH:-/home/qjh/llm_learning/base_model/qwen3_8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_healthbench_consensus_qwen3_8b_base}"
BENCHMARK="${BENCHMARK:-healthbench}"
SUBSET_NAME="${SUBSET_NAME:-consensus}"
MODE="${MODE:-generate_only}"
JUDGE_MODE="${JUDGE_MODE:-none}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"
GENERATOR_DEVICE="${GENERATOR_DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
CONSOLE_DIR="${OUTPUT_ROOT}/${RUN_NAME}/logs"
CONSOLE_LOG="${CONSOLE_DIR}/console.log"
ARGS=()

mkdir -p "${CONSOLE_DIR}"

{
  echo "run_name=${RUN_NAME}"
  echo "benchmark=${BENCHMARK}"
  echo "subset_name=${SUBSET_NAME}"
  echo "mode=${MODE}"
  echo "judge_mode=${JUDGE_MODE}"
  echo "judge_model=${JUDGE_MODEL}"
  echo "model_path=${MODEL_PATH}"
  echo "max_examples=${MAX_EXAMPLES}"
  echo "generator_device=${GENERATOR_DEVICE}"
  echo "enable_thinking=${ENABLE_THINKING}"
} | while IFS= read -r line; do
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
done | tee -a "${CONSOLE_LOG}"

if [[ "${ENABLE_THINKING}" == "1" ]]; then
  ARGS+=(--enable-thinking)
fi

"${PYTHON_BIN}" \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --benchmark "${BENCHMARK}" \
  --subset-name "${SUBSET_NAME}" \
  --mode "${MODE}" \
  --judge-mode "${JUDGE_MODE}" \
  --judge-model "${JUDGE_MODEL}" \
  --model-name-or-path "${MODEL_PATH}" \
  --model-alias "qwen3_8b_base" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --cache-dir "${PROJECT_ROOT}/cache" \
  --max-examples "${MAX_EXAMPLES}" \
  --generator-device "${GENERATOR_DEVICE}" \
  --dtype "${DTYPE}" \
  "${ARGS[@]}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  2>&1 | while IFS= read -r line; do
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
  done | tee -a "${CONSOLE_LOG}"
