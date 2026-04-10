#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/qjh/llm_learning/base_model/qwen3_8B}"
export ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_ROOT}/outputs/sft/20260408_222930_qwen3-8b_medical-sft-1k_lora_clean/final_model}"
export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_healthbench_consensus_qwen3_8b_huatuo_1k_lora}"

PYTHON_BIN="/home/qjh/miniconda3/envs/medicalgpt/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval}"
BENCHMARK="${BENCHMARK:-healthbench}"
SUBSET_NAME="${SUBSET_NAME:-consensus}"
MODE="${MODE:-generate_only}"
JUDGE_MODE="${JUDGE_MODE:-none}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
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
  echo "adapter_path=${ADAPTER_PATH}"
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
  "${PROJECT_ROOT}/evaluation/run_eval.py" \
  --benchmark "${BENCHMARK}" \
  --subset-name "${SUBSET_NAME}" \
  --mode "${MODE}" \
  --judge-mode "${JUDGE_MODE}" \
  --judge-model "${JUDGE_MODEL}" \
  --model-name-or-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --model-alias "qwen3_8b_huatuo_1k_lora" \
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
