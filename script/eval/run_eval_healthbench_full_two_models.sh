#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/qjh/miniconda3/envs/medicalgpt/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/qjh/llm_learning/base_model/qwen3_8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
GENERATOR_BATCH_SIZE="${GENERATOR_BATCH_SIZE:-8}"
GEN_GPU_0="${GEN_GPU_0:-0}"
GEN_GPU_1="${GEN_GPU_1:-1}"
SEED="${SEED:-42}"
STAGE_MAX_RETRIES="${STAGE_MAX_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-30}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)_healthbench_full_two_models}"

LOG_DIR="${PROJECT_ROOT}/evaluation/logs"
WRAPPER_LOG="${LOG_DIR}/${BATCH_ID}.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

SFT_RUN_NAME="${BATCH_ID}_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_full_all_seed${SEED}"
DPO_RUN_NAME="${BATCH_ID}_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_full_all_seed${SEED}"

SFT_ADAPTER="${PROJECT_ROOT}/outputs/sft/20260409_121822_qwen3-8b_huatuo-5w_lora_eval/checkpoints/checkpoint-75"
DPO_ADAPTER="${PROJECT_ROOT}/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/checkpoints/checkpoint-330"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${WRAPPER_LOG}"
}

extract_env_from_running_eval() {
  local key_name="$1"
  local pid
  pid="$(ps -eo pid,args | awk '/evaluation\/run_eval.py/ && /--mode judge_only/ && !/awk/ {print $1; exit}')"
  if [[ -z "${pid:-}" ]]; then
    pid="$(ps -eo pid,args | awk '/evaluation\/run_eval.py/ && !/awk/ {print $1; exit}')"
  fi
  if [[ -z "${pid:-}" ]]; then
    return 1
  fi

  KEY_NAME="${key_name}" TARGET_PID="${pid}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

key = os.environ["KEY_NAME"]
pid = os.environ["TARGET_PID"]
env_path = Path(f"/proc/{pid}/environ")
if not env_path.exists():
    raise SystemExit(1)
for item in env_path.read_bytes().split(b"\0"):
    prefix = f"{key}=".encode()
    if item.startswith(prefix):
        print(item.decode("utf-8", errors="ignore").split("=", 1)[1])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

ensure_openai_env() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="$(extract_env_from_running_eval OPENAI_API_KEY || true)"
    export OPENAI_API_KEY
  fi
  if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
    OPENAI_BASE_URL="$(extract_env_from_running_eval OPENAI_BASE_URL || true)"
    export OPENAI_BASE_URL
  fi

  if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
    log "missing OPENAI_API_KEY or OPENAI_BASE_URL"
    exit 1
  fi
}

prefetch_full_dataset() {
  log "prefetching HealthBench full dataset"
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" CACHE_DIR_ENV="${CACHE_DIR}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
from evaluation.benchmarks.healthbench import load_healthbench_examples

examples, cache_path = load_healthbench_examples(
    subset_name="full",
    cache_root=Path(os.environ["CACHE_DIR_ENV"]),
    max_examples=-1,
    sampling_mode="sequential",
    seed=42,
    shuffle=False,
    per_theme_examples=-1,
)
print(f"examples={len(examples)} cache_path={cache_path}")
PY
}

run_stage() {
  local mode="$1"
  local gpu="$2"
  local model_alias="$3"
  local adapter_path="$4"
  local run_name="$5"

  local -a cmd=(
    "${PYTHON_BIN}"
    "${PROJECT_ROOT}/evaluation/run_eval.py"
    --benchmark healthbench
    --subset-name full
    --mode "${mode}"
    --judge-mode $([[ "${mode}" == "judge_only" ]] && echo openai || echo none)
    --judge-model "${JUDGE_MODEL}"
    --model-name-or-path "${MODEL_PATH}"
    --model-alias "${model_alias}"
    --adapter-path "${adapter_path}"
    --cache-dir "${CACHE_DIR}"
    --output-root "${OUTPUT_ROOT}"
    --run-name "${run_name}"
    --max-examples -1
    --sampling-mode sequential
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
  )

  if [[ "${mode}" == "generate_only" ]]; then
    cmd+=(--generator-device cuda:0 --generator-batch-size "${GENERATOR_BATCH_SIZE}")
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  else
    CUDA_VISIBLE_DEVICES="" "${cmd[@]}"
  fi
}

retry_stage() {
  local label="$1"
  shift
  local attempt=1
  local exit_code=0
  while (( attempt <= STAGE_MAX_RETRIES )); do
    if (( attempt > 1 )); then
      log "retry ${attempt}/${STAGE_MAX_RETRIES} for ${label} after sleep=${RETRY_SLEEP_SECONDS}s"
      sleep "${RETRY_SLEEP_SECONDS}"
    fi

    set +e
    "$@"
    exit_code=$?
    set -e

    if [[ "${exit_code}" -eq 0 ]]; then
      log "${label} succeeded on attempt ${attempt}/${STAGE_MAX_RETRIES}"
      return 0
    fi

    log "${label} failed on attempt ${attempt}/${STAGE_MAX_RETRIES} exit=${exit_code}"
    attempt=$((attempt + 1))
  done

  return "${exit_code}"
}

wait_for_other_judges() {
  while ps -eo pid,args | awk '/evaluation\/run_eval.py/ && /--mode judge_only/ && !/awk/ {found=1} END {exit found ? 0 : 1}'; do
    log "another judge_only task is still running; sleep 60s before starting next full judge stage"
    sleep 60
  done
}

main() {
  log "wrapper_log=${WRAPPER_LOG}"
  log "batch_id=${BATCH_ID}"
  log "sft_run_name=${SFT_RUN_NAME}"
  log "dpo_run_name=${DPO_RUN_NAME}"
  log "generator_batch_size=${GENERATOR_BATCH_SIZE}"
  log "stage_max_retries=${STAGE_MAX_RETRIES}"
  log "retry_sleep_seconds=${RETRY_SLEEP_SECONDS}"

  ensure_openai_env
  prefetch_full_dataset >> "${WRAPPER_LOG}" 2>&1

  log "starting parallel full generation for two models"
  (
    retry_stage \
      "generate sft_5w_ckpt75" \
      run_stage generate_only "${GEN_GPU_0}" qwen3_8b_huatuo_5w_ckpt75 "${SFT_ADAPTER}" "${SFT_RUN_NAME}"
  ) >> "${WRAPPER_LOG}" 2>&1 &
  local gen_pid_0=$!

  (
    retry_stage \
      "generate dpo_v2_ckpt330" \
      run_stage generate_only "${GEN_GPU_1}" qwen3_8b_dpo_v2_ckpt330 "${DPO_ADAPTER}" "${DPO_RUN_NAME}"
  ) >> "${WRAPPER_LOG}" 2>&1 &
  local gen_pid_1=$!

  wait "${gen_pid_0}"
  wait "${gen_pid_1}"
  log "parallel generation finished for both models"

  wait_for_other_judges

  log "starting judge for sft_5w_ckpt75"
  retry_stage \
    "judge sft_5w_ckpt75" \
    run_stage judge_only "-" qwen3_8b_huatuo_5w_ckpt75 "${SFT_ADAPTER}" "${SFT_RUN_NAME}" \
    >> "${WRAPPER_LOG}" 2>&1

  log "starting judge for dpo_v2_ckpt330"
  retry_stage \
    "judge dpo_v2_ckpt330" \
    run_stage judge_only "-" qwen3_8b_dpo_v2_ckpt330 "${DPO_ADAPTER}" "${DPO_RUN_NAME}" \
    >> "${WRAPPER_LOG}" 2>&1

  log "all full evaluation stages finished"
}

main "$@"
