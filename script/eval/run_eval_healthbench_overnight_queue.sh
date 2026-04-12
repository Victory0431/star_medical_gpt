#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/qjh/miniconda3/envs/medicalgpt/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/qjh/llm_learning/base_model/qwen3_8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"
BENCHMARK="${BENCHMARK:-healthbench}"
SUBSET_NAME="${SUBSET_NAME:-consensus}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
SAMPLING_MODE="${SAMPLING_MODE:-stratified_theme}"
PER_THEME_EXAMPLES="${PER_THEME_EXAMPLES:-100}"
THEME_COUNT="${THEME_COUNT:-7}"
MAX_EXAMPLES="${MAX_EXAMPLES:-$((PER_THEME_EXAMPLES * THEME_COUNT))}"
GENERATOR_BATCH_SIZE="${GENERATOR_BATCH_SIZE:-8}"
GPU_LIST="${GPU_LIST:-0,1}"
JUDGE_PARALLELISM="${JUDGE_PARALLELISM:-1}"
SEED="${SEED:-42}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
STAGE_MAX_RETRIES="${STAGE_MAX_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-15}"

SCHEDULER_LOG_DIR="${PROJECT_ROOT}/evaluation/logs"
SCHEDULER_LOG="${SCHEDULER_LOG_DIR}/${BATCH_ID}_healthbench_theme${PER_THEME_EXAMPLES}x${THEME_COUNT}_queue.log"
MANIFEST_PATH="${OUTPUT_ROOT}/${BATCH_ID}_healthbench_theme${PER_THEME_EXAMPLES}x${THEME_COUNT}_manifest.tsv"

mkdir -p "${SCHEDULER_LOG_DIR}" "${OUTPUT_ROOT}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${SCHEDULER_LOG}"
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    log "missing required env: ${name}"
    exit 1
  fi
}

split_csv() {
  local csv="$1"
  local -n target_ref="$2"
  IFS=',' read -r -a target_ref <<< "${csv}"
}

remove_pid_from_list() {
  local target_pid="$1"
  shift
  local -a source=("$@")
  local -a filtered=()
  local pid
  for pid in "${source[@]}"; do
    if [[ "${pid}" != "${target_pid}" ]]; then
      filtered+=("${pid}")
    fi
  done
  if (( ${#filtered[@]} > 0 )); then
    printf '%s\n' "${filtered[@]}"
  fi
}

build_run_name() {
  local alias="$1"
  echo "${BATCH_ID}_healthbench_${alias}_${JUDGE_MODEL//./}_consensus_theme${PER_THEME_EXAMPLES}x${THEME_COUNT}_seed${SEED}"
}

append_manifest_row() {
  local label="$1"
  local alias="$2"
  local adapter_path="$3"
  local run_name="$4"
  printf '%s\t%s\t%s\t%s\n' "${label}" "${alias}" "${adapter_path}" "${run_name}" >> "${MANIFEST_PATH}"
}

run_eval_stage() {
  local stage="$1"
  local gpu="$2"
  local label="$3"
  local alias="$4"
  local adapter_path="$5"
  local run_name="$6"

  local -a cmd=(
    "${PYTHON_BIN}"
    "${PROJECT_ROOT}/evaluation/run_eval.py"
    --benchmark "${BENCHMARK}"
    --subset-name "${SUBSET_NAME}"
    --judge-model "${JUDGE_MODEL}"
    --model-name-or-path "${MODEL_PATH}"
    --model-alias "${alias}"
    --output-root "${OUTPUT_ROOT}"
    --run-name "${run_name}"
    --cache-dir "${CACHE_DIR}"
    --max-examples "${MAX_EXAMPLES}"
    --sampling-mode "${SAMPLING_MODE}"
    --per-theme-examples "${PER_THEME_EXAMPLES}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
  )

  if [[ -n "${adapter_path}" ]]; then
    cmd+=(--adapter-path "${adapter_path}")
  fi

  if [[ "${stage}" == "generate" ]]; then
    cmd+=(
      --mode generate_only
      --judge-mode none
      --generator-device cuda:0
      --generator-batch-size "${GENERATOR_BATCH_SIZE}"
    )
  else
    cmd+=(
      --mode judge_only
      --judge-mode openai
    )
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${stage}" == "generate" ]]; then
      log "[dry-run][generate][gpu${gpu}][${label}] CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    else
      log "[dry-run][judge][${label}] ${cmd[*]}"
    fi
    return 0
  fi

  if [[ "${stage}" == "generate" ]]; then
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | while IFS= read -r line; do
        printf '[%s] [generate][gpu%s][%s] %s\n' "$(timestamp)" "${gpu}" "${label}" "${line}"
      done
    ) >> "${SCHEDULER_LOG}" 2>&1
  else
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="" "${cmd[@]}" 2>&1 | while IFS= read -r line; do
        printf '[%s] [judge][%s] %s\n' "$(timestamp)" "${label}" "${line}"
      done
    ) >> "${SCHEDULER_LOG}" 2>&1
  fi
}

run_eval_stage_with_retries() {
  local stage="$1"
  local gpu="$2"
  local label="$3"
  local alias="$4"
  local adapter_path="$5"
  local run_name="$6"

  local attempt=1
  local exit_code=0
  while (( attempt <= STAGE_MAX_RETRIES )); do
    if (( attempt > 1 )); then
      log "retry ${attempt}/${STAGE_MAX_RETRIES} for ${stage} ${label} after sleep=${RETRY_SLEEP_SECONDS}s"
      sleep "${RETRY_SLEEP_SECONDS}"
    fi

    set +e
    run_eval_stage "${stage}" "${gpu}" "${label}" "${alias}" "${adapter_path}" "${run_name}"
    exit_code=$?
    set -e

    if [[ "${exit_code}" -eq 0 ]]; then
      return 0
    fi

    log "${stage} attempt ${attempt}/${STAGE_MAX_RETRIES} failed for ${label} -> ${run_name} (exit=${exit_code})"
    attempt=$((attempt + 1))
  done

  return "${exit_code}"
}

launch_generate() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r label alias adapter_path <<< "${spec}"
  local run_name
  run_name="$(build_run_name "${alias}")"

  log "queue generate on gpu${gpu}: ${label} -> ${run_name}"
  append_manifest_row "${label}" "${alias}" "${adapter_path}" "${run_name}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_eval_stage generate "${gpu}" "${label}" "${alias}" "${adapter_path}" "${run_name}"
    return 0
  fi

  (
    run_eval_stage_with_retries generate "${gpu}" "${label}" "${alias}" "${adapter_path}" "${run_name}"
  ) &
  local pid=$!

  ACTIVE_GEN_PIDS+=("${pid}")
  GEN_PID_TO_GPU["${pid}"]="${gpu}"
  GEN_PID_TO_LABEL["${pid}"]="${label}"
  GEN_PID_TO_ALIAS["${pid}"]="${alias}"
  GEN_PID_TO_ADAPTER["${pid}"]="${adapter_path}"
  GEN_PID_TO_RUN["${pid}"]="${run_name}"
}

launch_judge() {
  local label="$1"
  local alias="$2"
  local adapter_path="$3"
  local run_name="$4"

  log "queue judge: ${label} -> ${run_name}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_eval_stage judge "-" "${label}" "${alias}" "${adapter_path}" "${run_name}"
    return 0
  fi

  (
    run_eval_stage_with_retries judge "-" "${label}" "${alias}" "${adapter_path}" "${run_name}"
  ) &
  local pid=$!

  ACTIVE_JUDGE_PIDS+=("${pid}")
  JUDGE_PID_TO_LABEL["${pid}"]="${label}"
  JUDGE_PID_TO_ALIAS["${pid}"]="${alias}"
  JUDGE_PID_TO_ADAPTER["${pid}"]="${adapter_path}"
  JUDGE_PID_TO_RUN["${pid}"]="${run_name}"
}

maybe_launch_next_generate() {
  local gpu="$1"
  if (( NEXT_MODEL_INDEX >= ${#MODEL_SPECS[@]} )); then
    return 0
  fi
  launch_generate "${gpu}" "${MODEL_SPECS[${NEXT_MODEL_INDEX}]}"
  NEXT_MODEL_INDEX=$((NEXT_MODEL_INDEX + 1))
}

maybe_launch_pending_judges() {
  while (( ${#PENDING_JUDGE_TOKENS[@]} > 0 && ${#ACTIVE_JUDGE_PIDS[@]} < JUDGE_PARALLELISM )); do
    local token="${PENDING_JUDGE_TOKENS[0]}"
    PENDING_JUDGE_TOKENS=("${PENDING_JUDGE_TOKENS[@]:1}")
    IFS='|' read -r label alias adapter_path run_name <<< "${token}"
    launch_judge "${label}" "${alias}" "${adapter_path}" "${run_name}"
  done
}

require_env OPENAI_API_KEY
require_env OPENAI_BASE_URL

split_csv "${GPU_LIST}" GPU_IDS

declare -a MODEL_SPECS=(
  "base|qwen3_8b_base|"
  "sft_1k|qwen3_8b_huatuo_1k_lora|${PROJECT_ROOT}/outputs/sft/20260408_222930_qwen3-8b_medical-sft-1k_lora_clean/final_model"
  "sft_5w_ckpt75|qwen3_8b_huatuo_5w_ckpt75|${PROJECT_ROOT}/outputs/sft/20260409_121822_qwen3-8b_huatuo-5w_lora_eval/checkpoints/checkpoint-75"
  "dpo_v2_ckpt30|qwen3_8b_dpo_v2_ckpt30|${PROJECT_ROOT}/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/checkpoints/checkpoint-30"
  "dpo_v2_ckpt330|qwen3_8b_dpo_v2_ckpt330|${PROJECT_ROOT}/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/checkpoints/checkpoint-330"
  "hq50k_best|qwen3_8b_hq50k_best|${PROJECT_ROOT}/outputs/sft/20260410_234458_qwen3-8b_hq-50k_lora_eval/final_model"
)

if [[ -n "${MODEL_FILTER}" ]]; then
  split_csv "${MODEL_FILTER}" MODEL_FILTER_ITEMS
  declare -A MODEL_FILTER_SET=()
  for item in "${MODEL_FILTER_ITEMS[@]}"; do
    MODEL_FILTER_SET["${item}"]=1
  done

  FILTERED_MODEL_SPECS=()
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r label _ _ <<< "${spec}"
    if [[ -n "${MODEL_FILTER_SET[${label}]:-}" ]]; then
      FILTERED_MODEL_SPECS+=("${spec}")
    fi
  done
  MODEL_SPECS=("${FILTERED_MODEL_SPECS[@]}")
fi

declare -a ACTIVE_GEN_PIDS=()
declare -a ACTIVE_JUDGE_PIDS=()
declare -a PENDING_JUDGE_TOKENS=()
declare -A GEN_PID_TO_GPU=()
declare -A GEN_PID_TO_LABEL=()
declare -A GEN_PID_TO_ALIAS=()
declare -A GEN_PID_TO_ADAPTER=()
declare -A GEN_PID_TO_RUN=()
declare -A JUDGE_PID_TO_LABEL=()
declare -A JUDGE_PID_TO_ALIAS=()
declare -A JUDGE_PID_TO_ADAPTER=()
declare -A JUDGE_PID_TO_RUN=()

NEXT_MODEL_INDEX=0

{
  echo -e "label\talias\tadapter_path\trun_name"
} > "${MANIFEST_PATH}"

log "batch_id=${BATCH_ID}"
log "gpu_list=${GPU_LIST}"
log "judge_parallelism=${JUDGE_PARALLELISM}"
log "per_theme_examples=${PER_THEME_EXAMPLES}"
log "max_examples=${MAX_EXAMPLES}"
log "seed=${SEED}"
log "generator_batch_size=${GENERATOR_BATCH_SIZE}"
log "skip_judge=${SKIP_JUDGE}"
log "stage_max_retries=${STAGE_MAX_RETRIES}"
log "retry_sleep_seconds=${RETRY_SLEEP_SECONDS}"
log "model_filter=${MODEL_FILTER:-<all>}"
log "output_root=${OUTPUT_ROOT}"
log "manifest_path=${MANIFEST_PATH}"
log "scheduler_log=${SCHEDULER_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    maybe_launch_next_generate "${gpu}"
  done
  while (( NEXT_MODEL_INDEX < ${#MODEL_SPECS[@]} )); do
    maybe_launch_next_generate "${GPU_IDS[0]}"
  done
  log "dry-run complete"
  exit 0
fi

for gpu in "${GPU_IDS[@]}"; do
  maybe_launch_next_generate "${gpu}"
done

while (( ${#ACTIVE_GEN_PIDS[@]} > 0 || ${#ACTIVE_JUDGE_PIDS[@]} > 0 || ${#PENDING_JUDGE_TOKENS[@]} > 0 || NEXT_MODEL_INDEX < ${#MODEL_SPECS[@]} )); do
  maybe_launch_pending_judges

  if (( ${#ACTIVE_GEN_PIDS[@]} == 0 && ${#ACTIVE_JUDGE_PIDS[@]} == 0 )); then
    break
  fi

  finished_pid=""
  set +e
  wait -n -p finished_pid
  finished_status=$?
  set -e

  if [[ -z "${finished_pid:-}" ]]; then
    break
  fi

  if [[ -n "${GEN_PID_TO_LABEL[${finished_pid}]:-}" ]]; then
    label="${GEN_PID_TO_LABEL[${finished_pid}]}"
    alias="${GEN_PID_TO_ALIAS[${finished_pid}]}"
    adapter_path="${GEN_PID_TO_ADAPTER[${finished_pid}]}"
    run_name="${GEN_PID_TO_RUN[${finished_pid}]}"
    gpu="${GEN_PID_TO_GPU[${finished_pid}]}"

    mapfile -t ACTIVE_GEN_PIDS < <(remove_pid_from_list "${finished_pid}" "${ACTIVE_GEN_PIDS[@]}")

    if [[ "${finished_status}" -eq 0 ]]; then
      log "generate finished on gpu${gpu}: ${label} -> ${run_name}"
      if [[ "${SKIP_JUDGE}" == "1" ]]; then
        log "skip judge for ${label} because SKIP_JUDGE=1"
      else
        PENDING_JUDGE_TOKENS+=("${label}|${alias}|${adapter_path}|${run_name}")
      fi
    else
      log "generate failed on gpu${gpu}: ${label} -> ${run_name} (exit=${finished_status})"
    fi

    unset 'GEN_PID_TO_GPU[$finished_pid]' 'GEN_PID_TO_LABEL[$finished_pid]' 'GEN_PID_TO_ALIAS[$finished_pid]' 'GEN_PID_TO_ADAPTER[$finished_pid]' 'GEN_PID_TO_RUN[$finished_pid]'
    maybe_launch_next_generate "${gpu}"
    continue
  fi

  if [[ -n "${JUDGE_PID_TO_LABEL[${finished_pid}]:-}" ]]; then
    label="${JUDGE_PID_TO_LABEL[${finished_pid}]}"
    run_name="${JUDGE_PID_TO_RUN[${finished_pid}]}"

    mapfile -t ACTIVE_JUDGE_PIDS < <(remove_pid_from_list "${finished_pid}" "${ACTIVE_JUDGE_PIDS[@]}")

    if [[ "${finished_status}" -eq 0 ]]; then
      log "judge finished: ${label} -> ${run_name}"
    else
      log "judge failed: ${label} -> ${run_name} (exit=${finished_status})"
    fi

    unset 'JUDGE_PID_TO_LABEL[$finished_pid]' 'JUDGE_PID_TO_ALIAS[$finished_pid]' 'JUDGE_PID_TO_ADAPTER[$finished_pid]' 'JUDGE_PID_TO_RUN[$finished_pid]'
    continue
  fi

  log "unknown child finished: pid=${finished_pid} exit=${finished_status}"
done

log "all queued evaluations finished"
