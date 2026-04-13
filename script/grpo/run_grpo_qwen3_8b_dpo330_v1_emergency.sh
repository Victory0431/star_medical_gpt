#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/home/qjh/miniconda3/envs/medicalgpt/bin/python"
TORCHRUN_BIN="/home/qjh/miniconda3/envs/medicalgpt/bin/torchrun"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PROJECT_ROOT}/outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model}"
INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH:-${PROJECT_ROOT}/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/checkpoints/checkpoint-330}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/alignment/grpo/v1_emergency_context/train/medical_grpo_prompt_v1_emergency_context.train.jsonl}"
VALID_DATA="${VALID_DATA:-${PROJECT_ROOT}/data/alignment/grpo/v1_emergency_context/valid/medical_grpo_prompt_v1_emergency_context.valid.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/grpo}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_qwen3-8b_dpo330_grpo_v1_emergency}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CONSOLE_LOG="${RUN_DIR}/logs/console.log"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29581}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2560}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-120}"
NUM_PROC="${NUM_PROC:-16}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-120}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
BETA="${BETA:-0.02}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
NUM_GENERATIONS_EVAL="${NUM_GENERATIONS_EVAL:-4}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.03}"
LOGGING_INTERVAL="${LOGGING_INTERVAL:-2}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_reward}"
LOSS_TYPE="${LOSS_TYPE:-dapo}"
MULTI_OBJECTIVE_AGGREGATION="${MULTI_OBJECTIVE_AGGREGATION:-sum_then_normalize}"
SCALE_REWARDS="${SCALE_REWARDS:-group}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"

mkdir -p "${RUN_DIR}/logs"

export WANDB_PROJECT="${WANDB_PROJECT:-my-medical-gpt-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

{
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "base_model=${BASE_MODEL_PATH}"
  echo "init_adapter=${INIT_ADAPTER_PATH}"
  echo "train_data=${TRAIN_DATA}"
  echo "valid_data=${VALID_DATA}"
  echo "console_log=${CONSOLE_LOG}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "master_port=${MASTER_PORT}"
  echo "wandb_project=${WANDB_PROJECT}"
  echo "wandb_mode=${WANDB_MODE}"
} | while IFS= read -r line; do
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
done | tee -a "${CONSOLE_LOG}"

TRAIN_ARGS=(
  --base-model-name-or-path "${BASE_MODEL_PATH}"
  --init-adapter-path "${INIT_ADAPTER_PATH}"
  --train-data "${TRAIN_DATA}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --cache-dir "${PROJECT_ROOT}/cache"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --model-max-length "${MODEL_MAX_LENGTH}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-completion-length "${MAX_COMPLETION_LENGTH}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --num-proc "${NUM_PROC}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --beta "${BETA}"
  --num-generations "${NUM_GENERATIONS}"
  --num-generations-eval "${NUM_GENERATIONS_EVAL}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --repetition-penalty "${REPETITION_PENALTY}"
  --logging-interval "${LOGGING_INTERVAL}"
  --eval-strategy "${EVAL_STRATEGY}"
  --eval-interval "${EVAL_INTERVAL}"
  --save-strategy "${SAVE_STRATEGY}"
  --save-interval "${SAVE_INTERVAL}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --metric-for-best-model "${METRIC_FOR_BEST_MODEL}"
  --loss-type "${LOSS_TYPE}"
  --multi-objective-aggregation "${MULTI_OBJECTIVE_AGGREGATION}"
  --scale-rewards "${SCALE_REWARDS}"
  --mask-truncated-completions
  --log-completions
  --num-completions-to-print 2
  --bf16
  --gradient-checkpointing
)

if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
  TRAIN_ARGS+=(--flash-attn)
else
  TRAIN_ARGS+=(--no-flash-attn)
fi

if [[ -n "${VALID_DATA}" ]]; then
  TRAIN_ARGS+=(--valid-data "${VALID_DATA}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  TRAIN_ARGS+=(--dry-run)
fi

"${TORCHRUN_BIN}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${PROJECT_ROOT}/script/grpo/train_grpo.py" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | while IFS= read -r line; do
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
  done | tee -a "${CONSOLE_LOG}"
