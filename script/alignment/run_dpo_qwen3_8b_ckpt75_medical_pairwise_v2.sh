#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/home/qjh/miniconda3/envs/medicalgpt/bin/python"
TORCHRUN_BIN="/home/qjh/miniconda3/envs/medicalgpt/bin/torchrun"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/alignment/processed/dpo_v2/train/medical_pairwise_train_v2.processed.jsonl}"
VALID_DATA="${VALID_DATA:-${PROJECT_ROOT}/data/alignment/processed/dpo_v2/valid/medical_pairwise_valid_v2.processed.jsonl}"
AUX_VALID_DATA="${AUX_VALID_DATA:-${PROJECT_ROOT}/data/sft/processed/valid/valid_zh_500.processed.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dpo}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_qwen3-8b_ckpt75_medical_pairwise_v2_dpo}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CONSOLE_LOG="${RUN_DIR}/logs/console.log"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29541}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
NUM_PROC="${NUM_PROC:-16}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
MAX_AUX_EVAL_SAMPLES="${MAX_AUX_EVAL_SAMPLES:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
AUX_EVAL_BATCH_SIZE="${AUX_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
BETA="${BETA:-0.1}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
LOGGING_INTERVAL="${LOGGING_INTERVAL:-5}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
BEST_MODEL_METRIC="${BEST_MODEL_METRIC:-eval_rewards/accuracies}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"

mkdir -p "${RUN_DIR}/logs"

export WANDB_PROJECT="${WANDB_PROJECT:-my-medical-gpt-dpo}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"

{
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "model_path=${MODEL_PATH}"
  echo "train_data=${TRAIN_DATA}"
  echo "valid_data=${VALID_DATA}"
  echo "aux_valid_data=${AUX_VALID_DATA}"
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
  --model-name-or-path "${MODEL_PATH}"
  --train-data "${TRAIN_DATA}"
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --cache-dir "${PROJECT_ROOT}/cache"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-completion-length "${MAX_COMPLETION_LENGTH}"
  --max-length "${MAX_LENGTH}"
  --model-max-length "${MODEL_MAX_LENGTH}"
  --num-proc "${NUM_PROC}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-eval-samples "${MAX_EVAL_SAMPLES}"
  --max-aux-eval-samples "${MAX_AUX_EVAL_SAMPLES}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --aux-eval-batch-size "${AUX_EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --beta "${BETA}"
  --loss-type "${LOSS_TYPE}"
  --logging-interval "${LOGGING_INTERVAL}"
  --eval-strategy "${EVAL_STRATEGY}"
  --eval-interval "${EVAL_INTERVAL}"
  --save-strategy "${SAVE_STRATEGY}"
  --save-interval "${SAVE_INTERVAL}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --target-modules "${TARGET_MODULES}"
  --metric-for-best-model "${BEST_MODEL_METRIC}"
  --greater-is-better
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

if [[ -n "${AUX_VALID_DATA}" ]]; then
  TRAIN_ARGS+=(--aux-valid-data "${AUX_VALID_DATA}")
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
  "${PROJECT_ROOT}/script/alignment/train_dpo.py" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | while IFS= read -r line; do
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
  done | tee -a "${CONSOLE_LOG}"
