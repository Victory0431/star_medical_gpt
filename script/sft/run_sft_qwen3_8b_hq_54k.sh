#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/sft/curation/subsets/hq_54k_high_bucket_all.jsonl}"
export VALID_DATA="${VALID_DATA:-${PROJECT_ROOT}/data/sft/processed/valid/valid_zh_500.processed.jsonl}"
export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_qwen3-8b_hq-54k_lora_eval}"

bash "${SCRIPT_DIR}/run_sft_qwen3_8b_medical_1k.sh"
