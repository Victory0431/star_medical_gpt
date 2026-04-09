#!/usr/bin/env bash

set -euo pipefail

export TRAIN_DATA="${TRAIN_DATA:-/home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_5w.processed.jsonl}"
export VALID_DATA="${VALID_DATA:-/home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl}"
export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_qwen3-8b_huatuo-5w_lora_eval}"

bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
