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
SEED="${SEED:-42}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)_healthbench_consensus_from_full_two_models}"

LOG_DIR="${PROJECT_ROOT}/evaluation/logs"
WRAPPER_LOG="${LOG_DIR}/${BATCH_ID}.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

SFT_FULL_RUN="20260413_healthbench_full_sft5w_dpo330_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_full_all_seed42"
DPO_FULL_RUN="20260413_healthbench_full_sft5w_dpo330_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_full_all_seed42"

SFT_CONS_RUN="${BATCH_ID}_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_consensus_all_seed${SEED}"
DPO_CONS_RUN="${BATCH_ID}_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_consensus_all_seed${SEED}"

SFT_ADAPTER="${PROJECT_ROOT}/outputs/sft/20260409_121822_qwen3-8b_huatuo-5w_lora_eval/checkpoints/checkpoint-75"
DPO_ADAPTER="${PROJECT_ROOT}/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/checkpoints/checkpoint-330"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${WRAPPER_LOG}"
}

ensure_openai_env() {
  if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
    log "missing OPENAI_API_KEY or OPENAI_BASE_URL"
    exit 1
  fi
}

ensure_consensus_dataset() {
  log "ensuring HealthBench consensus dataset cache exists"
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" CACHE_DIR_ENV="${CACHE_DIR}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
from evaluation.benchmarks.healthbench import load_healthbench_examples

examples, cache_path = load_healthbench_examples(
    subset_name="consensus",
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

filter_responses() {
  local source_run_name="$1"
  local target_run_name="$2"

  local source_responses="${OUTPUT_ROOT}/${source_run_name}/responses.jsonl"
  local target_dir="${OUTPUT_ROOT}/${target_run_name}"
  local target_responses="${target_dir}/responses.jsonl"

  mkdir -p "${target_dir}/artifacts" "${target_dir}/logs"

  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  SOURCE_RESPONSES="${source_responses}" \
  TARGET_RESPONSES="${target_responses}" \
  CACHE_DIR_ENV="${CACHE_DIR}" \
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os
from evaluation.benchmarks.healthbench import load_healthbench_examples

source_responses = Path(os.environ["SOURCE_RESPONSES"])
target_responses = Path(os.environ["TARGET_RESPONSES"])
cache_root = Path(os.environ["CACHE_DIR_ENV"])

examples, cache_path = load_healthbench_examples(
    subset_name="consensus",
    cache_root=cache_root,
    max_examples=-1,
    sampling_mode="sequential",
    seed=42,
    shuffle=False,
    per_theme_examples=-1,
)
consensus_prompt_ids = [example.prompt_id for example in examples]
consensus_prompt_id_set = set(consensus_prompt_ids)

rows_by_prompt_id = {}
with source_responses.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        rows_by_prompt_id[str(row["prompt_id"])] = row

missing = [prompt_id for prompt_id in consensus_prompt_ids if prompt_id not in rows_by_prompt_id]
if missing:
    raise SystemExit(f"missing {len(missing)} consensus prompt_ids in {source_responses}: {missing[:10]}")

target_responses.parent.mkdir(parents=True, exist_ok=True)
with target_responses.open("w", encoding="utf-8") as f:
    for prompt_id in consensus_prompt_ids:
        f.write(json.dumps(rows_by_prompt_id[prompt_id], ensure_ascii=False) + "\n")

manifest = {
    "source_responses": str(source_responses),
    "target_responses": str(target_responses),
    "subset_name": "consensus",
    "dataset_path": str(cache_path),
    "num_examples": len(consensus_prompt_ids),
    "copied_examples": len(consensus_prompt_ids),
    "source_total_examples": len(rows_by_prompt_id),
}
manifest_path = target_responses.parent / "artifacts" / "copied_from_full_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PY
}

run_judge() {
  local model_alias="$1"
  local adapter_path="$2"
  local run_name="$3"
  local responses_path="${OUTPUT_ROOT}/${run_name}/responses.jsonl"

  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/evaluation/run_eval.py" \
    --benchmark healthbench \
    --subset-name consensus \
    --mode judge_only \
    --judge-mode openai \
    --judge-model "${JUDGE_MODEL}" \
    --model-name-or-path "${MODEL_PATH}" \
    --model-alias "${model_alias}" \
    --adapter-path "${adapter_path}" \
    --cache-dir "${CACHE_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${run_name}" \
    --responses-path "${responses_path}" \
    --max-examples -1 \
    --sampling-mode sequential \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}"
}

main() {
  log "wrapper_log=${WRAPPER_LOG}"
  log "batch_id=${BATCH_ID}"
  log "sft_full_run=${SFT_FULL_RUN}"
  log "dpo_full_run=${DPO_FULL_RUN}"
  log "sft_consensus_run=${SFT_CONS_RUN}"
  log "dpo_consensus_run=${DPO_CONS_RUN}"

  ensure_openai_env
  ensure_consensus_dataset >> "${WRAPPER_LOG}" 2>&1

  log "filtering full responses -> consensus responses for sft"
  filter_responses "${SFT_FULL_RUN}" "${SFT_CONS_RUN}" >> "${WRAPPER_LOG}" 2>&1

  log "filtering full responses -> consensus responses for dpo"
  filter_responses "${DPO_FULL_RUN}" "${DPO_CONS_RUN}" >> "${WRAPPER_LOG}" 2>&1

  log "starting consensus judges in parallel for sft and dpo"
  (
    run_judge qwen3_8b_huatuo_5w_ckpt75 "${SFT_ADAPTER}" "${SFT_CONS_RUN}"
  ) >> "${WRAPPER_LOG}" 2>&1 &
  local sft_judge_pid=$!

  (
    run_judge qwen3_8b_dpo_v2_ckpt330 "${DPO_ADAPTER}" "${DPO_CONS_RUN}"
  ) >> "${WRAPPER_LOG}" 2>&1 &
  local dpo_judge_pid=$!

  wait "${sft_judge_pid}"
  log "sft consensus judge finished"

  wait "${dpo_judge_pid}"
  log "dpo consensus judge finished"

  log "all consensus-from-full judge stages finished"
}

main "$@"
