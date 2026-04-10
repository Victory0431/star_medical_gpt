# Evaluation Integration

This document explains how to plug new checkpoints into the existing evaluation framework.

## Supported model connection patterns

The current evaluation code supports three common patterns.

### 1. Base model only

Use when you want to evaluate a raw pretrained model or a fully merged model directory.

```bash
conda activate medicalgpt
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

MODEL_PATH=/path/to/base_or_merged_model \
MODE=full JUDGE_MODE=openai JUDGE_MODEL=gpt-5.2 \
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_base.sh
```

### 2. Base model + LoRA adapter

Use when you want to evaluate `SFT`, `DPO`, or `GRPO` adapters without merging weights.

```bash
conda activate medicalgpt
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

MODEL_PATH=/path/to/base_model \
ADAPTER_PATH=/path/to/adapter_or_final_model \
MODE=full JUDGE_MODE=openai JUDGE_MODEL=gpt-5.2 \
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh
```

### 3. Reuse existing generations and only rerun judge

Use when:

- the local generation is expensive
- the judge provider changed
- you want to compare judge settings without regenerating outputs

```bash
conda activate medicalgpt
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json \
  --mode judge_only \
  --judge-mode openai \
  --judge-model gpt-5.2 \
  --responses-path /path/to/responses.jsonl \
  --run-name my_judge_only_run
```

## How this maps to later training stages

Use the same harness for:

- `SFT`: set `ADAPTER_PATH` to the SFT LoRA output
- `DPO`: set `ADAPTER_PATH` to the DPO LoRA output
- `GRPO`: set `ADAPTER_PATH` to the GRPO LoRA output

This means the benchmark layer stays fixed while the model checkpoint changes.
That is exactly what you want for clean comparison.

## Recommended naming convention

Use run names that expose:

- date
- benchmark
- model stage
- judge model
- sample count

Example:

- `20260409_healthbench_huatuo5w_gpt52_full_50`
- `20260409_healthbench_dpo_v1_gpt52_full_50`
- `20260409_healthbench_grpo_v1_gpt52_full_50`

## Recommended workflow

For formal evaluation, use this order:

1. `generate_only`
2. inspect a few outputs manually
3. `judge_only`
4. compare `summary.json` files across stages

Reason:

- local generation and external judging have different failure modes
- separating them makes reruns much cheaper

## Judge settings

Current recommendation for judge runs:

- model: `gpt-5.2`
- `temperature=0`
- strict JSON output
- fixed benchmark subset for fair comparison

Only change this if you intentionally want a judge-ablation experiment.
