# Star Medical GPT

Chinese-first documentation is now available at [README.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/README.zh-CN.md).
For Chinese usage guides, prefer the `*.zh-CN.md` files across this repository.

A resume-oriented medical LLM fine-tuning project built around `Qwen3-8B`, Chinese medical SFT data, LoRA training, evaluation during training, W&B experiment tracking, and lightweight reproducibility records.

## Project focus

- Base model: `Qwen3-8B`
- Fine-tuning method: `LoRA`
- Loss design: standard SFT with assistant-only loss by default
- Runtime: single-node 2-GPU `torchrun`
- Tracking: Weights & Biases
- Evaluation: periodic `eval_loss` during training plus final evaluation
- Experiment traceability: lightweight exported records under `experiment_records/`

## Repo layout

```text
evaluation/
  benchmarks/
  generators/
  judges/
  configs/
script/
  sft/
    sft_data_prepare.py
    train_sft.py
    run_sft_qwen3_8b_medical_1k.sh
    run_sft_qwen3_8b_huatuo_5w.sh
  alignment/
    merge_lora.py
    dpo_data_prepare.py
    train_dpo.py
    run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh
  eval/
    run_eval_healthbench_qwen3_8b_base.sh
    run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh
  ops/
    export_experiment_records.py
  grpo/
data/
  sft/
    raw/
    processed/
outputs/
  sft/
experiment_records/
```

## Documentation

- [中文总览](/home/qjh/llm_learning/my_medical_gpt/README.zh-CN.md)
  Chinese-first entry for the repository.
- [脚本使用指南（中文）](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.zh-CN.md)
  Chinese guide for core scripts, commands, parameters, and outputs.
- [脚本目录说明（中文）](/home/qjh/llm_learning/my_medical_gpt/script/README.zh-CN.md)
  Chinese overview of how `script/sft`, `script/alignment`, `script/eval`, `script/ops`, and `script/grpo` are organized.
- [项目工作流（中文）](/home/qjh/llm_learning/my_medical_gpt/docs/WORKFLOW.zh-CN.md)
  Chinese workflow from data prep to training, evaluation, and archiving.
- [评测设计（中文）](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.zh-CN.md)
  Chinese explanation of the benchmark design and HealthBench positioning.
- [评测结果解读（中文）](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.zh-CN.md)
  Chinese interpretation of the current base vs `huatuo_1k` smoke comparison.
- [评测接入指南（中文）](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_INTEGRATION.zh-CN.md)
  Chinese instructions for plugging future checkpoints into the same eval harness.
- [评测模块说明（中文）](/home/qjh/llm_learning/my_medical_gpt/evaluation/README.zh-CN.md)
  Chinese overview of eval modes, outputs, and recovery behavior.
- [实验记录说明（中文）](/home/qjh/llm_learning/my_medical_gpt/experiment_records/README.zh-CN.md)
  Chinese explanation of exported experiment snapshots.

- [Script Guide](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.md)
  Detailed usage for every core script, including commands, parameters, examples, and output files.
- [Workflow Guide](/home/qjh/llm_learning/my_medical_gpt/docs/WORKFLOW.md)
  Recommended project workflow from raw data to smoke test, formal run, and experiment archival.
- [Evaluation Design](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.md)
  Explains why `HealthBench` is the main open-ended benchmark and how it maps to `SFT`, `DPO`, and `GRPO`.
- [Evaluation Results](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.md)
  Summarizes the current base vs `huatuo_1k` smoke comparison and how to read the result.
- [Evaluation Integration](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_INTEGRATION.md)
  Shows how to plug future `SFT`, `DPO`, `GRPO`, merged models, or LoRA adapters into the same evaluation harness.
- [Evaluation README](/home/qjh/llm_learning/my_medical_gpt/evaluation/README.md)
  Covers evaluation modes, output files, recovery behavior, and benchmark-facing code layout.
- [DPO Workflow](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.md)
  Explains how the merged `SFT checkpoint` feeds into `DPO`, and why pairwise metrics and heterogeneous `valid_zh` monitoring are tracked together.
- [Experiment Records README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/README.md)
  Explains what gets exported into git-tracked experiment snapshots.

## Quick start

Activate the environment:

```bash
conda activate medicalgpt
```

Prepare a dataset:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft/sft_data_prepare.py \
  --input-files /path/to/raw.jsonl \
  --split train \
  --input-format auto
```

Run the 1k smoke test:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_medical_1k.sh
```

Run the 5w formal version:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_huatuo_5w.sh
```

Merge the best `SFT checkpoint` into a stable alignment starting point:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/merge_lora.py \
  --base-model-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter-path /home/qjh/llm_learning/my_medical_gpt/outputs/sft/20260409_121822_qwen3-8b_huatuo-5w_lora_eval/checkpoints/checkpoint-75 \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/merged_models/sft \
  --run-name 20260410_qwen3-8b_huatuo-5w_ckpt75_merged \
  --log-root /home/qjh/llm_learning/my_medical_gpt/outputs/logs/merge \
  --device cuda \
  --dtype bfloat16
```

Prepare pairwise `DPO` data:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py \
  --input-files /home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_train.jsonl \
  --split train \
  --output-name medical_pairwise_train
```

Launch `DPO`:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh
```

Export lightweight experiment records:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/ops/export_experiment_records.py --all --force
```

Run a HealthBench base-model smoke evaluation without official judging:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_base.sh
```

Run official HealthBench judging after exporting the API key:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
MODE=full JUDGE_MODE=openai \
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_base.sh
```

## Data policy

Large raw and processed datasets are intentionally **not committed** to this repository.
Large checkpoints, LoRA weights, optimizer states, tokenizer dumps, and local W&B artifacts are also intentionally **not committed**.

Only lightweight reproducibility artifacts such as:

- run arguments
- dataset statistics
- metrics history
- final train/eval summaries
- sanitized logs

are exported into `experiment_records/` and suitable for GitHub.

## Notes

- The current environment falls back to standard attention when `flash_attn` is not installed.
- Validation can come from an external validation set or from a train split holdout, depending on launcher arguments.
- `export_experiment_records.py --all` skips `dryrun` directories and obvious failed runs by default.
- This repo is organized to support later extension toward larger medical SFT experiments, evaluation benchmarks, and alignment stages.
- The evaluation judge supports OpenAI-compatible APIs through `OPENAI_API_KEY` and `OPENAI_BASE_URL`, and defaults to `gpt-5.2` for judge runs.
- `DPO` training now tracks both pairwise preference metrics for checkpoint selection and auxiliary heterogeneous `valid_zh` LM loss for stability monitoring.
