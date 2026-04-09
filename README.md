# Star Medical GPT

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
  run_eval_healthbench_qwen3_8b_base.sh
  run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh
script/
  sft_data_prepare.py
  train_sft.py
  run_sft_qwen3_8b_medical_1k.sh
  run_sft_qwen3_8b_huatuo_5w.sh
  export_experiment_records.py
data/
  sft/
    raw/
    processed/
outputs/
  sft/
experiment_records/
```

## Documentation

- [Script Guide](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.md)
  Detailed usage for every core script, including commands, parameters, examples, and output files.
- [Workflow Guide](/home/qjh/llm_learning/my_medical_gpt/docs/WORKFLOW.md)
  Recommended project workflow from raw data to smoke test, formal run, and experiment archival.
- [Evaluation Design](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.md)
  Explains why `HealthBench` is the main open-ended benchmark and how it maps to `SFT`, `DPO`, and `GRPO`.
- [Evaluation README](/home/qjh/llm_learning/my_medical_gpt/evaluation/README.md)
  Covers evaluation modes, output files, recovery behavior, and benchmark-facing code layout.
- [Experiment Records README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/README.md)
  Explains what gets exported into git-tracked experiment snapshots.

## Quick start

Activate the environment:

```bash
conda activate medicalgpt
```

Prepare a dataset:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/raw.jsonl \
  --split train \
  --input-format auto
```

Run the 1k smoke test:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

Run the 5w formal version:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

Export lightweight experiment records:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```

Run a HealthBench base-model smoke evaluation without official judging:

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_eval_healthbench_qwen3_8b_base.sh
```

Run official HealthBench judging after exporting the API key:

```bash
export OPENAI_API_KEY=your_key_here
MODE=full JUDGE_MODE=openai \
bash /home/qjh/llm_learning/my_medical_gpt/script/run_eval_healthbench_qwen3_8b_base.sh
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
