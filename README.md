# Star Medical GPT

A resume-oriented medical LLM fine-tuning project built around `Qwen3-8B`, Chinese medical SFT data, LoRA training, and W&B-based experiment tracking.

## What is in this repo

- `script/sft_data_prepare.py`
  Preprocesses raw SFT datasets into unified `conversations` format.
- `script/train_sft.py`
  Runs standard SFT with assistant-only loss, LoRA, periodic evaluation, checkpointing, and W&B logging.
- `script/run_sft_qwen3_8b_medical_1k.sh`
  Smoke-test launcher for `huatuo_1k + valid_zh_500`.
- `script/run_sft_qwen3_8b_huatuo_5w.sh`
  Formal small-version launcher for `huatuo_5w + valid_zh_500`.
- `script/export_experiment_records.py`
  Exports lightweight, git-friendly experiment records from `outputs/` into `experiment_records/`.

## Current training design

- Base model: `Qwen3-8B`
- Fine-tuning method: `LoRA`
- Loss design: only assistant response tokens contribute to loss
- Runtime: single-node 2-GPU `torchrun`
- Tracking: Weights & Biases
- Evaluation: periodic `eval_loss` during training, plus final evaluation

## Data policy

Large raw and processed datasets are intentionally **not committed** to this repository.
Large checkpoints, LoRA weights, optimizer states, tokenizer dumps, and local W&B artifacts are also intentionally **not committed**.

Typical local layout:

```text
data/
  sft/
    raw/
    processed/
```

You can regenerate processed datasets locally with:

```bash
python /path/to/script/sft_data_prepare.py ...
```

## Example runs

Smoke test:

```bash
conda activate medicalgpt
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

Formal small version:

```bash
conda activate medicalgpt
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

Export lightweight experiment records:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py \
  --run-name 20260409_1204_qwen3-8b_huatuo-1k_eval_smoke \
  --force
```

Or export every discovered run under `outputs/sft`:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```

By default, `--all` skips `dryrun` directories and obvious failed runs. If you really want everything, add `--include-dryrun --include-failed`.

## Notes

- The current environment falls back to standard attention when `flash_attn` is not installed.
- Validation can come from an external validation set or from a train split holdout, depending on launcher arguments.
- This repo is intentionally organized to support later extension toward larger medical SFT experiments, evaluation benchmarks, and alignment stages.
