# Script Guide

This document explains the purpose, commands, parameters, and outputs of the core scripts in this repository.

Evaluation scripts are included at the end so training and benchmark usage live in one place.

## 1. `sft_data_prepare.py`

Path:

- [`script/sft_data_prepare.py`](/home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py)

### What it does

- Reads raw SFT datasets in `sharegpt` or `instruction/input/output` format
- Normalizes them into unified `{"conversations": [...]}` JSONL
- Optionally deduplicates exact duplicate samples
- Writes preprocessing reports for later audit

### Supported input formats

`sharegpt`

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

`instruction`

```json
{"instruction":"...","input":"...","output":"..."}
```

### Default output layout

```text
data/sft/processed/
  train/
  valid/
  test/
  reports/
```

### Basic command

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/raw.jsonl \
  --split train \
  --input-format auto
```

### Common examples

Prepare the 1k smoke-test dataset:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /home/qjh/llm_learning/medicalgpt/MedicalGPT-main/data/finetune/finetune/medical_sft_1K_format.jsonl \
  --split train \
  --input-format sharegpt \
  --output-name huatuo_1k
```

Prepare a 5w ShareGPT dataset:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /home/qjh/llm_learning/medicalgpt/MedicalGPT-main/data/finetune/finetune/HuatuoGPT2_sft_instruct_GPT4_sharegpt.jsonl \
  --split train \
  --input-format sharegpt \
  --workers 32 \
  --output-name huatuo_5w
```

Convert instruction format into conversations:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/train_zh_0.json \
  --split train \
  --input-format instruction \
  --workers 32 \
  --output-name train_zh_195w
```

Prepare validation data:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/valid.json \
  --split valid \
  --input-format instruction \
  --output-name valid_zh_500
```

### Parameters

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--input-files` | One or more raw `.json` or `.jsonl` files | Required |
| `--split` | Output split directory | `train`, `valid`, `test` |
| `--input-format` | Parsing mode | `auto`, `sharegpt`, `instruction` |
| `--output-root` | Root for processed data | `data/sft/processed` |
| `--workers` | Thread count for preprocessing | `16`, `32` |
| `--no-deduplicate` | Disable exact-record deduplication | Optional |
| `--output-name` | Custom output filename stem for one input file | `huatuo_5w` |

### Output files

For one processed dataset:

- `data/sft/processed/train/xxx.processed.jsonl`
- `data/sft/processed/reports/xxx.processed.report.json`
- `data/sft/processed/reports/summary.json`

### Notes

- `--input-format auto` is recommended when the raw source may mix file suffixes such as `.json` and `.jsonl`.
- The output is already in the format expected by `train_sft.py`.
- The script streams JSONL-like files line by line and is suitable for larger datasets.

## 2. `train_sft.py`

Path:

- [`script/train_sft.py`](/home/qjh/llm_learning/my_medical_gpt/script/train_sft.py)

### What it does

- Loads processed `conversations` datasets
- Applies the Qwen chat template
- Computes standard SFT loss on assistant tokens only by default
- Runs LoRA fine-tuning
- Performs periodic evaluation and checkpoint saving
- Logs to W&B and local JSONL/text logs

### Standard command

Single-process direct launch:

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/train_sft.py \
  --model-name-or-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_1k.processed.jsonl \
  --valid-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/sft \
  --run-name 20260409_demo_run \
  --wandb-project my-medical-gpt-sft \
  --wandb-mode online \
  --model-max-length 2048 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --eval-strategy steps \
  --eval-steps 25 \
  --save-strategy steps \
  --save-steps 25 \
  --bf16 \
  --gradient-checkpointing \
  --flash-attn
```

Formal 2-GPU launch with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/qjh/miniconda3/envs/medicalgpt/bin/torchrun \
  --nproc_per_node 2 \
  --master_port 29521 \
  /home/qjh/llm_learning/my_medical_gpt/script/train_sft.py \
  --model-name-or-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_5w.processed.jsonl \
  --valid-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/sft \
  --run-name qwen3_8b_huatuo_5w_manual \
  --cache-dir /home/qjh/llm_learning/my_medical_gpt/cache \
  --wandb-project my-medical-gpt-sft \
  --wandb-mode online \
  --model-max-length 2048 \
  --num-proc 16 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --logging-steps 5 \
  --eval-strategy steps \
  --eval-steps 25 \
  --save-strategy steps \
  --save-steps 25 \
  --save-total-limit 3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules all-linear \
  --bf16 \
  --gradient-checkpointing \
  --flash-attn
```

### Recommended use

Use launcher scripts for daily training, and call `train_sft.py` directly when:

- you need a custom run name
- you want to swap datasets quickly
- you want to change batch size, evaluation cadence, or precision settings
- you want a dry run to validate tokenization and dataset loading
- you want to debug a run before freezing it into a launcher

### Parameters

#### Data and path parameters

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--model-name-or-path` | Base model directory | `/home/qjh/llm_learning/base_model/qwen3_8B` |
| `--train-data` | One or more processed train files or directories | `.../huatuo_5w.processed.jsonl` |
| `--valid-data` | Optional processed valid files or directories | `.../valid_zh_500.processed.jsonl` |
| `--validation-split-ratio` | Auto-holdout ratio when no valid set is given | `0.05` |
| `--output-root` | Training output root | `outputs/sft` |
| `--run-name` | Current experiment name | timestamped name |
| `--cache-dir` | Hugging Face and dataset cache directory | `cache` |

#### Tracking and reproducibility

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--wandb-project` | W&B project name | `my-medical-gpt-sft` |
| `--wandb-mode` | W&B mode | `online`, `offline`, `disabled` |
| `--seed` | Random seed | `42` |

#### Training schedule

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--model-max-length` | Sequence length limit | `2048` |
| `--max-train-samples` | Cap train sample count for quick runs | `-1` |
| `--max-eval-samples` | Cap eval sample count | `-1` |
| `--max-steps` | Max optimizer steps, `-1` means epoch-based | `10`, `-1` |
| `--num-proc` | Tokenization worker processes | `16` |
| `--per-device-train-batch-size` | Micro-batch size per GPU | `4` |
| `--per-device-eval-batch-size` | Eval batch size per GPU | `4` |
| `--gradient-accumulation-steps` | Accumulation steps | `8` |
| `--num-train-epochs` | Number of epochs | `1`, `2` |
| `--learning-rate` | Learning rate | `2e-5` |
| `--weight-decay` | Weight decay | `0.01` |
| `--warmup-ratio` | Warmup ratio used to derive warmup steps | `0.03` |

#### Logging, evaluation, and checkpoints

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--logging-steps` | Local/W&B logging frequency | `5`, `10` |
| `--eval-strategy` | Evaluation cadence | `steps`, `epoch`, `no` |
| `--eval-steps` | Eval interval when `steps` is used | `25`, `50` |
| `--save-strategy` | Checkpoint saving cadence | `steps`, `epoch`, `no` |
| `--save-steps` | Save interval when `steps` is used | `25`, `50` |
| `--save-total-limit` | Number of checkpoints to keep | `3` |

#### LoRA and memory

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--lora-r` | LoRA rank | `16` |
| `--lora-alpha` | LoRA alpha | `32` |
| `--lora-dropout` | LoRA dropout | `0.05` |
| `--target-modules` | Target module selection | `all-linear` |
| `--gradient-checkpointing` | Enable activation checkpointing | Enabled by default |
| `--bf16` | Use bfloat16 | Enabled by default |
| `--fp16` | Use fp16 | Optional |
| `--use-cpu` | Force CPU mode | Debug only |
| `--load-in-4bit` | QLoRA-style 4-bit loading | Optional |
| `--flash-attn` | Prefer FlashAttention2 when installed | Enabled by default |

#### Behavior switches

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--train-on-inputs` | Compute loss on full sequence instead of assistant-only outputs | Usually leave off |
| `--dry-run` | Stop after tokenization/statistics, without training | Useful for smoke checks |

### Important behavior

- By default, the script computes loss on assistant response tokens only.
- If `--valid-data` is omitted, the script can auto-split validation data from train via `--validation-split-ratio`.
- If `flash_attn` is unavailable, the script falls back to standard attention and keeps running.
- `--disable-assistant-only-loss` currently exists as a reserved parameter and is not part of the recommended workflow.

### Output layout

Each run creates a directory under `outputs/sft/<run_name>/`:

```text
artifacts/
  run_args.json
  dataset_stats.json
  training_args.json
logs/
  train.log
  console.log
  metrics.jsonl
checkpoints/
final_model/
wandb/
```

## 3. `evaluation/run_eval.py`

Path:

- [`evaluation/run_eval.py`](/home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py)

### What it does

- Loads `HealthBench`
- Runs local generation with the base model or base-plus-LoRA
- Optionally calls the official OpenAI rubric judge
- Saves reusable responses, judgments, summary JSON, summary Markdown, and logs

### Standard command

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --benchmark healthbench \
  --subset-name consensus \
  --mode full \
  --judge-mode openai \
  --judge-model gpt-5.2 \
  --model-name-or-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter-path /home/qjh/llm_learning/my_medical_gpt/outputs/sft/20260408_222930_qwen3-8b_medical-sft-1k_lora_clean/final_model \
  --run-name 20260409_healthbench_huatuo1k \
  --max-examples 10 \
  --generator-device cuda:0
```

### Common examples

Base model generate-only smoke test:

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json \
  --mode generate_only \
  --judge-mode none \
  --run-name 20260409_healthbench_base_generate
```

LoRA model full evaluation:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_huatuo_1k_lora.json \
  --run-name 20260409_healthbench_huatuo1k_full
```

Judge-only retry on existing responses:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json \
  --mode judge_only \
  --responses-path /home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260409_healthbench_base_generate/responses.jsonl \
  --run-name 20260409_healthbench_base_judge
```

### Parameters

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--config` | Optional JSON config | `evaluation/configs/*.json` |
| `--subset-name` | HealthBench subset | `consensus`, `hard`, `full` |
| `--mode` | Run mode | `full`, `generate_only`, `judge_only` |
| `--judge-mode` | Judge backend | `openai`, `none` |
| `--judge-model` | Judge model name | `gpt-5.2` |
| `--model-name-or-path` | Base model path | `/home/qjh/llm_learning/base_model/qwen3_8B` |
| `--adapter-path` | Optional LoRA path | `.../final_model` |
| `--max-examples` | Sample cap for smoke tests | `1`, `10`, `50` |
| `--generator-device` | Generation device | `cuda:0` |
| `--enable-thinking` | Enable Qwen3 thinking mode | usually keep off |
| `--responses-path` | Reuse an existing response file | optional |
| `--judgments-path` | Reuse an existing judgment file | optional |
| `--overwrite-responses` | Force regenerate responses | optional |
| `--overwrite-judgments` | Force rejudge outputs | optional |

### Output files

- `outputs/eval/<run_name>/artifacts/run_args.json`
- `outputs/eval/<run_name>/artifacts/dataset_manifest.json`
- `outputs/eval/<run_name>/logs/eval.log`
- `outputs/eval/<run_name>/responses.jsonl`
- `outputs/eval/<run_name>/judgments.jsonl`
- `outputs/eval/<run_name>/summary.json`
- `outputs/eval/<run_name>/summary.md`

### Launcher scripts

- [`script/run_eval_healthbench_qwen3_8b_base.sh`](/home/qjh/llm_learning/my_medical_gpt/script/run_eval_healthbench_qwen3_8b_base.sh)
- [`script/run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh`](/home/qjh/llm_learning/my_medical_gpt/script/run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh)

Launcher default behavior:

- default mode is `generate_only`
- default judge mode is `none`
- to run official scoring, export `OPENAI_API_KEY`, optionally export `OPENAI_BASE_URL`, and override `MODE=full JUDGE_MODE=openai`

## 4. `run_sft_qwen3_8b_medical_1k.sh`

Path:

- [`script/run_sft_qwen3_8b_medical_1k.sh`](/home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh)

### What it does

- Wraps `torchrun` for 2-GPU training
- Sets default paths for model, train data, valid data, output root, and W&B
- Adds timestamped console logging
- Suitable for the first end-to-end smoke test

### Default training target

- Train set: `huatuo_1k.processed.jsonl`
- Validation set: `valid_zh_500.processed.jsonl`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- Processes: `NPROC_PER_NODE=2`

### Standard command

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

### `nohup` command

```bash
nohup bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh \
  > /home/qjh/llm_learning/my_medical_gpt/outputs/nohup_1k.out 2>&1 &
```

### Common environment overrides

```bash
RUN_NAME=demo_1k_lr1e5 \
LEARNING_RATE=1e-5 \
NUM_TRAIN_EPOCHS=1 \
CUDA_VISIBLE_DEVICES=0,1 \
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

### Common launcher variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `MODEL_PATH` | Base model path | `.../base_model/qwen3_8B` |
| `TRAIN_DATA` | Processed train file | `huatuo_1k.processed.jsonl` |
| `VALID_DATA` | Processed valid file | `valid_zh_500.processed.jsonl` |
| `RUN_NAME` | Experiment name | timestamped |
| `OUTPUT_ROOT` | Output root | `outputs/sft` |
| `CUDA_VISIBLE_DEVICES` | Selected GPUs | `0,1` |
| `NPROC_PER_NODE` | `torchrun` process count | `2` |
| `MASTER_PORT` | Distributed port | `29521` |
| `NUM_TRAIN_EPOCHS` | Epoch count | `2` |
| `MAX_STEPS` | Step cap | `-1` |
| `EVAL_INTERVAL` | Eval interval | `25` |
| `SAVE_INTERVAL` | Save interval | `25` |
| `WANDB_PROJECT` | W&B project | `my-medical-gpt-sft` |
| `WANDB_MODE` | W&B mode | `online` |

## 5. `run_sft_qwen3_8b_huatuo_5w.sh`

Path:

- [`script/run_sft_qwen3_8b_huatuo_5w.sh`](/home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh)

### What it does

- Reuses the 1k launcher
- Overrides `TRAIN_DATA`, `VALID_DATA`, and `RUN_NAME`
- Acts as the first formal small-version training entrypoint

### Standard command

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

### `nohup` command

```bash
nohup bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh \
  > /home/qjh/llm_learning/my_medical_gpt/outputs/nohup_5w.out 2>&1 &
```

### Example override

```bash
RUN_NAME=huatuo_5w_epoch1_eval20 \
NUM_TRAIN_EPOCHS=1 \
EVAL_INTERVAL=20 \
SAVE_INTERVAL=20 \
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

## 6. `export_experiment_records.py`

Path:

- [`script/export_experiment_records.py`](/home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py)

### What it does

- Copies small reproducibility artifacts from `outputs/sft/` into git-tracked `experiment_records/sft/`
- Keeps your repo lightweight
- Makes experiment history reviewable on GitHub
- Skips `dryrun` and obvious failed runs by default when `--all` is used

### Standard commands

Export one run:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py \
  --run-name 20260409_1204_qwen3-8b_huatuo-1k_eval_smoke \
  --force
```

Export all eligible runs:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```

Export absolutely everything:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py \
  --all \
  --include-dryrun \
  --include-failed \
  --force
```

### Parameters

| Parameter | Meaning | Typical value |
| --- | --- | --- |
| `--outputs-root` | Source run directory root | `outputs/sft` |
| `--records-root` | Export destination root | `experiment_records/sft` |
| `--run-name` | Specific run to export. Repeatable | run directory name |
| `--all` | Export all eligible runs under `outputs-root` | Common |
| `--include-dryrun` | Also export `dryrun` runs | Rare |
| `--include-failed` | Also export obvious failed runs | Rare |
| `--force` | Overwrite existing exported records | Common |
| `--max-log-mb` | Max full log size before truncation | `2.0` |

### Exported content

- `artifacts/run_args.json`
- `artifacts/training_args.json`
- `artifacts/dataset_stats.json`
- `logs/metrics.jsonl`
- `logs/train.log`
- `logs/console.log`
- `checkpoints/train_results.json`
- `checkpoints/eval_results.json`
- `checkpoints/all_results.json`
- `summary.json`

### Not exported

- `adapter_model.safetensors`
- `optimizer.pt`
- tokenizer dumps
- local W&B binary payloads

## Recommended daily workflow

1. Prepare or update processed datasets with `sft_data_prepare.py`.
2. Run `run_sft_qwen3_8b_medical_1k.sh` for smoke testing after code or data changes.
3. Run `run_sft_qwen3_8b_huatuo_5w.sh` for the first formal training version.
4. Export reproducibility records with `export_experiment_records.py --all --force`.
5. Commit code and exported experiment records together when the run is worth preserving.
