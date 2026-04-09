# Workflow Guide

This document describes the intended working flow for this repository.

## 1. Prepare raw data

Raw medical SFT data is stored locally and not committed to GitHub.

Recommended local layout:

```text
data/
  sft/
    raw/
    processed/
```

Use [`sft_data_prepare.py`](/home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py) to convert raw files into unified `conversations` JSONL.

## 2. Build processed train and validation sets

For the current project, the common staged datasets are:

- `huatuo_1k.processed.jsonl`
- `huatuo_5w.processed.jsonl`
- `huatuo_v1_226k.processed.jsonl`
- `train_zh_195w.processed.jsonl`
- `valid_zh_500.processed.jsonl`
- `test_zh_500.processed.jsonl`

Recommended usage:

- `huatuo_1k` for smoke test and pipeline verification
- `huatuo_5w` for first formal SFT version
- larger datasets for later scaling experiments

## 3. Run a smoke test first

Before formal training, always run a cheap smoke test:

```bash
conda activate medicalgpt
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

This verifies:

- data loading
- tokenizer and chat template
- assistant-only loss masking
- 2-GPU distributed launch
- W&B logging
- checkpoint writing

## 4. Run the first formal version

After the smoke test is stable:

```bash
conda activate medicalgpt
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

This is the version you can usually treat as the first interview-grade training run.

## 5. Track evaluation during training

Current project convention:

- train and eval are run together in one training job
- `eval_loss` is measured periodically during training
- final evaluation is run again after training ends

For now, external validation such as `valid_zh_500` is acceptable as a first baseline.
Later you can compare it with same-distribution holdout splits for a stronger experimental story.

## 6. Export experiment records

After a run becomes worth keeping:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```

This exports only lightweight reproducibility artifacts into:

- [`experiment_records/`](/home/qjh/llm_learning/my_medical_gpt/experiment_records)

By default it skips:

- `dryrun` directories
- obvious failed runs

## 7. Commit what belongs in Git

Recommended Git policy:

- commit code
- commit documentation
- commit lightweight experiment records
- do not commit raw datasets
- do not commit processed datasets
- do not commit model weights or checkpoint binaries

## 8. Suggested project narrative

For interview discussion, the most natural story is:

1. Normalize heterogeneous medical instruction data into a single chat format.
2. Use assistant-only loss to perform standard SFT.
3. Start with a small smoke-test dataset to validate the whole pipeline.
4. Scale to a larger formal dataset with periodic evaluation and W&B monitoring.
5. Export lightweight experiment records so each training run is traceable and reviewable.
