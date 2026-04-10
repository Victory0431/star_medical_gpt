# DPO Workflow

Chinese primary version: [DPO_WORKFLOW.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.zh-CN.md).

This document explains the current `DPO` pipeline in this repo:

- merge the best `SFT checkpoint-75` back into the base model,
- use that merged `SFT policy` as the DPO initialization point,
- train a fresh `DPO LoRA`,
- monitor both pairwise preference metrics and an auxiliary heterogeneous `valid_zh` LM-loss view.

## Core idea

Recommended stage order:

1. `Base`
2. `SFT LoRA`
3. merge the best `SFT checkpoint`
4. train `DPO LoRA` on top of the merged `SFT model`

This keeps the project story clean:

- `SFT` teaches the model how to answer medical questions
- `DPO` teaches the model which style of answer should be preferred

## Data

Raw medical pairwise data:

- `medical_pairwise_train.jsonl`: `3800`
- `medical_pairwise_valid.jsonl`: `100`
- `medical_pairwise_test.jsonl`: `100`

Processed DPO format:

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "source": "medical_pairwise_train"
}
```

Preparation script:

- [`script/alignment/dpo_data_prepare.py`](/home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py)

Training script:

- [`script/alignment/train_dpo.py`](/home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py)

Launcher:

- [`script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh`](/home/qjh/llm_learning/my_medical_gpt/script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh)

## Training-time evaluation

### Main evaluation

Use pairwise validation as the primary DPO selection signal:

- `eval_rewards/accuracies`
- `eval_rewards/margins`
- `eval_loss`

By default, `eval_rewards/accuracies` is used to pick the best checkpoint.

### Auxiliary heterogeneous evaluation

Use `valid_zh` as an auxiliary `SFT-style` validation set:

- metric: `aux_eval/valid_zh_loss`

This does not replace pairwise preference evaluation. It is only there to monitor whether DPO harms general medical QA quality on a different distribution.

## Standard launch

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh
```

## Outputs

Each DPO run writes:

- `outputs/dpo/<run_name>/checkpoints/`
- `outputs/dpo/<run_name>/final_model/`
- `outputs/dpo/<run_name>/logs/train.log`
- `outputs/dpo/<run_name>/logs/metrics.jsonl`
- `outputs/dpo/<run_name>/logs/aux_eval.jsonl`
- `outputs/dpo/<run_name>/artifacts/best_checkpoint.json`
- `outputs/dpo/<run_name>/artifacts/training_summary.json`

Central timestamped logs are also written under:

- `outputs/logs/dpo/<timestamp>_<run_name>.log`
