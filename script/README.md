# Script Layout

Chinese-first version: [README.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/script/README.zh-CN.md).

The `script/` directory is grouped by stage and responsibility so future `SFT / DPO / GRPO / evaluation / ops` additions do not accumulate as a flat script dump.

## Layout

```text
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
```

## Intent

- `script/sft`
  SFT data prep, SFT training, and daily launcher scripts.
- `script/alignment`
  LoRA merge, DPO data prep, DPO training, and future alignment-stage scripts.
- `script/eval`
  Benchmark launchers kept separate from training entrypoints.
- `script/ops`
  Operational helpers such as experiment export and reporting.
- `script/grpo`
  Reserved for future GRPO and online RL code.
