# 脚本目录说明

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/script/README.md)。

当前 `script/` 目录按训练阶段与职责拆分，避免后续 `SFT / DPO / GRPO / 评测 / 运维` 脚本继续平铺在一个目录里。

## 当前结构

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

## 分层原则

- `script/sft`
  放 `SFT` 数据处理、训练主程序、日常训练 launcher。
- `script/alignment`
  放 `LoRA merge`、`DPO` 数据处理、`DPO` 训练及后续 `RM / ORPO / PPO / GRPO` 的对齐相关脚本。
- `script/eval`
  放 benchmark 评测 launcher，保持和训练入口解耦。
- `script/ops`
  放实验归档、导表、批量整理这类运维辅助脚本。
- `script/grpo`
  预留给后续 `GRPO / reward shaping / online sampling` 相关代码。

## 为什么这样拆

- 更符合工业项目常见的“按阶段/职责分层”方式。
- 新增 `DPO / ORPO / RM / GRPO` 时，不会把 `script/` 顶层继续堆满。
- 文档和命令路径更清晰，面试时也更容易解释“训练链路”和“评测链路”是如何拆开的。

## 使用建议

- 训练入口优先走各目录下的 `run_*.sh`。
- 需要改超参、调试异常、做新实验时，再直接调用对应的 `train_*.py`。
- 文档统一参考 [docs/SCRIPT_GUIDE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.zh-CN.md)。
