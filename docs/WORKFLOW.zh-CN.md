# 项目工作流

英文原版见 [WORKFLOW.md](/home/qjh/llm_learning/my_medical_gpt/docs/WORKFLOW.md)。

这份文档描述了当前仓库推荐的工作流程，目标是让项目从原始数据到训练、评测、归档都尽量贴近真实工业化节奏。

## 1. 准备原始数据

原始医疗 `SFT` 数据只保存在本地，不提交到 GitHub。

推荐目录：

```text
data/
  sft/
    raw/
    processed/
```

使用 [sft_data_prepare.py](/home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py) 把原始文件转成统一的 `conversations` JSONL。

## 2. 构建处理后的 train / valid / test 集

当前项目里比较常见的阶段化数据集有：

- `huatuo_1k.processed.jsonl`
- `huatuo_5w.processed.jsonl`
- `huatuo_v1_226k.processed.jsonl`
- `train_zh_195w.processed.jsonl`
- `valid_zh_500.processed.jsonl`
- `test_zh_500.processed.jsonl`

推荐用法：

- `huatuo_1k`：pipeline 冒烟测试
- `huatuo_5w`：首个正式小版本
- 更大数据：后续扩容实验

## 3. 先跑 smoke test

在正式训练前，先跑一版便宜的 smoke test：

```bash
conda activate medicalgpt
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

这一步主要验证：

- 数据加载
- tokenizer 与 chat template
- assistant-only loss mask
- 双卡分布式启动
- `W&B` 记录
- checkpoint 写出

## 4. 再跑首个正式版本

当 smoke test 稳定后，再进入正式训练：

```bash
conda activate medicalgpt
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

这一版通常就是你可以拿去讲面试的第一个“像样版本”。

## 5. 训练中持续评估

当前仓库的约定是：

- train 与 eval 放在同一个训练任务里
- 训练过程中周期性记录 `eval_loss`
- 训练结束后再做一次最终评估

现阶段先用外部验证集例如 `valid_zh_500` 没问题，后面也可以和同分布 holdout 做对比，形成更强的实验叙事。

## 6. 导出实验记录

当某个 run 值得保留时，执行：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```

这会把轻量可复现信息导出到：

- [experiment_records/](/home/qjh/llm_learning/my_medical_gpt/experiment_records)

默认会跳过：

- `dryrun`
- 明显失败的 run

## 7. 跑外部 benchmark

当你已经有稳定的 base run 或 LoRA checkpoint 时，再接入外部 benchmark。

当前推荐顺序：

- 优先使用 `HealthBench consensus`
- 如果 judge key 还没准备好，先跑 `generate_only`
- 后面再对同一个 `responses.jsonl` 跑 `judge_only`

这样做的好处是：

- 本地生成和外部 judge 的失败模式不同
- 重跑成本更低
- 更适合后续多阶段 `SFT / DPO / GRPO` 对比

相关文档：

- [评测设计](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.zh-CN.md)
- [评测模块说明](/home/qjh/llm_learning/my_medical_gpt/evaluation/README.zh-CN.md)

## 8. Git 里应该提交什么

推荐策略：

- 提交代码
- 提交文档
- 提交轻量实验记录
- 不提交原始数据
- 不提交处理后的大数据
- 不提交模型权重和 checkpoint 二进制

## 9. 面试叙事建议

比较自然的一条线是：

1. 把异构医疗指令数据统一成同一种 chat 格式。
2. 用 assistant-only loss 做标准 `SFT`。
3. 先用小数据集完成全链路 smoke test。
4. 再扩到更大的正式数据集，并在训练中周期性做 eval 与 `W&B` 监控。
5. 用开放式医疗 benchmark，而不是只靠医学选择题。
6. 导出轻量实验记录，让每个 run 都可追溯、可复盘。
