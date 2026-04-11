# DPO V2 运行摘要

对应正式 run：

- `20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo`

核心结论：

- 训练成功跑满 `3` 个 epoch，共 `357` 步
- pairwise `accuracy` 在 `step=30` 就达到 `1.0`
- 但 `margin` 持续增长到 `step=340` 左右才接近峰值
- 异构 `valid_zh_loss` 最优点也出现在 `step=30`

这次实验最重要的结论不是“模型没学会”，而是：

- 当前 `metric_for_best_model=eval_rewards/accuracies` 太容易饱和
- 导致 best checkpoint 被锁在最早达到 `1.0` 的 `checkpoint-30`
- 所以最终 `all_results.json` 反映的是早期 best checkpoint，而不是训练后期最强 pairwise margin 状态

推荐后续动作：

- 用 `checkpoint-30 / checkpoint-230 / checkpoint-330或340` 分别跑 `HealthBench`
- 下次把 `metric_for_best_model` 改成 `eval_rewards/margins` 或组合指标

详细分析见：

- [DPO_V2_TRAINING_REPORT.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_TRAINING_REPORT.zh-CN.md)
- [summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/summary.json)
