# GRPO 训练问题修复记录

本文记录 `2026-04-14` 对 `GRPO v1 emergency/context` 训练中断问题的定位、修复和验证过程。

## 1. 问题现象

`GRPO v1 emergency/context` 在正式训练中能够正常完成前若干个 training step，也能够跑到第一次评估，但会在评估结束后的 best metric / checkpoint 处理阶段崩溃。

典型报错见：

- [20260414_110300 console.log](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_110300_qwen3-8b_dpo330_grpo_v1_emergency/logs/console.log)

核心错误：

```text
KeyError: "The `metric_for_best_model` training argument is set to 'eval_reward', which is not found in the evaluation metrics. The available evaluation metrics are: ['eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']."
```

这导致：

- rank 1 先报错退出
- `torchrun` 随后给另一个 worker 发 `SIGTERM`
- 整轮训练停止

## 2. 根因分析

项目的 `GRPOTrainer` 会在日志里输出丰富的评估指标，例如：

- `eval_reward`
- `eval_rewards/context_awareness_reward/mean`
- `eval_rewards/emergency_referral_reward/mean`

但 `Transformers Trainer` 内部 `_determine_best_metric()` 使用的，并不是这份完整日志字典，而是 `evaluate()` 返回的那份更“瘦”的 metrics。

在当前环境下，这份内部 metrics 里只有：

- `eval_loss`
- `eval_runtime`
- `eval_samples_per_second`
- `eval_steps_per_second`

因此，只要把：

- `metric_for_best_model = eval_reward`

继续传给 `Trainer`，即使：

- `load_best_model_at_end = False`

也仍然会在 `_maybe_log_save_evaluate()` 里触发 `_determine_best_metric()`，最终抛出 `KeyError`。

## 3. 为什么之前修过一次还是会复现

之前的修复只做了半步：

- 关闭了 `load_best_model_at_end`
- 增加了自定义 `BestMetricCallback`

但还保留了：

- `metric_for_best_model = eval_reward`
- `greater_is_better = True`

并继续把这两个参数传入 `GRPOConfig / Trainer`

所以结果变成了：

- 我们自己的 callback 能在 `on_log()` 里记录 `eval_reward`
- 但 `Trainer` 内部仍然会尝试按 `eval_reward` 选 best metric
- 于是第一次 eval 后照样崩

## 4. 代码修复

修复文件：

- [train_grpo.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/train_grpo.py)

本次修复做了三件事：

1. 彻底切断 `Trainer` 内部的 best metric 逻辑
   - `metric_for_best_model=None`
   - `greater_is_better=None`
   - `load_best_model_at_end=False`

2. 保留并强化自定义 best metric 跟踪
   - 仍然使用 `BestMetricCallback`
   - 从 `on_log()` 中捕获 `eval_reward`
   - 将最佳 checkpoint 写入：
     - `artifacts/best_checkpoint.json`

3. 训练收尾时不再依赖 `trainer.state.best_metric`
   - 改为优先使用 `BestMetricCallback.best_payload`
   - 避免训练正常结束后又因为 `trainer.state.best_metric is None` 导致汇总信息不准确

## 5. 烟测验证

为确保不是“代码看起来对”，而是真的跨过了原来会炸的路径，做了一轮单卡小烟测：

- `run_name = 20260414_132800_qwen3-8b_dpo330_grpo_v1_smoke_fix`
- 目录：
  - [smoke run dir](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_132800_qwen3-8b_dpo330_grpo_v1_smoke_fix)

关键配置：

- `CUDA_VISIBLE_DEVICES=1`
- `NPROC_PER_NODE=1`
- `MAX_TRAIN_SAMPLES=8`
- `MAX_EVAL_SAMPLES=4`
- `MAX_STEPS=2`
- `EVAL_INTERVAL=1`
- `SAVE_INTERVAL=1`
- `NUM_GENERATIONS=2`
- `NUM_GENERATIONS_EVAL=2`

验证目标非常明确：

- 每个 step 都触发 `eval`
- 每个 step 都触发 `save`
- 强行经过原来最容易崩的 `eval -> best metric -> save checkpoint` 路径

烟测结果：

- `step 1` 正常训练
- `step 1 eval` 正常完成
- `Best metric updated: eval_reward=...`
- `checkpoint-1` 正常写出
- `step 2` 正常训练
- `step 2 eval` 正常完成
- `checkpoint-2` 正常写出
- 最终：
  - `GRPO training finished`
  - `Best checkpoint: ... checkpoint-2`

相关产物：

- [smoke metrics.jsonl](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_132800_qwen3-8b_dpo330_grpo_v1_smoke_fix/logs/metrics.jsonl)
- [smoke best_checkpoint.json](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_132800_qwen3-8b_dpo330_grpo_v1_smoke_fix/artifacts/best_checkpoint.json)
- [smoke summary.json](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_132800_qwen3-8b_dpo330_grpo_v1_smoke_fix/artifacts/summary.json)

这说明：

- bug 已经真正修复
- 不是“刚好没跑到 eval”
- 也不是“eval 跑了但 save 还没碰到”

## 6. 修复后的结论

可以把这次问题总结为：

- 不是 reward 函数问题
- 不是数据问题
- 不是 GRPOTrainer 本体不能训练
- 而是 `Trainer` 内部 best metric 选择机制和 `GRPO` 自定义日志指标之间的接口不一致

正确做法是：

- 让 `Trainer` 只负责训练 / eval / save
- 让我们自己的 callback 负责追踪 `eval_reward` 这类富指标

## 7. 正式训练复跑状态

在烟测通过后，已重新启动正式双卡训练：

- `run_name = 20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1`
- 启动脚本：
  - [run_grpo_qwen3_8b_dpo330_v1_emergency.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh)
- 运行目录：
  - [full run dir](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1)

启动命令为：

```bash
RUN_NAME=20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1 \
MASTER_PORT=29661 \
CUDA_VISIBLE_DEVICES=0,1 \
setsid bash script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh \
  > outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1.nohup.log 2>&1 < /dev/null &
```

截至 `2026-04-14 13:47`，正式训练状态为：

- 已稳定跑过 `step 10`
- 没有再出现：
  - `KeyError: 'eval_reward'`
  - worker 退出
  - `torchrun` 联动 `SIGTERM`
- 训练进程与 GPU 计算仍处于活跃状态，说明当前已不再是“第一次 eval 结束立刻崩溃”的旧问题

这意味着本次修复已经：

- 在烟测中完整验证了 `eval -> best metric -> save checkpoint` 路径
- 在正式训练中验证了“第一次真实触发旧 bug 的阶段”不再立即崩溃

正式训练最终 best checkpoint / summary 仍以运行目录中的产物为准：

- `artifacts/best_checkpoint.json`
- `artifacts/summary.json`

## 8. 后续训练建议

后续正式训练继续使用：

- [run_grpo_qwen3_8b_dpo330_v1_emergency.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh)

建议保留当前设计：

- `Trainer` 内部不再使用 `metric_for_best_model`
- best checkpoint 统一读：
  - `artifacts/best_checkpoint.json`

后续如果需要 resume / 复盘，优先查看：

- `logs/metrics.jsonl`
- `artifacts/best_checkpoint.json`
- `artifacts/summary.json`

而不是依赖 `trainer.state.best_metric`
