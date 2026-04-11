# DPO V2 训练总结

这份文档总结 `20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo` 这次 `DPO v2` 正式训练的训练数据、训练过程、关键指标变化，以及这次实验最重要的结论。

对应轻量实验记录见：

- [experiment_records/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo/summary.json)

对应完整本地 run 目录见：

- [outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo)

## 1. 先说结论

这次 `DPO v2` 训练本身是成功的，模型在 pairwise 验证集上的偏好区分能力显著增强，训练过程也没有出现发散或异常震荡。

但这次实验同时暴露出一个很重要的问题：

- 当前 `metric_for_best_model=eval_rewards/accuracies`
- 而 `valid` 只有 `100` 条，并且和训练数据同分布、难度偏低
- 导致 `eval_rewards/accuracies` 在 `step=30` 就饱和到 `1.0`
- 之后虽然 `margin` 还在持续上升，但 best checkpoint 选择逻辑已经失去区分能力

因此，这次 run 的最大收获不是“DPO 一路完美起飞”，而是我们定位了一个真实的工业化问题：

- `训练有效`
- `best checkpoint 选择指标不够好`

这点其实很适合面试时展开，因为它说明你不是只会“把脚本跑完”，而是能发现训练闭环里真正的瓶颈。

## 2. 训练数据

### 初始化模型

- 基座不是原始 base model
- 使用的是 `SFT checkpoint-75 merge` 后的模型

路径：

- [outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model](/home/qjh/llm_learning/my_medical_gpt/outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model)

### 数据集组成

- `train = 3800`
- `valid = 100`
- `aux_valid = 500`

路径：

- [medical_pairwise_train_v2.processed.jsonl](/home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo_v2/train/medical_pairwise_train_v2.processed.jsonl)
- [medical_pairwise_valid_v2.processed.jsonl](/home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo_v2/valid/medical_pairwise_valid_v2.processed.jsonl)
- [valid_zh_500.processed.jsonl](/home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl)

其中：

- `valid` 是 `DPO` 主评估集，直接衡量 `chosen > rejected` 是否成立
- `aux_valid` 是异构分布的开放式医疗问答验证集，用来观察 `DPO` 是否损伤一般医疗问答能力

### DPO V2 数据特点

这批 `DPO v2` 数据不是原始 pairwise，而是先做过一次重构。

参考文档：

- [DPO_V2_RECONSTRUCTION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_RECONSTRUCTION.zh-CN.md)

关键重构统计：

- `3800 / 3800` 条都被修改过
- `swap_count = 2198`
- 高频修正问题包括：
  - `poor_communication`
  - `missing_context_awareness`
  - `factual_risk`
  - `overconfidence`
  - `label_direction_wrong`

这意味着本次训练的优化目标，已经不再是“让模型更像旧论坛医疗回答”，而是更贴近：

- `accuracy`
- `communication_quality`
- `context_awareness`
- `hedging`
- `emergency_referrals`

## 3. 训练配置

本次正式参数：

- `2 x H200`
- `bf16`
- `LoRA r=16`
- `LoRA alpha=32`
- `LoRA dropout=0.05`
- `beta=0.1`
- `loss_type=sigmoid`
- `num_train_epochs=3`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `eval_steps=10`
- `save_steps=10`
- `metric_for_best_model=eval_rewards/accuracies`
- `flash-attn=off`

全局有效 batch 可以这样算：

- `2` 卡
- 每卡 `2` 条
- 梯度累计 `8`

所以每次参数更新约等价于：

- `2 x 2 x 8 = 32` 条 pairwise 样本

在 `3800` 条训练集上，大约：

- `3800 / 32 ≈ 118.75`
- 所以每个 epoch 约 `119` 个 optimizer step
- `3` 个 epoch 总共 `357` 个 step

这和最终训练步数完全吻合。

## 4. 训练过程

### 总体运行情况

- 训练总步数：`357`
- 最终 epoch：`3.0`
- 训练时长：`2166.7s`，约 `36.1` 分钟
- `train_steps_per_second = 0.165`
- `train_samples_per_second = 5.261`

### 早期阶段

训练初期模型很快学会区分 `chosen / rejected`：

- `step=10`
  - `eval_rewards/accuracies = 0.63`
  - `eval_rewards/margins = 0.024`
- `step=20`
  - `eval_rewards/accuracies = 0.97`
  - `eval_rewards/margins = 0.137`
- `step=30`
  - `eval_rewards/accuracies = 1.0`
  - `eval_rewards/margins = 0.403`

也就是说，这个 `valid` 集在 `step=30` 就已经被“做穿了”。

### 中后期阶段

虽然 `accuracy` 饱和，但 `margin` 和 `eval_loss` 还在持续变好：

- `step=100`
  - `eval_loss = 0.0185`
  - `eval_rewards/margins = 4.849`
- `step=200`
  - `eval_loss = 0.0070`
  - `eval_rewards/margins = 6.014`
- `step=300`
  - `eval_loss = 0.00528`
  - `eval_rewards/margins = 6.331`

本次 run 的两个更有区分度的拐点是：

- `min eval_loss @ step=330`
  - `eval_loss = 0.005195`
- `max eval_rewards/margins @ step=340`
  - `eval_rewards/margins = 6.353`

这说明从纯 `DPO` 主目标看，训练并没有在 `step=30` 停止进步，而是一直优化到了大约 `330-340`。

## 5. 异构分布辅助评估

异构 `valid_zh_500` 的 `assistant-only loss` 变化如下：

- `best valid_zh_loss @ step=30`
  - `2.5100`
- `step=100`
  - `2.5520`
- `step=200`
  - `2.5595`
- `step=300`
  - `2.5624`

趋势很清楚：

- `step=30` 之后，`valid_zh_loss` 基本一路上升
- 说明随着偏好优化变强，普通开放式医疗问答分布上的语言建模质量在缓慢下降

这正是 DPO 常见的“目标优化更强，但异构泛化变差”的现象。

所以从异构分布角度看：

- `step=30` 最保守
- `step=330/340` 最偏向 pairwise 偏好目标

## 6. 为什么最终 best checkpoint 是 step 30

当前脚本用的是：

- `metric_for_best_model = eval_rewards/accuracies`

问题在于：

- `accuracy` 在 `step=30` 就已经达到 `1.0`
- 后续 `step=40 ~ 357` 也基本全是 `1.0`

在这种情况下，Trainer 没法继续区分后续 checkpoint，只会保留“最早达到最好值”的那个。

所以最终出现了下面这个结果：

- `training_summary.json` 里的 best checkpoint 是 `checkpoint-30`
- `all_results.json / eval_results.json` 也对应这个被重新加载的早期 checkpoint

这就是为什么你会看到一种“看起来矛盾”的现象：

- 训练中后期 `margin` 明明在持续上涨
- 但最终 `all_results.json` 却回到了 `step=30` 那种较小的 margin

这不是训练崩了，而是 `load_best_model_at_end` 加上“不合适的 best metric” 导致的。

## 7. 这次实验最重要的结论

### 结论 1：DPO V2 数据确实起作用了

从训练曲线看，`DPO v2` 数据是有效的：

- `chosen reward` 持续升高
- `rejected reward` 持续降低
- `margin` 从 `0.024` 拉到 `6.353`
- 没有出现明显训练不稳定

这说明重构后的 pairwise 数据，比旧 `DPO v1` 数据更适合作为偏好学习输入。

### 结论 2：当前主评估集太容易

`100` 条同分布 pairwise `valid` 集过于容易，导致：

- `accuracy` 太早饱和
- 无法承担 best checkpoint 选择职责

所以它更适合作为：

- “最低门槛正确性检查”

而不适合作为：

- “唯一 best model 选择指标”

### 结论 3：这次真正暴露出来的是 checkpoint 选择问题

当前训练最需要改的，不是 LoRA 参数，也不是 batch size，而是：

- `metric_for_best_model`
- `valid` 集构造方式

这比简单说“loss 降了没降”更接近真实工程问题。

## 8. 下一步建议

最推荐的下一步不是立刻重训，而是先把现有 run 用好。

### 建议 1

对这几个 checkpoint 分别跑 `HealthBench`：

- `checkpoint-30`
- `checkpoint-230`
- `checkpoint-330` 或 `checkpoint-340`

原因：

- `checkpoint-30` 代表“最保守，最接近通用问答保持”
- `checkpoint-330/340` 代表“pairwise 目标最强”
- 只有拉到外部 benchmark 上，才能知道哪种更符合你真正想优化的医疗行为

### 建议 2

下次重训时，把 `metric_for_best_model` 改成下面之一：

- `eval_rewards/margins`
- `eval_loss`
- 自定义组合指标

如果你的目标是“既保留普通医疗问答质量，又做偏好优化”，那更好的策略通常是：

- 主指标看 `eval_rewards/margins`
- 辅助监控 `aux_eval/valid_zh_loss`
- 最终再用 `HealthBench` 做外部选择

### 建议 3

后续可以把 `valid` 从现在的 `100` 条小集升级成更难的 holdout：

- 增大样本量
- 增加更难的 chosen/rejected
- 引入更贴近 `HealthBench` 行为目标的 pairwise 验证样本

## 9. 面试时可以怎么讲

你可以这样说：

- “这次 DPO v2 训练本身是成功的，模型很快学会把 chosen 排在 rejected 前面，pairwise reward margin 也持续上升。”
- “但我在复盘时发现，当前 best checkpoint 选择指标用的是 reward accuracy，而这个指标在小验证集上过早饱和，导致 trainer 把 step-30 锁成 best model。”
- “所以我没有把这次实验简单包装成成功，而是进一步把问题拆成两部分：一部分是 DPO 数据本身有效，另一部分是验证集和 best-model 选择策略需要升级。”
- “这更像真实工业项目，因为很多时候不是模型没学会，而是评估设计先成了瓶颈。”

这段表述会比单纯说“我把 DPO 跑通了”更有说服力。
