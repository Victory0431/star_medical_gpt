# DPO V2 HealthBench 对比分析

这份文档记录 `DPO v2` 两个有明确实验意义的 checkpoint 在 `HealthBench consensus` 上的正式对比结果，并把它们放回当前主线结果矩阵里统一分析。

关于 `checkpoint-330` 与 `huatuo_5w checkpoint-75` 的第二轮稳定性复测，见：

- [HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md)

对应轻量评测快照见：

- [experiment_records/eval/20260411_healthbench_dpo_v2_ckpt30_gpt52_consensus_theme15x7/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_dpo_v2_ckpt30_gpt52_consensus_theme15x7/summary.json)
- [experiment_records/eval/20260411_healthbench_dpo_v2_ckpt330_gpt52_consensus_theme15x7/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_dpo_v2_ckpt330_gpt52_consensus_theme15x7/summary.json)

对应训练总结见：

- [DPO_V2_TRAINING_REPORT.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_TRAINING_REPORT.zh-CN.md)

## 1. 为什么只测这两个 checkpoint

这次不是“随便挑两个点测一测”，而是先定义实验问题，再选 checkpoint：

- `checkpoint-30`
  - 含义：最保守的 `DPO v2` 版本
  - 训练信号：`eval_rewards/accuracies` 最早饱和到 `1.0`
  - 异构辅助验证：`valid_zh_500` 上的最佳点
  - 想回答的问题：如果外部 benchmark 更看重通用医疗问答保持和沟通稳健性，它会不会更好
- `checkpoint-330`
  - 含义：更强的 pairwise 偏好优化版本
  - 训练信号：接近 `min eval_loss`，且 `margin` 已经很高
  - 想回答的问题：如果外部 benchmark 更看重 `DPO v2` 这批重构数据强化出来的行为，它会不会更好

这两个点刚好对应你项目里最有价值的一个工业化问题：

- `训练内指标最优`
- `异构辅助集最优`
- `外部 benchmark 最优`

它们不一定是同一个 checkpoint。

## 2. 评测设置

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，共 `105` 条
- judge 模型：`gpt-5.2`
- judge API：OpenAI 兼容接口
- 生成模型基座：`Qwen3-8B`
- 对比对象：
  - `checkpoint-30`
  - `checkpoint-330`

本次评测中途真实遇到过一次 judge API 断连，后面已经把评测链路补成了“可断点续判 + 网络抖动重试”的版本，再续跑补齐。

## 3. 先看结果

### DPO v2 内部对比

| 模型 | overall clipped mean | 结论定位 |
| --- | ---: | --- |
| `DPO v2 checkpoint-30` | `0.2492` | 更保守，更强调异构医疗问答保持 |
| `DPO v2 checkpoint-330` | `0.2619` | 更激进，更强调 pairwise 偏好目标 |

结论先说：

- `checkpoint-330` 明确高于 `checkpoint-30`
- 这说明在你这版 `DPO v2` 数据上，外部 `HealthBench` 更认可后期那种更强的 pairwise 优化结果
- 也就是说，这次不是“最保守 checkpoint 更安全地赢了”，而是“更深一点的 DPO 优化确实带来了外部收益”

### 放回当前主线结果矩阵

| 模型 | overall clipped mean |
| --- | ---: |
| `Qwen3-8B base` | `0.2206` |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2508` |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)` | `0.2889` |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-925)` | `0.2587` |
| `Qwen3-8B + DPO v1 medical_pairwise (checkpoint-100)` | `0.2111` |
| `Qwen3-8B + DPO v2 (checkpoint-30)` | `0.2492` |
| `Qwen3-8B + DPO v2 (checkpoint-330)` | `0.2619` |

关键变化：

- 相比 `DPO v1 checkpoint-100`
  - `checkpoint-30` 提升 `+0.0381`
  - `checkpoint-330` 提升 `+0.0508`
- 相比 `base`
  - `checkpoint-30` 提升 `+0.0286`
  - `checkpoint-330` 提升 `+0.0413`
- 相比 `huatuo_1k`
  - `checkpoint-30` 略低 `-0.0016`
  - `checkpoint-330` 略高 `+0.0111`
- 相比 `huatuo_5w checkpoint-925`
  - `checkpoint-330` 略高 `+0.0032`
- 相比当前最强的 `huatuo_5w checkpoint-75`
  - `checkpoint-30` 低 `-0.0397`
  - `checkpoint-330` 低 `-0.0270`

这意味着：

- `DPO v2` 已经明显修复了 `DPO v1` 的失效问题
- 但它还没有超过你当前最强的 `SFT 5w checkpoint-75`

## 4. Axis 层面的含义

| axis | ckpt-30 | ckpt-330 | 谁更强 |
| --- | ---: | ---: | --- |
| `accuracy` | `0.1908` | `0.2105` | `ckpt-330` |
| `communication_quality` | `0.2222` | `0.1556` | `ckpt-30` |
| `completeness` | `0.0833` | `0.1667` | `ckpt-330` |
| `context_awareness` | `0.2963` | `0.3889` | `ckpt-330` |
| `instruction_following` | `0.5000` | `0.4333` | `ckpt-30` |

解读：

- `checkpoint-330` 的优势非常清晰，主要集中在：
  - `accuracy`
  - `completeness`
  - `context_awareness`
- `checkpoint-30` 的优势主要集中在：
  - `communication_quality`
  - `instruction_following`

这和训练阶段的预期基本一致：

- `checkpoint-30` 更像“保守、稳一点的版本”
- `checkpoint-330` 更像“更用力地贴近 pairwise 偏好目标的版本”

## 5. Theme 层面的含义

| theme | ckpt-30 | ckpt-330 | 谁更强 |
| --- | ---: | ---: | --- |
| `communication` | `0.1000` | `0.0333` | `ckpt-30` |
| `complex_responses` | `0.2333` | `0.2333` | 持平 |
| `context_seeking` | `0.0667` | `0.1000` | `ckpt-330` |
| `emergency_referrals` | `0.3000` | `0.4333` | `ckpt-330` |
| `global_health` | `0.2667` | `0.2333` | `ckpt-30` |
| `health_data_tasks` | `0.4667` | `0.4000` | `ckpt-30` |
| `hedging` | `0.3111` | `0.4000` | `ckpt-330` |

这张表特别重要，因为它几乎直接回答了你后面该怎么讲这个项目：

- `checkpoint-330` 提升最明显的，是 `HealthBench` 里最像“医疗对齐行为”的几个主题：
  - `context_seeking`
  - `emergency_referrals`
  - `hedging`
- `checkpoint-30` 保住得更好的，是更像“通用对话和任务表达”的几个主题：
  - `communication`
  - `health_data_tasks`
  - 一部分 `global_health`

所以更准确的表述不是：

- “DPO 后面训久了就更好。”

而是：

- “更深的 DPO 优化把模型往 `HealthBench` 更看重的安全分诊、语境感知和不确定性表达方向推了一步，但同时牺牲了一部分沟通自然度和指令稳健性。”

## 6. 样本级比较说明了什么

按 `105` 条样本逐条对齐后，可以看到各主题的胜负分布：

- `communication`
  - `ckpt-30` 胜 `3`
  - `ckpt-330` 胜 `1`
- `emergency_referrals`
  - `ckpt-30` 胜 `0`
  - `ckpt-330` 胜 `4`
- `hedging`
  - `ckpt-30` 胜 `2`
  - `ckpt-330` 胜 `6`
- `health_data_tasks`
  - `ckpt-30` 胜 `3`
  - `ckpt-330` 胜 `1`

这进一步验证了上面的主题结论。

更具体地说，这次对比里至少有两类很典型的样本：

### 1. `ckpt-330` 更好的样本

例如紧急分诊题：

- 用户问“走几步就明显气短而且嘴唇发青，要不要去急诊”
- `ckpt-330` 被打到 `1.0`
- `ckpt-30` 是 `0.5`

这类题上，`ckpt-330` 更倾向于：

- 明确给出“立刻去急诊”
- 更直接指出低氧、心肺风险
- 语气更像真正的分诊回复

这和它在 `emergency_referrals` 上更强完全一致。

### 2. `ckpt-30` 更好的样本

例如沟通类题：

- 用户只是说“我想问一下 postpartum depression 的几个问题”
- `ckpt-30` 得分 `0.5`
- `ckpt-330` 得分 `0.0`

这里 `ckpt-330` 的问题不是“完全不会答”，而是更容易：

- 过早假设用户接下来真正要问的具体子问题
- 把回答写得过满、过实
- 导致 `instruction_following` 和 `communication_quality` 被扣分

这也是为什么它总分更高，但 `communication` 反而更差。

## 7. 这次实验最重要的结论

### 结论 1：DPO v2 是有效的

这次 `DPO v2` 不再像 `DPO v1` 那样在外部 benchmark 上明显失败，而是已经出现了正式收益：

- 超过 `base`
- 超过 `huatuo_1k`
- 超过 `DPO v1`
- 最好的点还略高于 `huatuo_5w checkpoint-925`

这说明：

- 你对 `3800` 条 pairwise 数据的重构是有价值的
- `label_direction`
- `communication`
- `context_awareness`
- `hedging`
- `emergency_referrals`

这些重构方向，确实把 DPO 从“无效”推到了“开始有正式收益”

### 结论 2：best checkpoint 不能只看训练内指标

训练里：

- `checkpoint-30` 是 `metric_for_best_model=eval_rewards/accuracies` 下的 best
- 也是 `valid_zh_500` 的 best

但外部 `HealthBench` 上：

- `checkpoint-330` 更强

这就是一个非常标准、也非常适合面试展开的工业经验点：

- 单一训练内指标不够
- 异构辅助集也不够
- 最终仍要靠统一外部 benchmark 闭环来拍板

### 结论 3：DPO v2 还没超过最强 SFT

虽然 `DPO v2` 已经明显变好了，但最强点 `0.2619` 仍低于：

- `huatuo_5w checkpoint-75 = 0.2889`

所以现阶段更合理的主线不是：

- “宣布 DPO 已经全面替代 SFT”

而是：

- “SFT 仍然是当前主战模型，DPO v2 开始在特定行为维度上带来外部收益，但还需要进一步把沟通质量和任务稳健性补回来。”

## 8. 下一步怎么做最合理

最值得继续推进的方向不是盲目再拉长 DPO，而是：

1. 保留 `huatuo_5w checkpoint-75` 作为当前主 baseline。
2. 把 `DPO v2 checkpoint-330` 视为当前最有潜力的偏好对齐版本。
3. 下一轮数据改造重点补：
   - `communication`
   - `instruction_following`
   - `health_data_tasks`
4. 如果进入 `GRPO`，奖励函数优先围绕：
   - `context_awareness`
   - `hedging`
   - `emergency_referrals`
   - `communication_quality`

这条路线最顺，因为它和你当前 `HealthBench` 暴露出的短板是严格对齐的。

## 9. 面试里怎么讲会更强

更强的说法不是：

- “我把 DPO 跑出来了，后面 checkpoint-330 分更高。”

而是：

- “我先把 3800 条 pairwise 数据按医疗正确性、沟通方式、风险提示和上下文感知做了重构，再用同一套 `HealthBench consensus 7 x 15` 做统一验收。结果说明 DPO v2 已经明显优于旧 DPO，且外部 benchmark 更偏好后期的 `checkpoint-330`，因为它在 `context_seeking / emergency_referrals / hedging` 上有正式收益；但它同时牺牲了一部分 `communication_quality` 和 `instruction_following`。所以我最后没有用单一训练指标拍板，而是把它放回统一 benchmark 闭环里做多信号决策。” 

这比“我会跑 DPO”更像真实工业实践。
