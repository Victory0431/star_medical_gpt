# 评测结果解读

英文原版见 [EVAL_RESULTS.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.md)。

这份文档记录了当前最值得在面试里讨论的一组 benchmark 快照。

## 2026-04-15 更新：`3671` 条 full consensus 正式结果与稳定性复核

这次新增的是两组最完整口径的正式结果：

- `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)`
- `Qwen3-8B + DPO v2 (checkpoint-330)`

统一口径：

- benchmark：`HealthBench consensus`
- 样本数：`3671`
- 采样方式：`sequential`
- 随机种子：`42`
- judge 模型：`gpt-5.2`

对应 run：

- [SFT 5w full consensus](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_consensus_from_full_sft5w_dpo330_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_consensus_all_seed42)
- [DPO v2 ckpt330 full consensus](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_consensus_from_full_sft5w_dpo330_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_consensus_all_seed42)

### `3671` 条总结果表

| 模型 | overall | 相对 `sft_5w_ckpt75` | 备注 |
| --- | ---: | ---: | --- |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)` | `0.2731` | `0.0000` | 当前最完整 SFT 正式结果 |
| `Qwen3-8B + DPO v2 (checkpoint-330)` | `0.2518` | `-0.0213` | 当前最完整 DPO 正式结果 |

这轮 full 结果给出的核心信息非常重要：

- 在最完整的 `3671` 条口径下，`SFT 5w` 明显高于 `DPO330`
- 前面的 `105` 和 `700` 都没有把这个差距完全拉开
- 因此当前最稳妥的项目结论应该更新为：
  - `DPO330` 在若干医疗行为主题上有结构化优势
  - 但如果只看 full consensus 总分，它仍然没有超过当前主 SFT baseline

### `3671` 条主题分数表

| theme | sft_5w_ckpt75 | dpo_v2_ckpt330 | `DPO - SFT` |
| --- | ---: | ---: | ---: |
| `Expertise-tailored communication` | `0.0483` | `0.0918` | `+0.0436` |
| `Response depth` | `0.2284` | `0.2160` | `-0.0123` |
| `Context seeking` | `0.1189` | `0.1324` | `+0.0135` |
| `Emergency referrals` | `0.2395` | `0.2991` | `+0.0596` |
| `Global health` | `0.4219` | `0.2500` | `-0.1719` |
| `Health data tasks` | `0.3810` | `0.4266` | `+0.0456` |
| `Responding under uncertainty` | `0.4468` | `0.3788` | `-0.0680` |

### `3671` 条 axis 分数表

| axis | sft_5w_ckpt75 | dpo_v2_ckpt330 | `DPO - SFT` |
| --- | ---: | ---: | ---: |
| `accuracy` | `0.2297` | `0.2139` | `-0.0158` |
| `communication_quality` | `0.2570` | `0.1772` | `-0.0798` |
| `completeness` | `0.0815` | `0.1442` | `+0.0627` |
| `context_awareness` | `0.3499` | `0.3459` | `-0.0040` |
| `instruction_following` | `0.4595` | `0.4911` | `+0.0316` |

这两张 full 表说明得很清楚：

- `DPO330` 仍然稳定更强的地方：
  - `Expertise-tailored communication`
  - `Context seeking`
  - `Emergency referrals`
  - `Health data tasks`
  - `completeness`
  - `instruction_following`
- `DPO330` 仍然稳定更弱的地方：
  - `Global health`
  - `Responding under uncertainty`
  - `communication_quality`
  - `accuracy`

其中最关键的拉分项就是：

- `Global health = -0.1719`
- `communication_quality = -0.0798`
- `Responding under uncertainty = -0.0680`

这三个点基本解释了为什么 `DPO330` 在 `3671` 条 full consensus 下没能守住前面小样本里那种“接近甚至偶尔反超”的位置。

### `105 / 700 / 3671` 三档稳定性总表

| 模型 | `105` | `700` | `3671` | `max-min` | 当前结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `SFT 5w checkpoint-75` | `0.2619` | `0.2607` | `0.2731` | `0.0124` | 大样本/full 下略有抬升，整体稳定 |
| `DPO v2 checkpoint-330` | `0.2619` | `0.2614` | `0.2518` | `0.0101` | 小样本看似追平，full 下回落 |
| `GRPO v1 checkpoint-60` | `0.3143` | `-` | `打分中` | `-` | 已有强正向小样本信号，full 结果待确认 |

### `DPO330 - SFT5w` 在三档样本上的变化

| 对比项 | `105` | `700` | `3671` |
| --- | ---: | ---: | ---: |
| `overall (DPO - SFT)` | `0.0000` | `+0.0007` | `-0.0213` |

这个表非常值得记住，因为它直接回答了“前面几轮到底稳不稳”：

- `105` 条时，`DPO330` 和 `SFT5w` 几乎完全打平
- `700` 条时，`DPO330` 仍然只是非常轻微地领先 `+0.0007`
- 到 `3671` 条 full consensus 时，差距被拉开成 `SFT5w +0.0213`

也就是说：

- 前两轮并不是“完全错了”
- 但它们确实低估了 `SFT 5w` 在 full consensus 上的最终领先幅度
- 因此后续如果要做路线决策，`3671` 条这轮应该拥有最高优先级

### 为什么 `3671` 条结果和 `105 / 700` 不一样

从 `700 -> 3671` 的变化看，有几个趋势非常清楚。

`SFT 5w` 在 full 下进一步变强的点：

- `accuracy`: `+0.0468`
- `context_awareness`: `+0.0279`
- `Context seeking`: `+0.0289`
- `Responding under uncertainty`: `+0.0768`

`DPO330` 在 full 下相对 `700` 没有继续扩大的点：

- `Emergency referrals`: `-0.0409`
- `Global health`: `-0.0700`
- `communication_quality`: `-0.0261`

这说明 full consensus 更像在放大模型的“长期稳定行为”而不是少量高亮主题：

- `DPO330` 的优势主题确实存在
- 但 `SFT 5w` 在 `Global health / uncertainty / communication_quality` 这些覆盖面更大的维度上更稳
- 样本一旦扩大到 `3671`，这些稳定收益就会把总分重新拉回去

### 截至 2026-04-15 的最合理判断

把三轮结果合起来看，当前最稳妥的判断应该是：

1. `SFT 5w` 仍然是当前更强的正式 baseline。
   尤其是在 `3671` 条 full consensus 口径下，这个结论已经比前面的 `105 / 700` 更可信。

2. `DPO330` 的价值没有消失，但它的价值应被定义为“结构化行为收益”，不是“当前总分最优解”。
   它最值得保留的强项仍是：
   - `Emergency referrals`
   - `Context seeking`
   - `Expertise-tailored communication`
   - `Health data tasks`

3. `DPO330` 的主要短板也进一步坐实了：
   - `Global health`
   - `Responding under uncertainty`
   - `communication_quality`

4. 这也反过来说明，`GRPO` 的主任务仍然很明确：
   不是重新学 `emergency/context`，而是把这些 `DPO` 已有强项保住，同时把
   - `Global health`
   - `communication_quality`
   - `uncertainty`
   这几个 full consensus 下真正拉总分的点补上。

### GRPO full consensus 当前状态

`GRPO v1 checkpoint-60` 的 `3671` 条 full consensus 目前状态是：

- `responses.jsonl = 3671`
- `judge_only` 已于 `2026-04-15 12:09` 启动
- 当前正在打分中

对应 run：

- [20260414_healthbench_grpo_v1_ckpt60_gpt-52_consensus_all_seed42](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260414_healthbench_grpo_v1_ckpt60_gpt-52_consensus_all_seed42)
- [judge log](/home/qjh/llm_learning/my_medical_gpt/evaluation/logs/20260415_grpo_ckpt60_full_judge.nohup.log)

## 2026-04-14 更新：GRPO v1 `checkpoint-60` 浅测并入结果矩阵

这次新增的是 `GRPO v1 checkpoint-60` 的一轮快速外部评测，用来验证训练内最优点是否能兑现到 `HealthBench`。

先说明口径，避免和后面的 `700` 条正式结果混淆：

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，共 `105` 条
- 随机种子：`42`
- judge 模型：`gpt-5.2`
- 生成模型：`SFT 5w merged backbone + GRPO v1 checkpoint-60 adapter`

对应 run：

- [20260414_healthbench_grpo_v1_ckpt60_gpt52_consensus_theme15x7](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260414_healthbench_grpo_v1_ckpt60_gpt52_consensus_theme15x7)
- [summary.json](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260414_healthbench_grpo_v1_ckpt60_gpt52_consensus_theme15x7/summary.json)

### 105 条总结果表

| 模型 | overall | 相对 base | 相对 `sft_5w_ckpt75` | 相对 `dpo_v2_ckpt330` | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Qwen3-8B base` | `0.2206` | `0.0000` | `-0.0413` | `-0.0413` | 105 条浅测基线 |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2508` | `+0.0302` | `-0.0111` | `-0.0111` | 小规模 SFT |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)` | `0.2619` | `+0.0413` | `0.0000` | `0.0000` | 旧主 SFT baseline |
| `Qwen3-8B + DPO v2 (checkpoint-330)` | `0.2619` | `+0.0413` | `0.0000` | `0.0000` | 当前主 DPO baseline |
| `Qwen3-8B + HQ-50k SFT best` | `0.2905` | `+0.0698` | `+0.0286` | `+0.0286` | 当前强 SFT 对照 |
| `Qwen3-8B + GRPO v1 (checkpoint-60)` | `0.3143` | `+0.0937` | `+0.0524` | `+0.0524` | `2026-04-14` 新增 |

这张总表的直接结论是：

- 在这轮 `105` 条浅测里，`GRPO v1 checkpoint-60 = 0.3143` 已经排到当前第一
- 它不仅高于 `SFT 5w / DPO330`，也高于当前强 SFT 对照 `HQ-50k best`
- 但这仍然只是小样本快速验证，更适合拿来确认“方向成立”，不应替代后面的 `700` 条正式口径

### 105 条主题分数总表

| theme | base | sft_1k | sft_5w_ckpt75 | dpo_v2_ckpt330 | hq50k_best | grpo_v1_ckpt60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Expertise-tailored communication` | `0.0333` | `0.1333` | `0.0000` | `0.0333` | `0.0000` | `0.1333` |
| `Response depth` | `0.2333` | `0.2333` | `0.3000` | `0.2333` | `0.2667` | `0.2000` |
| `Context seeking` | `0.0333` | `0.0333` | `0.1000` | `0.1000` | `0.0667` | `0.1667` |
| `Emergency referrals` | `0.2667` | `0.3000` | `0.1667` | `0.4333` | `0.2667` | `0.3000` |
| `Global health` | `0.2667` | `0.2667` | `0.4333` | `0.2333` | `0.5333` | `0.5000` |
| `Health data tasks` | `0.4667` | `0.4333` | `0.5000` | `0.4000` | `0.4333` | `0.5000` |
| `Responding under uncertainty` | `0.2444` | `0.3556` | `0.3333` | `0.4000` | `0.4667` | `0.4000` |

### 105 条 axis 分数总表

| axis | base | sft_1k | sft_5w_ckpt75 | dpo_v2_ckpt330 | hq50k_best | grpo_v1_ckpt60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `accuracy` | `0.1711` | `0.2039` | `0.1667` | `0.2105` | `0.2434` | `0.2566` |
| `communication_quality` | `0.2000` | `0.2444` | `0.3111` | `0.1556` | `0.3111` | `0.2667` |
| `completeness` | `0.0000` | `0.0833` | `0.0000` | `0.1667` | `0.0833` | `0.0833` |
| `context_awareness` | `0.2407` | `0.2778` | `0.2857` | `0.3889` | `0.2963` | `0.3889` |
| `instruction_following` | `0.5000` | `0.4667` | `0.6000` | `0.4333` | `0.5667` | `0.5667` |

### GRPO 对 `SFT 5w` / `DPO330` 的主题增量对比

| theme | `sft_5w_ckpt75` | `grpo_v1_ckpt60` | `GRPO - SFT` | `dpo_v2_ckpt330` | `GRPO - DPO` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Expertise-tailored communication` | `0.0000` | `0.1333` | `+0.1333` | `0.0333` | `+0.1000` |
| `Response depth` | `0.3000` | `0.2000` | `-0.1000` | `0.2333` | `-0.0333` |
| `Context seeking` | `0.1000` | `0.1667` | `+0.0667` | `0.1000` | `+0.0667` |
| `Emergency referrals` | `0.1667` | `0.3000` | `+0.1333` | `0.4333` | `-0.1333` |
| `Global health` | `0.4333` | `0.5000` | `+0.0667` | `0.2333` | `+0.2667` |
| `Health data tasks` | `0.5000` | `0.5000` | `+0.0000` | `0.4000` | `+0.1000` |
| `Responding under uncertainty` | `0.3333` | `0.4000` | `+0.0667` | `0.4000` | `+0.0000` |

从主题表看，`GRPO v1 checkpoint-60` 最像是把下面几块补起来了：

- 相对 `SFT 5w`：
  - `Expertise-tailored communication`
  - `Emergency referrals`
  - `Context seeking`
  - `Global health`
- 相对 `DPO330`：
  - `Global health`
  - `Expertise-tailored communication`
  - `Health data tasks`
  - `Context seeking`

但它这轮也还保留了明显代价：

- `Response depth` 低于 `SFT 5w`
- `Emergency referrals` 仍低于当前更激进的 `DPO330`

### GRPO 对 `SFT 5w` / `DPO330` 的 axis 增量对比

| axis | `sft_5w_ckpt75` | `grpo_v1_ckpt60` | `GRPO - SFT` | `dpo_v2_ckpt330` | `GRPO - DPO` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `accuracy` | `0.1667` | `0.2566` | `+0.0899` | `0.2105` | `+0.0461` |
| `communication_quality` | `0.3111` | `0.2667` | `-0.0444` | `0.1556` | `+0.1111` |
| `completeness` | `0.0000` | `0.0833` | `+0.0833` | `0.1667` | `-0.0833` |
| `context_awareness` | `0.2857` | `0.3889` | `+0.1032` | `0.3889` | `+0.0000` |
| `instruction_following` | `0.6000` | `0.5667` | `-0.0333` | `0.4333` | `+0.1333` |

这张 axis 表说明得更直观：

- 相对 `SFT 5w`，`GRPO` 这轮最大收获在
  - `accuracy`
  - `context_awareness`
  - `completeness`
- 相对 `DPO330`，`GRPO` 这轮最大收获在
  - `communication_quality`
  - `instruction_following`
  - `accuracy`
- 这也解释了为什么它总分能往上走：
  - 它并不是简单复制 `SFT` 或 `DPO`
  - 而是在两者之间做出了一种更均衡的补齐

### 2026-04-14 这轮更新该怎么理解

如果只看这次 `105` 条结果，当前最合理的理解是：

- `GRPO v1 checkpoint-60` 已经第一次拿到了“外部 benchmark 小样本领先”的明确信号
- 它补上了此前 `DPO330` 最需要补的 `communication / global_health / instruction_following` 中的一大部分
- 它保留了 `DPO` 带来的 `context / accuracy` 优势
- 但 `emergency` 还没有形成对 `DPO330` 的继续领先，`Response depth` 还有回撤

因此更严谨的结论不是“GRPO 已经彻底通关”，而是：

- `GRPO` 路线已经被证明有效
- 下一步应该做的是更大样本复测，确认这轮 `0.3143` 不是小样本波动
- 如果大样本还能保持领先，那么这条 `SFT -> DPO -> GRPO` 链路就真正闭环了

## 2026-04-12 大样本 700 条正式更新

在修复队列挂起方式、response cache 复用逻辑和 judge 重试逻辑后，我重新跑完了同口径的 `700` 条大样本评测批次。到 `2026-04-13` 为止，这一批 `6` 个模型都已经完成正式 judge，可以作为当前最可靠的一组 `HealthBench consensus` 决策基准。

纳入模型：

- `Qwen3-8B base`
- `Qwen3-8B + huatuo_1k LoRA`
- `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)`
- `Qwen3-8B + DPO v2 (checkpoint-30)`
- `Qwen3-8B + DPO v2 (checkpoint-330)`
- `Qwen3-8B + HQ-50k SFT best`

统一口径：

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `100` 条，共 `700` 条
- 随机种子：`42`
- judge 模型：`gpt-5.2`
- 每个 run 的 `num_judged_examples` 都是 `700`

### 总表先看

| 模型 | overall clipped mean | 相对 base | 相对 `sft_5w_ckpt75` | 备注 |
| --- | ---: | ---: | ---: | --- |
| `Qwen3-8B base` | `0.2205` | `0.0000` | `-0.0402` | 大样本基线 |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2288` | `+0.0083` | `-0.0319` | 小规模 SFT / smoke test |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)` | `0.2607` | `+0.0402` | `0.0000` | 旧主 SFT baseline |
| `Qwen3-8B + DPO v2 (checkpoint-30)` | `0.2398` | `+0.0193` | `-0.0209` | 更保守的 DPO 点 |
| `Qwen3-8B + DPO v2 (checkpoint-330)` | `0.2614` | `+0.0410` | `+0.0007` | 当前最强 DPO 点，和 `SFT 5w` 基本同档 |
| `Qwen3-8B + HQ-50k SFT best` | `0.2767` | `+0.0562` | `+0.0160` | 当前这批 `700` 条里总分最高 |

这张表最值得先记住的结论是：

- `HQ-50k best = 0.2767`，已经成为当前这批 `700` 条大样本口径下的新总分第一
- `DPO v2 checkpoint-330 = 0.2614`，已经追平到 `huatuo_5w checkpoint-75 = 0.2607` 附近
- `DPO v2 checkpoint-330` 的优势不是“全面更强”，而是集中在更像医疗行为对齐的主题和 axis 上
- 这批结果让三条路线的角色更清楚了：
  - `5w SFT`：稳定强基线
  - `DPO v2`：更偏行为强化和问诊流程控制
  - `HQ-50k SFT`：当前总分最强的高质量数据路线

### 700 条主题分数表

| theme | base | sft_1k | sft_5w_ckpt75 | dpo_v2_ckpt30 | dpo_v2_ckpt330 | hq50k_best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Expertise-tailored communication` | `0.0300` | `0.0650` | `0.0350` | `0.0550` | `0.0900` | `0.0450` |
| `Response depth` | `0.2000` | `0.1800` | `0.2150` | `0.2100` | `0.2000` | `0.2200` |
| `Context seeking` | `0.1000` | `0.1150` | `0.0900` | `0.1050` | `0.1250` | `0.0950` |
| `Emergency referrals` | `0.2250` | `0.1700` | `0.2750` | `0.2350` | `0.3400` | `0.2700` |
| `Global health` | `0.2550` | `0.2700` | `0.4650` | `0.2950` | `0.3200` | `0.4850` |
| `Health data tasks` | `0.4100` | `0.4550` | `0.3750` | `0.4250` | `0.4050` | `0.3950` |
| `Responding under uncertainty` | `0.3233` | `0.3467` | `0.3700` | `0.3533` | `0.3500` | `0.4267` |

### 700 条 axis 分数表

| axis | base | sft_1k | sft_5w_ckpt75 | dpo_v2_ckpt30 | dpo_v2_ckpt330 | hq50k_best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `accuracy` | `0.1533` | `0.1762` | `0.1829` | `0.1686` | `0.2143` | `0.2152` |
| `communication_quality` | `0.1733` | `0.1700` | `0.2867` | `0.2200` | `0.2033` | `0.2867` |
| `completeness` | `0.0274` | `0.0411` | `0.1233` | `0.0822` | `0.1233` | `0.1096` |
| `context_awareness` | `0.3164` | `0.2797` | `0.3220` | `0.2966` | `0.3362` | `0.3164` |
| `instruction_following` | `0.4700` | `0.5250` | `0.4550` | `0.5050` | `0.4600` | `0.4650` |

### 这轮 700 条最重要的分析

#### 1. `HQ-50k best` 现在是新的最强 SFT baseline

这次最大的新增信息不是 `DPO`，而是 `HQ-50k best` 正式并入矩阵后，把当前总分第一抬到了 `0.2767`。

相对 `huatuo_5w checkpoint-75 = 0.2607`，它的提升主要集中在：

- `Responding under uncertainty`：`0.4267 vs 0.3700`，高 `+0.0567`
- `Global health`：`0.4850 vs 0.4650`，高 `+0.0200`
- `Health data tasks`：`0.3950 vs 0.3750`，高 `+0.0200`
- `axis:accuracy`：`0.2152 vs 0.1829`，高 `+0.0324`
- `axis:instruction_following`：`0.4650 vs 0.4550`，高 `+0.0100`

同时它基本没有在 `communication_quality` 上回退：

- `axis:communication_quality = 0.2867`
- 和 `huatuo_5w checkpoint-75` 持平

这说明高质量 SFT 数据筛选这条线不是“看起来更干净”，而是真的在更大样本 benchmark 上给出了正式收益。

#### 2. `DPO v2 checkpoint-330` 的定位也被重新定义了

在旧的 `105` 条轻量结果里，`DPO v2 checkpoint-330` 仍落后于 `huatuo_5w checkpoint-75`。但在这轮 `700` 条大样本结果里：

- `DPO v2 checkpoint-330 = 0.2614`
- `huatuo_5w checkpoint-75 = 0.2607`
- 差值只有 `+0.0007`

因此现在更严谨的说法不是“`DPO v2` 已明显超过 `SFT 5w`”，而是：

- 在 `700` 条 `consensus` 大样本口径下，`DPO v2 checkpoint-330` 和 `huatuo_5w checkpoint-75` 已经进入同档区间
- `DPO v2` 的价值主要体现为“把分数结构改成更偏医疗行为对齐”，不是全面改写榜单

`DPO v2 checkpoint-330` 相比 `checkpoint-30`，这轮 `700` 条结果提升：

- overall：`+0.0217`
- `Emergency referrals` (`theme:emergency_referrals`)：`+0.1050`
- `Expertise-tailored communication` (`theme:communication`)：`+0.0350`
- `Context seeking` (`theme:context_seeking`)：`+0.0200`
- `Global health` (`theme:global_health`)：`+0.0250`
- `axis:accuracy`：`+0.0457`
- `axis:context_awareness`：`+0.0395`
- `axis:completeness`：`+0.0411`

同时它也付出了一些代价：

- `axis:communication_quality`：`-0.0167`
- `axis:instruction_following`：`-0.0450`
- `Health data tasks` (`theme:health_data_tasks`)：`-0.0200`

所以这轮 `700` 条结果也再次证明了：

- `checkpoint-330` 不是“偶然比 checkpoint-30 高一点”
- 而是一个更偏向医疗行为强化、风险分诊和上下文处理的版本

#### 3. `DPO v2 checkpoint-330` 为什么能追到这个位置

和 `huatuo_5w checkpoint-75` 对比，`DPO v2 checkpoint-330` 的明显优势在：

- `Expertise-tailored communication`：`0.0900 vs 0.0350`，高 `+0.0550`
- `Context seeking`：`0.1250 vs 0.0900`，高 `+0.0350`
- `Emergency referrals`：`0.3400 vs 0.2750`，高 `+0.0650`
- `Health data tasks`：`0.4050 vs 0.3750`，高 `+0.0300`
- `axis:accuracy`：`0.2143 vs 0.1829`，高 `+0.0314`
- `axis:context_awareness`：`0.3362 vs 0.3220`，高 `+0.0141`

这说明 `DPO v2` 的收益不是随机噪声，而是集中在更像医疗问诊行为和风险分诊能力的切片上。

#### 4. `DPO v2 checkpoint-330` 还没有补齐的短板

和 `huatuo_5w checkpoint-75` 对比，`DPO v2 checkpoint-330` 当前最稳定的短板仍然是：

- `Global health`：`0.3200 vs 0.4650`，低 `-0.1450`
- `axis:communication_quality`：`0.2033 vs 0.2867`，低 `-0.0833`
- `Responding under uncertainty`：`0.3500 vs 0.3700`，低 `-0.0200`
- `Response depth`：`0.2000 vs 0.2150`，低 `-0.0150`

如果和当前总分最高的 `HQ-50k best` 再对比一次，差距会更清楚：

- `overall`：`0.2614 -> 0.2767`，落后 `-0.0152`
- `Global health`：`0.3200 -> 0.4850`，落后 `-0.1650`
- `Responding under uncertainty`：`0.3500 -> 0.4267`，落后 `-0.0767`
- `axis:communication_quality`：`0.2033 -> 0.2867`，落后 `-0.0833`

但 `DPO v2 checkpoint-330` 也保留了自己独有的强项：

- `Emergency referrals`：`0.3400 vs 0.2700`，高 `+0.0700`
- `Expertise-tailored communication`：`0.0900 vs 0.0450`，高 `+0.0450`
- `Context seeking`：`0.1250 vs 0.0950`，高 `+0.0300`
- `axis:context_awareness`：`0.3362 vs 0.3164`，高 `+0.0198`

所以它更像一个“安全分诊和上下文处理更激进”的版本，而不是当前最均衡的版本。

#### 5. 当前最合理的工程结论

截至现在，这轮 `700` 条大样本结果支持下面这几个结论：

- `SFT 1K` 仍高于 `base`，但优势有限，更适合作为 smoke test 路线
- `huatuo_5w checkpoint-75` 仍是稳定可信的主线 SFT baseline
- `HQ-50k best` 已经把“高质量数据筛选 + 干净 SFT”这条线抬成当前正式总分第一
- `DPO v2 checkpoint-330` 没有明显打穿最强 SFT，但它在 `Emergency referrals / Context seeking / accuracy / context_awareness` 上给出了很有意义的结构化收益
- 这意味着后续最值得做的不是盲目继续拉长 DPO，而是做针对性 `GRPO / reward design`

如果目标是“尽快把最终分数做高”，更合适的起点是：

- 以 `HQ-50k best` 为起点，补它相对 `DPO v2` 还偏弱的
  - `Emergency referrals`
  - `Expertise-tailored communication`
  - `Context seeking`

如果目标是“把整条 SFT -> DPO -> GRPO 链路讲完整”，更合适的起点是：

- 以 `DPO v2 checkpoint-330` 为起点，针对性补它最稳定的短板
  - `communication_quality`
  - `Global health`
  - `Responding under uncertainty`
  - `instruction_following`

这条叙事其实是很强的，因为它已经形成了清晰的工程分工：

- `SFT` 负责把医学知识和任务能力拉起来
- `DPO v2` 负责把模型往更强的临床行为和分诊方向推
- `GRPO` 负责把 `DPO` 引入的结构性短板再补回去，争取在统一 `HealthBench consensus` 口径下稳定超过当前最强 SFT baseline

### 当前批次状态

截至 `2026-04-13`：

- `6/6` 个模型已经完成正式 judge
- 队列日志显示：
  - `judge finished: hq50k_best`
  - `all queued evaluations finished`

对应 run：

- `base`
  - `20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_base_gpt-52_consensus_theme100x7_seed42`
- `sft_1k`
  - `20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_huatuo_1k_lora_gpt-52_consensus_theme100x7_seed42`
- `sft_5w_ckpt75`
  - `20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_consensus_theme100x7_seed42`
- `dpo_v2_ckpt30`
  - `20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_dpo_v2_ckpt30_gpt-52_consensus_theme100x7_seed42`
- `dpo_v2_ckpt330`
  - `20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_consensus_theme100x7_seed42`
- `hq50k_best`
  - `20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_hq50k_best_gpt-52_consensus_theme100x7_seed42`

## 2026-04-11 夜跑验收

昨晚额外启动了一轮更大样本的 `HealthBench consensus` 队列评测，目标是一次性评完下面 `6` 个模型：

- `base`
- `SFT 1K`
- `SFT 5W checkpoint-75`
- `DPO v2 checkpoint-30`
- `DPO v2 checkpoint-330`
- `HQ-50k best`

对应口径：

- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `100` 条，共 `700` 条
- 随机种子：`42`
- 生成 batch size：`8`
- 目标：主要用于检查大样本下的排序稳定性

这轮夜跑的验收结论要先说清楚：

- 这批任务没有完整跑完
- 实际只完成了 `base` 和 `SFT 1K` 的生成阶段
- 没有形成 `6` 模型同口径、同批样本、同轮 judge 的最终对比结果
- 因此这批夜跑结果目前不能直接并入“正式分数矩阵”

已确认完成的部分如下：

| 模型 | 状态 | 样本数 | generation_seconds | 直观耗时 |
| --- | --- | ---: | ---: | --- |
| `Qwen3-8B base` | 已完成生成，未完成 judge | `700` | `1097.105` | 约 `18.3` 分钟 |
| `Qwen3-8B + huatuo_1k LoRA` | 已完成生成，未完成 judge | `700` | `2133.430` | 约 `35.6` 分钟 |

对应 run：

- `20260411_overnight_theme100x7_live_healthbench_qwen3_8b_base_gpt-52_consensus_theme100x7_seed42`
- `20260411_overnight_theme100x7_live_healthbench_qwen3_8b_huatuo_1k_lora_gpt-52_consensus_theme100x7_seed42`

日志验收结论：

- 队列日志显示 `base` 于 `2026-04-11 23:44:43` 完成 `700` 条生成
- `SFT 1K` 于 `2026-04-12 00:02:02` 完成 `700` 条生成
- 之后没有继续看到 `sft_5w_ckpt75 / dpo_v2_ckpt30 / dpo_v2_ckpt330 / hq50k_best` 的排队执行记录
- 也没有看到这批任务的 judge 阶段完成记录

工程上这批夜跑并不是完全白跑，原因是：

- `base` 和 `SFT 1K` 的 `responses.jsonl` 已经落盘
- 共享 response cache 也已经写入
- 后续如果继续用完全相同的采样配置补 judge，或者补跑同配置复现实验，这两部分生成可以直接复用

所以当前最准确的说法是：

- 这次夜跑更像一次“大样本生成吞吐验收 + 缓存验证”
- 还不是一份可以正式引用的 `6` 模型大样本 benchmark 报告
- 当前仍应以本文后面的 `theme 15 x 7 = 105` 正式结果矩阵作为对外主结论

## 正式分层结果矩阵

对比配置：

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- 随机种子：`42`
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`
- judge API 形式：OpenAI 兼容 chat completions

参与对比的 run：

- base：`20260409_healthbench_base_gpt52_full_theme15x7`
- sft 1k：`20260410_healthbench_huatuo1k_gpt52_full_theme15x7`
- sft 5w 最优 checkpoint：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- sft 5w 较晚 checkpoint：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`
- dpo v1：`20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7`

## 结论先看

| 模型 | overall clipped mean | 备注 |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.2206` | 当前正式 stratified baseline |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2508` | 当前正式小规模 SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-75)` | `0.2889` | 当前最优正式 SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-925)` | `0.2587` | 同一 run 中更晚的 checkpoint |
| `Qwen3-8B + DPO medical_pairwise (ckpt-100)` | `0.2111` | 当前 DPO v1 正式结果 |

结论摘要：

- `huatuo_5w checkpoint-75` 仍然是当前最强正式 baseline，说明主线 SFT 方案依然成立
- `huatuo_1k` 在正式 `105` 条分层评测上高于 base，说明它虽然不适合作为最终主模型，但已经是有效的“小数据跑通版”
- `DPO v1` 低于 base，也明显低于最佳 SFT，说明“对齐阶段已跑通”不等于“当前偏好数据和目标设计已经正确”
- `checkpoint-75` 明确优于 `checkpoint-925`，这让“早停 + 最佳 checkpoint 选择”成为 benchmark 支撑下的标准流程

## 把你的分数放到官方坐标系里看

这一节的目标只有一个：

- 帮你建立对 `0.01 / 0.04 / 0.10` 这种分差的直观认识

先说一个非常重要的前提：

- 论文里的分数来自官方 `HealthBench` 全量评测
- 你当前仓库里的正式分数来自 `HealthBench consensus + stratified_theme 7 x 15 = 105` 抽样

所以：

- 绝对分数不能直接横比
- 但“分差量级”是可以拿来做直觉参照的

### 官方 overall 分数锚点

下面这组 exact overall 来自论文 `Table 7`：

| 官方模型 | overall mean |
| --- | ---: |
| `GPT-3.5 Turbo` | `0.1554` |
| `GPT-4o (Aug 2024)` | `0.3233` |
| `o1` | `0.4200` |
| `GPT-4.1` | `0.4778` |
| `o3` | `0.5990` |

这组数据告诉我们：

- 从 `GPT-3.5 Turbo` 到 `o3`，官方顶级模型总跨度约 `+0.4436`
- 在强模型区间里，`+0.05 ~ +0.12` 已经是非常实在的提升

### 官方模型间分差

| 官方对比 | 分差 |
| --- | ---: |
| `GPT-4o - GPT-3.5 Turbo` | `+0.1679` |
| `o1 - GPT-4o` | `+0.0967` |
| `GPT-4.1 - o1` | `+0.0578` |
| `o3 - GPT-4.1` | `+0.1212` |
| `o3 - GPT-4o` | `+0.2757` |

这就是最值得记住的量级感：

- `+0.001 ~ +0.005`：通常接近同档
- `+0.01 ~ +0.02`：有信号，但还需要更多 seed 或更大样本确认
- `+0.03 ~ +0.07`：已经不是“小修小补”
- `> +0.10`：在 HealthBench 这种 `0~1` 分数上，属于很大的代际差

### 你的当前结果放回这个量级里

| 你的对比 | 分差 | 直观解读 |
| --- | ---: | --- |
| `SFT 1k - base` | `+0.0302` | 已经是有效提升 |
| `SFT 5w ckpt-75 - base` | `+0.0683` | 明显提升，不是噪声 |
| `DPO v2 ckpt-330 - base` | `+0.0413` | 也属于有意义提升 |
| `HQ-50k best - base` | `+0.0699` | 与 `5w ckpt-75` 同量级 |
| `HQ-50k best - 5w ckpt-75` | `+0.0016` | 基本视为同档 |
| `SFT 5w ckpt-75 - DPO v2 ckpt-330` | `+0.0270` | 单次有信号，但需要复测确认 |

对应结论就很清楚了：

- `base -> 5w SFT` 这条线绝对不是“白做了”
- `base -> DPO v2` 也不是“没效果”
- 真正接近“同档”的，是 `HQ-50k best` 和 `5w ckpt-75` 这种 `+0.0016`

### 官方 theme 切片长什么样

论文正文对主题层面的总结是：

- `emergency referrals` 和 `expertise-tailored communication` 普遍最高
- `context seeking`、`health data tasks`、`global health` 普遍更难

下面这张表不是论文中的现成表格，而是我根据 `Figure 5` 做的近似视觉读取，用来建立直觉。
这些值请理解为：

- `~` 表示约值
- 用于看“高低分布”和“差值量级”
- 不要当成论文 appendix 里的严格数表

| 模型 | ER | ETC | RUU | RD | HDT | GH | CS | overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `o3` | `~0.72` | `~0.70` | `~0.60` | `~0.56` | `~0.45` | `~0.57` | `~0.53` | `0.5990` |
| `GPT-4.1` | `~0.66` | `~0.58` | `~0.49` | `~0.48` | `~0.39` | `~0.39` | `~0.38` | `0.4778` |
| `o1` | `~0.52` | `~0.54` | `~0.46` | `~0.42` | `~0.34` | `~0.33` | `~0.33` | `0.4200` |
| `GPT-4o` | `~0.46` | `~0.40` | `~0.35` | `~0.36` | `~0.32` | `~0.21` | `~0.20` | `0.3233` |
| `GPT-3.5` | `~0.27` | `~0.21` | `~0.19` | `~0.21` | `~0.20` | `~0.05` | `~0.06` | `0.1554` |

缩写说明：

- `ER` = `emergency referrals`
- `ETC` = `expertise-tailored communication`
- `RUU` = `responding under uncertainty`
- `RD` = `response depth`
- `HDT` = `health data tasks`
- `GH` = `global health`
- `CS` = `context seeking`

这张图最有价值的地方在于：

- 即使是 `o3`，在 `health data tasks / context seeking` 上也远没有到“接近满分”
- 主题间天然就存在很大难度差
- 所以你的模型如果在某些主题只提升 `0.03 ~ 0.08`，并不算小

### 官方 axis 切片长什么样

论文正文对 axis 层面的总结是：

- `completeness` 和 `context_awareness` 普遍更低
- `accuracy`、`communication_quality`、`instruction_following` 普遍更高
- 在论文那组模型里，`completeness` 与 overall 排名最相关
- 原文还特别指出：差不多 `40%` 的 rubric items 与 `completeness` 相关

下面同样是基于 `Figure 6` 的近似视觉读取：

| 模型 | CQ | IF | ACC | CA | COMP | overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `o3` | `~0.66` | `~0.61` | `~0.69` | `~0.41` | `~0.62` | `0.5990` |
| `GPT-4.1` | `~0.76` | `~0.63` | `~0.60` | `~0.38` | `~0.39` | `0.4778` |
| `o1` | `~0.71` | `~0.56` | `~0.56` | `~0.32` | `~0.33` | `0.4200` |
| `GPT-4o` | `~0.74` | `~0.51` | `~0.47` | `~0.27` | `~0.18` | `0.3233` |
| `GPT-3.5` | `~0.61` | `~0.42` | `~0.31` | `~0.16` | `<0.02` | `0.1554` |

缩写说明：

- `CQ` = `communication_quality`
- `IF` = `instruction_following`
- `ACC` = `accuracy`
- `CA` = `context_awareness`
- `COMP` = `completeness`

对你项目最重要的启发是：

- `context_awareness / completeness` 本来就是最难拉的
- 这正好解释了为什么你现在围绕 `context_awareness / emergency_referrals / hedging / communication_quality` 去做 DPO/GRPO 是有意义的
- 这些不是“你自己随便挑的点”，而是官方 benchmark 本身就最敏感、也最难的地方

### 官方稳定性和你当前稳定性

论文 `Table 7` 里，官方对 `16` 次重复运行的 overall 波动给出的结果是：

| 模型 | mean | min | max | std |
| --- | ---: | ---: | ---: | ---: |
| `o3` | `0.5990` | `0.5951` | `0.6014` | `0.0016` |
| `GPT-4.1` | `0.4778` | `0.4742` | `0.4815` | `0.0022` |
| `o1` | `0.4200` | `0.4153` | `0.4230` | `0.0022` |
| `GPT-4o (Aug 2024)` | `0.3233` | `0.3188` | `0.3257` | `0.0020` |
| `GPT-3.5 Turbo` | `0.1554` | `0.1505` | `0.1611` | `0.0029` |

而你当前两轮复测里：

| 模型 | mean | std | 备注 |
| --- | ---: | ---: | --- |
| `huatuo_5w checkpoint-75` | `0.2754` | `0.0135` | 当前波动明显大于论文 full-eval 口径 |
| `DPO v2 checkpoint-330` | `0.2603` | `0.0016` | 已经接近论文里强模型的重复波动量级 |

这不是说你的评测不行，而是说明：

- 论文是更大规模、更完整的评测
- 你现在的 `105` 条抽样更适合做工程迭代
- 所以 `+0.003` 这种差距在你这里确实不能讲太满
- 但 `+0.03 ~ +0.07` 仍然完全不是“没做出效果”

### 官方 pairwise 胜率锚点

论文还给了几个很适合建立直觉的 pairwise 胜率：

| 官方对比 | 胜率 | 长度控制后 |
| --- | ---: | ---: |
| `o3 vs GPT-4.1` | `72.9%` | `63.7%` |
| `GPT-4.1 vs GPT-4o (Aug 2024)` | `77.5%` | `75.4%` |
| `GPT-4.1 vs o1` | `61.0%` | `65.2%` |

这说明：

- HealthBench 不只是看“输出更长”
- 更强模型即使控制长度后，仍然保留明显优势

### 这一页最后该怎么记

如果你后面想快速判断一次实验值不值得兴奋，可以先用下面这个口径：

- `< 0.01`：默认同档，不要夸大
- `0.01 ~ 0.03`：有信号，继续复测
- `0.03 ~ 0.07`：已经是有效提升
- `> 0.10`：在 HealthBench 上属于很大差距

所以你当前项目最合理的自我判断是：

- `1k SFT`：有效
- `5w SFT`：明确有效
- `DPO v2`：有效，但还没稳定超过最强 SFT
- `HQ-50k best`：和 `5w ckpt-75` 基本同档，不能夸成“显著领先”

这一节的官方参照来源：

- OpenAI HealthBench 页面：<https://openai.com/zh-Hans-CN/index/healthbench/>
- HealthBench 论文 PDF：<https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf>
- exact overall 与重复波动：论文 `Table 7`
- pairwise 胜率：论文 `Table 4`
- theme / axis 约值：基于论文 `Figure 5 / Figure 6` 的视觉近似读取

## Axis 对比

| axis | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 | DPO ckpt-100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `axis:accuracy` | `0.1711` | `0.2039` | `0.2368` | `0.1974` | `0.1447` |
| `axis:communication_quality` | `0.2000` | `0.2444` | `0.2889` | `0.2444` | `0.1778` |
| `axis:completeness` | `0.0000` | `0.0833` | `0.2500` | `0.0000` | `0.0833` |
| `axis:context_awareness` | `0.2407` | `0.2778` | `0.3704` | `0.3333` | `0.2778` |
| `axis:instruction_following` | `0.5000` | `0.4667` | `0.3667` | `0.5333` | `0.4667` |

解读：

- `checkpoint-75` 在 `accuracy`、`completeness`、`context_awareness` 上最强，整体最均衡
- `huatuo_1k` 已经在 `accuracy`、`communication_quality`、`context_awareness` 上超过 base，说明小规模 SFT 不是完全无效，而是上限较低
- `DPO v1` 的 `context_awareness` 和 `instruction_following` 没有崩，但 `accuracy`、`communication_quality` 明显回退，这更像“偏好优化目标与正式 benchmark 目标不一致”
- `checkpoint-925` 的 `instruction_following` 虽高，但没能转化为更高总分，进一步说明不能只盯住单一切片

## Theme 对比

| theme | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 | DPO ckpt-100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Expertise-tailored communication` (`theme:communication`) | `0.0333` | `0.1333` | `0.0667` | `0.0667` | `0.0333` |
| `Response depth` (`theme:complex_responses`) | `0.2333` | `0.2333` | `0.2000` | `0.2333` | `0.2333` |
| `Context seeking` (`theme:context_seeking`) | `0.0333` | `0.0333` | `0.0333` | `0.1667` | `0.0333` |
| `Emergency referrals` (`theme:emergency_referrals`) | `0.2667` | `0.3000` | `0.4333` | `0.2000` | `0.2000` |
| `Global health` (`theme:global_health`) | `0.2667` | `0.2667` | `0.5333` | `0.3333` | `0.2667` |
| `Health data tasks` (`theme:health_data_tasks`) | `0.4667` | `0.4333` | `0.3333` | `0.4333` | `0.4000` |
| `Responding under uncertainty` (`theme:hedging`) | `0.2444` | `0.3556` | `0.4222` | `0.3778` | `0.3111` |

解读：

- `checkpoint-75` 在 `Emergency referrals`、`Global health`、`Responding under uncertainty` 上的优势最明显
- `huatuo_1k` 在 `Expertise-tailored communication`、`Emergency referrals`、`Responding under uncertainty` 上已经优于 base，这解释了它为什么正式总分高于 base
- `DPO v1` 只在 `Responding under uncertainty` 上略高于 base，其他关键主题基本没有带来正式收益，尤其 `Emergency referrals` 仍然偏弱
- `Health data tasks` 是 base 本身不弱的主题，因此正式结论不能只靠这个切片来讲

## 面试里最值得讲的故事

更强的表述不是：

- “我把模型做了 SFT、DPO，然后分数自然会越来越高。”

而是：

- “我在 `HealthBench consensus` 上做了按主题分层抽样，每个主题 `15` 条，共 `105` 条，形成了 base -> 1k SFT -> 5w SFT -> DPO 的同口径闭环。结果 `huatuo_5w checkpoint-75` 最强，`checkpoint-925` 回退，说明 SFT 需要最佳 checkpoint 选择；同时 `DPO v1` 低于最佳 SFT 甚至略低于 base，说明偏好优化不能只靠旧 pairwise 数据硬上，而要让 reward 目标和医疗 benchmark 切片真正对齐。” 

这种说法会明显更接近真实工业实验。

## 当前对齐阶段结论

基于这组正式结果，当前更合理的工程判断是：

- 当前主模型仍应保持为 `huatuo_5w checkpoint-75`
- `huatuo_1k` 继续保留为轻量 SFT baseline，很适合做 smoke test 和流程验证
- `DPO v1` 说明训练链路、评测链路、外部 judge 都已经打通，但现有 `medical_pairwise` 数据还不足以支撑正式收益
- 下一步更值得做的不是“盲目继续 DPO”，而是：
  - 扩充更贴近 `HealthBench` 的医疗偏好数据
  - 把 `Expertise-tailored communication`、`Responding under uncertainty`、`Emergency referrals` 这类切片纳入 reward 设计
  - 继续保留同一套 `theme 15 x 7` 正式评测，作为所有对齐算法的统一验收口径

## DPO V2 更新（2026-04-11）

`DPO v2` 的两组正式结果已经补进统一矩阵，详细分析见：

- [DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md)
- [HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md)

这次不是随便挑 checkpoint，而是按预先定义的实验意义选了两个点：

- `checkpoint-30`
  - 训练内 best checkpoint
  - 也是 `valid_zh_500` 辅助验证最优点
- `checkpoint-330`
  - 更接近 `min eval_loss`
  - 代表更强的 pairwise 偏好优化结果

最新总体分数：

| 模型 | overall clipped mean |
| --- | ---: |
| `DPO v1 checkpoint-100` | `0.2111` |
| `DPO v2 checkpoint-30` | `0.2492` |
| `DPO v2 checkpoint-330` | `0.2619` |
| `SFT 5w checkpoint-75` | `0.2889` |

更新后的工程判断：

- `DPO v2` 已经明显优于 `DPO v1`，说明重构后的 pairwise 数据是有效的
- 外部 `HealthBench` 更认可 `checkpoint-330`，而不是训练内 best 的 `checkpoint-30`
- `checkpoint-330` 在 `context_awareness`、`Emergency referrals`、`Responding under uncertainty` 上更强
- 单次最好分仍是 `SFT 5w checkpoint-75`，但第二轮独立抽样复测显示这个领先并不稳定
- 所以下一阶段最合理的主线是：
  - 主模型继续保持 `SFT 5w checkpoint-75`
  - `DPO v2 checkpoint-330` 作为当前最有潜力的偏好对齐版本继续推进
  - 后续优先补 `communication_quality` 和 `instruction_following`

## HQ-50k 更新（2026-04-11）

`HQ-50k` 的正式结果也已经补进统一矩阵，专项分析见：

- [HQ50K_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HQ50K_HEALTHBENCH_COMPARE.zh-CN.md)

这次想回答的是：

- 数据筛选出来的 `HQ-50k`
- 相比原始 `huatuo_5w`
- 是否真的能在外部 benchmark 上更好

最新结果：

| 模型 | overall clipped mean |
| --- | ---: |
| `huatuo_5w checkpoint-75` | `0.2889` |
| `HQ-50k best` | `0.2905` |
| `HQ-50k late-1564` | `0.2714` |

更新后的工程判断：

- `HQ-50k best` 目前是当前最强的 `SFT` 正式结果
- 它对 `huatuo_5w checkpoint-75` 的提升很小，只有 `+0.0016`
- 所以更合理的说法是“当前证据支持 `HQ-50k` 略优于原始 5w”，而不是“已经显著领先”
- `HQ-50k late-1564` 明显低于 `HQ-50k best`，说明即使数据质量更高，`early stopping` 仍然是必须的
- `HQ-50k` 的主要收益在：
  - `accuracy`
  - `communication_quality`
  - `instruction_following`
  - `Health data tasks`
- `huatuo_5w` 仍然更强的地方在：
  - `Emergency referrals`
  - `context_awareness`
  - 一部分 `Expertise-tailored communication`

这说明当前最合理的 SFT 主线已经可以更新为：

- 第一候选：`HQ-50k best`
- 强基线保留：`huatuo_5w checkpoint-75`
- 后续再用更定向的数据或对齐方法补 `emergency_referrals / context_awareness`

## 早期 smoke 结果矩阵（仅供参考）

对比配置：

- benchmark：`HealthBench consensus`
- sample count：`10`
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`
- judge API 形式：OpenAI 兼容 chat completions

参与对比的 run：

- base：`20260409_healthbench_base_gpt52_full_10`
- sft 1k：`20260409_healthbench_huatuo1k_gpt52_full_10`
- sft 5w 最优 checkpoint：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10`
- sft 5w 较晚 checkpoint：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10`

## 结论先看

| 模型 | overall clipped mean | 备注 |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.4167` | 当前 smoke baseline |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2833` | 当前 smoke SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-75)` | `0.4167` | 当前 5w run 中验证最优 checkpoint |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-925)` | `0.4000` | 同一 run 中更晚的 checkpoint |

结论摘要：

- `huatuo_5w` 明显修复了 `1k` 的退化问题
- 当前观测到的最优 `5w` checkpoint 在总分上追平了 base
- 同一条 `5w` 训练线上，较晚的 `checkpoint-925` 反而略差于 `checkpoint-75`
- 这说明下一步不应该简单地“训更久”，而应该把“最优 checkpoint 选择/早停”纳入标准流程

## Axis 对比

| axis | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 |
| --- | ---: | ---: | ---: | ---: |
| `axis:accuracy` | `0.5000` | `0.1429` | `0.3571` | `0.2857` |
| `axis:communication_quality` | `0.6667` | `0.6667` | `0.6667` | `0.6667` |
| `axis:context_awareness` | `0.2500` | `0.1250` | `0.2500` | `0.2500` |
| `axis:instruction_following` | `0.5000` | `1.0000` | `1.0000` | `1.0000` |

解读：

- `5w` 已经把 `context_awareness` 拉回到 base 水平
- `5w` 保留了 `1k` 在 `instruction_following` 上的提升
- `accuracy` 从 `1k` 到 `5w` 明显恢复，但在这组 smoke 样本上仍未超过 base
- `checkpoint-75` 的 `accuracy` 高于 `checkpoint-925`，和训练期 `eval_loss` 的趋势一致

## Theme 对比

| theme | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 |
| --- | ---: | ---: | ---: | ---: |
| `Response depth` (`theme:complex_responses`) | `1.0000` | `0.5000` | `1.0000` | `1.0000` |
| `Context seeking` (`theme:context_seeking`) | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `Emergency referrals` (`theme:emergency_referrals`) | `0.5000` | `0.2500` | `0.0000` | `0.0000` |
| `Global health` (`theme:global_health`) | `0.5000` | `0.2500` | `0.2500` | `0.5000` |
| `Health data tasks` (`theme:health_data_tasks`) | `0.5000` | `1.0000` | `1.0000` | `1.0000` |
| `Responding under uncertainty` (`theme:hedging`) | `0.2222` | `0.1111` | `0.5556` | `0.3333` |

解读：

- `5w` 在 `Responding under uncertainty` 上出现了比较有价值的提升，尤其是 `checkpoint-75`
- `5w` 保住了 `Health data tasks` 的优势
- 但 `Emergency referrals` 这个安全敏感切片目前明显偏弱，甚至低于 base
- 这意味着下一步不能只靠“继续堆通用医疗 SFT 数据”，而要补安全和分诊类数据/偏好信号

## 5w 训练信号本身也说明了问题

从 `20260409_121822_qwen3-8b_huatuo-5w_lora_eval` 的训练状态看：

- 最优验证 checkpoint：`checkpoint-75`
- 最优验证 `eval_loss`：`2.5285`
- 文档中额外评测的较晚 checkpoint：`checkpoint-925`
- 对应验证 `eval_loss`：`2.8470`

这很重要，因为：

- 外部 benchmark 和训练期验证信号是一致的
- 更晚的 checkpoint 没有更好，反而更差
- 这正是“要做早停、要做 best checkpoint 选择、不能只训更久”的直接证据

## 为什么这反而是亮点

一个很弱的项目故事是：

- “我微调了模型，分数提高了。”

更强的项目故事是：

- “我在同一个开放式医疗 benchmark 上对比了 base、1k SFT 和 5w SFT。结果 5w 明显修复了 1k 的退化，但最佳 checkpoint 出现在较早阶段，后面继续训练反而回退。所以我把‘checkpoint 选择 + 目标切片优化’作为下一步，而不是机械地继续堆训练时长。”

这种表述明显更接近真实工业研发。

## 下一步建议

基于当前结果矩阵，更合理的下一步是：

1. `1k` 继续保留为 pipeline 验证 checkpoint，但不再作为主 SFT 方案。
2. 当前主 SFT baseline 应切换为 `huatuo_5w checkpoint-75`，而不是 `checkpoint-925`。
3. 把“best checkpoint 选择/早停”纳入标准训练流程，这已经不是可选优化，而是必要步骤。
4. 正式比较时，把评测样本数从 `10` 提高到 `50` 或 `100`，降低 smoke test 偶然性。
5. 在进入 `DPO / GRPO` 前，先补强 `Emergency referrals` 这类安全敏感数据和偏好信号。
6. 如果目标是稳定超过 base，下一步更可能依赖：
   - 更合理的数据分布
   - 更好的验证集策略
   - 面向安全/沟通切片的对齐方法
  而不是简单延长通用 SFT 训练时间。

## 结果来源

- [formal base summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_theme15x7/summary.json)
- [formal 1k summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260410_healthbench_huatuo1k_gpt52_full_theme15x7/summary.json)
- [formal 5w ckpt-75 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7/summary.json)
- [formal 5w ckpt-925 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7/summary.json)
- [formal dpo ckpt-100 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7/summary.json)
- [formal dpo v2 ckpt-30 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_dpo_v2_ckpt30_gpt52_consensus_theme15x7/summary.json)
- [formal dpo v2 ckpt-330 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_dpo_v2_ckpt330_gpt52_consensus_theme15x7/summary.json)
- [formal base vs 5w compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_theme15x7/README.zh-CN.md)
- [formal base vs 1k vs 5w vs dpo compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260410_healthbench_compare_base_vs_1k_vs_5w_vs_dpo_gpt52_theme15x7/README.zh-CN.md)
- [base summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [1k LoRA summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json)
- [5w ckpt-75 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10/summary.json)
- [5w ckpt-925 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10/summary.json)
- [base vs 1k compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md)
- [base vs 5w compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_10/README.md)
