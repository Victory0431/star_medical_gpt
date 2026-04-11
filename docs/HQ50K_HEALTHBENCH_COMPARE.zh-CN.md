# HQ-50k SFT HealthBench 对比分析

这份文档记录 `HQ-50k` 数据筛选路线的第一轮正式外部 benchmark 结果，并与当前主线 `huatuo_5w` baseline 做同口径比较。

相关训练 run：

- [20260410_234458_qwen3-8b_hq-50k_lora_eval](/home/qjh/llm_learning/my_medical_gpt/outputs/sft/20260410_234458_qwen3-8b_hq-50k_lora_eval)

对应轻量评测快照：

- [experiment_records/eval/20260411_healthbench_hq50k_best_gpt52_consensus_theme15x7/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_hq50k_best_gpt52_consensus_theme15x7/summary.json)
- [experiment_records/eval/20260411_healthbench_hq50k_late1564_gpt52_consensus_theme15x7/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_hq50k_late1564_gpt52_consensus_theme15x7/summary.json)

相关数据筛选说明：

- [SFT_DATA_CURATION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/SFT_DATA_CURATION.zh-CN.md)

## 1. 这次实验想回答什么问题

这次实验不是单纯多训一版 SFT，而是回答一个更关键的问题：

- 用规则粗筛 + 分布分析 + 轻量质量打分做出来的 `HQ-50k`
- 相比原始 `huatuo_5w`
- 在外部医疗 benchmark 上是否真的更好

同时，这次也顺带验证另一个问题：

- 即使数据质量更高，SFT 是否仍然需要明显的 `early stopping`

## 2. 训练背景

### 数据

本次 `HQ-50k` 训练数据来自：

- [hq_50k_source_stratified.jsonl](/home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets/hq_50k_source_stratified.jsonl)

规模：

- `50,000` 条训练样本
- 验证集仍然使用 `valid_zh_500`

### 初始化方式

这次是从干净基座起跑，不是接着旧的 `SFT / DPO` 权重继续训：

- base model：`/home/qjh/llm_learning/base_model/qwen3_8B`
- `resume_from_checkpoint = null`

### 训练信号

最关键的训练结论：

- `best checkpoint = checkpoint-75`
- `best eval_loss = 2.4446`
- `final_model` 实际已经回滚到 `checkpoint-75`

并且这个 `best eval_loss` 明显好于原始 `huatuo_5w` 的：

- `huatuo_5w best eval_loss = 2.5285`

也就是说，从训练内验证信号看，`HQ-50k` 是更强的。

## 3. 评测设置

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- judge 模型：`gpt-5.2`
- 生成基座：`Qwen3-8B`

参与比较的模型：

- `Qwen3-8B base`
- `huatuo_1k checkpoint`
- `huatuo_5w checkpoint-75`
- `huatuo_5w checkpoint-925`
- `HQ-50k best`
  - 对应 `final_model`
  - 实际等价于 `checkpoint-75`
- `HQ-50k late`
  - 对应 `checkpoint-1564`

## 4. 先看核心结果

| 模型 | overall clipped mean | 备注 |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.2206` | 基座正式基线 |
| `huatuo_1k` | `0.2508` | 小规模 SFT baseline |
| `huatuo_5w checkpoint-75` | `0.2889` | 原始 5w 主 baseline |
| `huatuo_5w checkpoint-925` | `0.2587` | 原始 5w 晚期 checkpoint |
| `HQ-50k best` | `0.2905` | 当前 `HQ-50k` 最佳正式结果 |
| `HQ-50k late-1564` | `0.2714` | `HQ-50k` 晚期 checkpoint |

结论先说：

- `HQ-50k best = 0.2905`
- `huatuo_5w checkpoint-75 = 0.2889`

也就是说，在当前这套正式口径上：

- `HQ-50k best` 暂时是你现在所有 SFT 结果里的最高分
- 但领先幅度只有 `+0.0016`

所以更准确的说法是：

- “当前证据支持 HQ-50k 至少不弱于原始 5w，并出现了小幅正式优势”
- 而不是夸张地说“已经显著碾压”

## 5. HQ-50k best 相比 huatuo_5w best，强在哪里

### Axis 对比

| axis | huatuo_5w ckpt-75 | HQ-50k best | 谁更强 |
| --- | ---: | ---: | --- |
| `accuracy` | `0.2368` | `0.2434` | `HQ-50k best` |
| `communication_quality` | `0.2889` | `0.3111` | `HQ-50k best` |
| `completeness` | `0.2500` | `0.0833` | `huatuo_5w ckpt-75` |
| `context_awareness` | `0.3704` | `0.2963` | `huatuo_5w ckpt-75` |
| `instruction_following` | `0.3667` | `0.5667` | `HQ-50k best` |

解读：

- `HQ-50k best` 的主要收益在：
  - `accuracy`
  - `communication_quality`
  - `instruction_following`
- 原始 `huatuo_5w` 仍然明显更强的地方在：
  - `completeness`
  - `context_awareness`

也就是说，`HQ-50k` 并不是“全面提升”，而是：

- 更像把模型往“回答更稳、指令更贴、语言更顺”的方向推了一步
- 但在需要追问上下文、或者需要更完整展开时，还有损失

### Theme 对比

| theme | huatuo_5w ckpt-75 | HQ-50k best | 谁更强 |
| --- | ---: | ---: | --- |
| `communication` | `0.0667` | `0.0000` | `huatuo_5w ckpt-75` |
| `complex_responses` | `0.2000` | `0.2667` | `HQ-50k best` |
| `context_seeking` | `0.0333` | `0.0667` | `HQ-50k best` |
| `emergency_referrals` | `0.4333` | `0.2667` | `huatuo_5w ckpt-75` |
| `global_health` | `0.5333` | `0.5333` | 持平 |
| `health_data_tasks` | `0.3333` | `0.4333` | `HQ-50k best` |
| `hedging` | `0.4222` | `0.4667` | `HQ-50k best` |

这张表说明得更直接：

- `HQ-50k best` 更擅长：
  - `complex_responses`
  - `health_data_tasks`
  - `hedging`
  - 一部分 `context_seeking`
- `huatuo_5w ckpt-75` 更擅长：
  - `emergency_referrals`
  - `communication`

这很符合筛选数据的直觉：

- `HQ-50k` 把“结构更清楚、废话更少、信息密度更高”的样本提纯了
- 所以在任务型和稳健型回答上更强
- 但安全分诊、危险信号判断这种能力，不会仅靠通用质量筛选自然变强

## 6. HQ-50k 晚期 checkpoint 说明了什么

`HQ-50k late-1564 = 0.2714`

相比 `HQ-50k best = 0.2905`，下降了：

- `-0.0190`

而且它还低于：

- `huatuo_5w checkpoint-75 = 0.2889`

这说明一个很重要的事实：

- 即便数据质量提升了
- `early stopping` 仍然是必须的

这点和你之前 `huatuo_5w` 的结论完全一致：

- `huatuo_5w checkpoint-75 = 0.2889`
- `huatuo_5w checkpoint-925 = 0.2587`

也就是说，`HQ-50k` 并没有推翻你之前的工程判断，反而强化了它：

- 好数据很重要
- 但“好数据 + 正确的 best checkpoint 选择”才是完整答案

## 7. 当前最合理的项目结论

### 结论 1：数据筛选路线是有效的

目前最稳妥的结论可以写成：

- `HQ-50k` 在同口径 `HealthBench consensus 7 x 15` 上取得了当前 SFT 最好分数 `0.2905`
- 这说明规则粗筛 + 轻量质量打分 + 来源分层抽样的筛选路线是有效的

### 结论 2：提升是“小幅但真实”的

需要克制一点的地方是：

- `HQ-50k best` 只比 `huatuo_5w checkpoint-75` 高 `0.0016`

这意味着：

- 你可以说“当前正式结果显示 HQ-50k 略优于原始 5w”
- 但最好不要说成“显著领先”

更好的表达是：

- “在统一 benchmark 闭环下，HQ-50k 已经显示出小幅稳定优势，说明筛选方向大概率是对的；下一步需要更大规模或更多切片评测来确认收益是否稳固。”

### 结论 3：下一步不该只是继续堆数据

从结果看，后续更值得做的是：

1. 保留 `HQ-50k best` 作为当前最强 SFT 候选。
2. 把 `huatuo_5w checkpoint-75` 继续保留为强基线。
3. 下一轮补更定向的数据或对齐信号，优先强化：
   - `emergency_referrals`
   - `communication`
   - `context_awareness`

因为这些恰好是 `HQ-50k` 还没补齐的地方。

## 8. 面试里怎么讲最自然

最强的讲法不是：

- “我筛了一版高质量数据，分数就高了。”

而是：

- “我先搭了规则粗筛、轻量质量打分、来源分层抽样三层数据漏斗，再做了 `HQ-50k` 训练。训练内验证显示它优于原始 5w，外部 `HealthBench` 正式评测上 `HQ-50k best` 也以 `0.2905` 略高于 `huatuo_5w checkpoint-75` 的 `0.2889`。但这个优势并不大，而且 `HQ-50k` 在 `emergency_referrals` 和 `context_awareness` 上仍不占优，所以我没有简单地下结论说‘筛选就一定万能’，而是把它视为数据工程有效、但仍需要配合更定向对齐的证据。” 

这种说法会比“我做了数据清洗”更像真实工业实验。
