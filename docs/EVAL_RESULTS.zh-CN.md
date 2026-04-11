# 评测结果解读

英文原版见 [EVAL_RESULTS.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.md)。

这份文档记录了当前最值得在面试里讨论的一组 benchmark 快照。

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
| `theme:communication` | `0.0333` | `0.1333` | `0.0667` | `0.0667` | `0.0333` |
| `theme:complex_responses` | `0.2333` | `0.2333` | `0.2000` | `0.2333` | `0.2333` |
| `theme:context_seeking` | `0.0333` | `0.0333` | `0.0333` | `0.1667` | `0.0333` |
| `theme:emergency_referrals` | `0.2667` | `0.3000` | `0.4333` | `0.2000` | `0.2000` |
| `theme:global_health` | `0.2667` | `0.2667` | `0.5333` | `0.3333` | `0.2667` |
| `theme:health_data_tasks` | `0.4667` | `0.4333` | `0.3333` | `0.4333` | `0.4000` |
| `theme:hedging` | `0.2444` | `0.3556` | `0.4222` | `0.3778` | `0.3111` |

解读：

- `checkpoint-75` 在 `emergency_referrals`、`global_health`、`hedging` 上的优势最明显
- `huatuo_1k` 在 `communication`、`emergency_referrals`、`hedging` 上已经优于 base，这解释了它为什么正式总分高于 base
- `DPO v1` 只在 `hedging` 上略高于 base，其他关键主题基本没有带来正式收益，尤其 `emergency_referrals` 仍然偏弱
- `health_data_tasks` 是 base 本身不弱的主题，因此正式结论不能只靠这个切片来讲

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
  - 把 `communication`、`hedging`、`emergency_referrals` 这类切片纳入 reward 设计
  - 继续保留同一套 `theme 15 x 7` 正式评测，作为所有对齐算法的统一验收口径

## DPO V2 更新（2026-04-11）

`DPO v2` 的两组正式结果已经补进统一矩阵，详细分析见：

- [DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md)

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
- `checkpoint-330` 在 `context_awareness`、`emergency_referrals`、`hedging` 上更强
- 但它仍未超过当前最强的 `SFT 5w checkpoint-75`
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
  - `health_data_tasks`
- `huatuo_5w` 仍然更强的地方在：
  - `emergency_referrals`
  - `context_awareness`
  - 一部分 `communication`

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
| `theme:complex_responses` | `1.0000` | `0.5000` | `1.0000` | `1.0000` |
| `theme:context_seeking` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `theme:emergency_referrals` | `0.5000` | `0.2500` | `0.0000` | `0.0000` |
| `theme:global_health` | `0.5000` | `0.2500` | `0.2500` | `0.5000` |
| `theme:health_data_tasks` | `0.5000` | `1.0000` | `1.0000` | `1.0000` |
| `theme:hedging` | `0.2222` | `0.1111` | `0.5556` | `0.3333` |

解读：

- `5w` 在 `hedging` 上出现了比较有价值的提升，尤其是 `checkpoint-75`
- `5w` 保住了 `health_data_tasks` 的优势
- 但 `emergency_referrals` 这个安全敏感切片目前明显偏弱，甚至低于 base
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
5. 在进入 `DPO / GRPO` 前，先补强 `emergency_referrals` 这类安全敏感数据和偏好信号。
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
