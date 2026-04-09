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
- sft 5w 最优 checkpoint：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- sft 5w 较晚 checkpoint：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`

## 结论先看

| 模型 | overall clipped mean | 备注 |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.2206` | 当前正式 stratified baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-75)` | `0.2889` | 当前最优正式 SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-925)` | `0.2587` | 同一 run 中更晚的 checkpoint |

结论摘要：

- `huatuo_5w` 在 `105` 条按主题平衡采样的正式小规模评测上已经稳定超过 base
- `checkpoint-75` 明确优于 `checkpoint-925`，说明最佳 checkpoint 出现在更早阶段
- `checkpoint-925` 虽然仍高于 base，但已经在 `accuracy`、`completeness`、`emergency_referrals` 上出现回退
- 这使“早停 + 最佳 checkpoint 选择”从经验建议变成了有 benchmark 支撑的标准流程

## Axis 对比

| axis | base | 5w ckpt-75 | 5w ckpt-925 |
| --- | ---: | ---: | ---: |
| `axis:accuracy` | `0.1711` | `0.2368` | `0.1974` |
| `axis:communication_quality` | `0.2000` | `0.2889` | `0.2444` |
| `axis:completeness` | `0.0000` | `0.2500` | `0.0000` |
| `axis:context_awareness` | `0.2407` | `0.3704` | `0.3333` |
| `axis:instruction_following` | `0.5000` | `0.3667` | `0.5333` |

解读：

- `checkpoint-75` 在 `accuracy`、`completeness`、`context_awareness` 上最强，整体最均衡
- `checkpoint-925` 的 `instruction_following` 高于 `checkpoint-75`，但没能转化为更高总分
- 对医疗问答更关键的 `accuracy` 和 `context_awareness`，仍然是 `checkpoint-75` 更优

## Theme 对比

| theme | base | 5w ckpt-75 | 5w ckpt-925 |
| --- | ---: | ---: | ---: |
| `theme:communication` | `0.0333` | `0.0667` | `0.0667` |
| `theme:complex_responses` | `0.2333` | `0.2000` | `0.2333` |
| `theme:context_seeking` | `0.0333` | `0.0333` | `0.1667` |
| `theme:emergency_referrals` | `0.2667` | `0.4333` | `0.2000` |
| `theme:global_health` | `0.2667` | `0.5333` | `0.3333` |
| `theme:health_data_tasks` | `0.4667` | `0.3333` | `0.4333` |
| `theme:hedging` | `0.2444` | `0.4222` | `0.3778` |

解读：

- `checkpoint-75` 在 `emergency_referrals`、`global_health`、`hedging` 上的优势最明显
- `checkpoint-925` 在 `context_seeking` 上更高，但代价是 `emergency_referrals` 明显回退
- `health_data_tasks` 是 base 本身不弱的主题，因此正式结论不能只靠这个切片来讲

## 面试里最值得讲的故事

更强的表述不是：

- “我把模型训了更久，分数更高了。”

而是：

- “我在 `HealthBench consensus` 上做了按主题分层抽样，每个主题 `15` 条，共 `105` 条。结果 `huatuo_5w` 的最优 checkpoint-75 能稳定超过 base，但更晚的 checkpoint-925 反而回退。这说明 SFT 不应该只看训练时长，而要把最佳 checkpoint 选择和切片诊断纳入标准流程。”

这种说法会明显更接近真实工业实验。

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
- [formal 5w ckpt-75 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7/summary.json)
- [formal 5w ckpt-925 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7/summary.json)
- [formal base vs 5w compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_theme15x7/README.zh-CN.md)
- [base summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [1k LoRA summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json)
- [5w ckpt-75 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10/summary.json)
- [5w ckpt-925 summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10/summary.json)
- [base vs 1k compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md)
- [base vs 5w compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_10/README.md)
