# 评测结果解读

英文原版见 [EVAL_RESULTS.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.md)。

这份文档记录了当前最值得在面试里讨论的一组 benchmark 快照。

## 当前 smoke 对比设置

对比配置：

- benchmark：`HealthBench consensus`
- sample count：`10`
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`
- judge API 形式：OpenAI 兼容 chat completions

参与对比的 run：

- base：`20260409_healthbench_base_gpt52_full_10`
- sft：`20260409_healthbench_huatuo1k_gpt52_full_10`

## 结论先看

| 模型 | overall clipped mean | 备注 |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.4167` | 当前 smoke baseline |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2833` | 当前 smoke SFT baseline |

初步结论：

- 在这组 `10` 条样本的 smoke test 上，`1k` LoRA 版本整体上落后于基座
- 这不是评测失败，反而是非常有价值的研发信号
- 它说明第一阶段的小规模 `SFT` 还没有在开放式医疗 benchmark 上带来稳健增益

## Axis 对比

| axis | base | 1k LoRA | delta |
| --- | ---: | ---: | ---: |
| `axis:accuracy` | `0.5000` | `0.1429` | `-0.3571` |
| `axis:communication_quality` | `0.6667` | `0.6667` | `0.0000` |
| `axis:context_awareness` | `0.2500` | `0.1250` | `-0.1250` |
| `axis:instruction_following` | `0.5000` | `1.0000` | `+0.5000` |

解读：

- `1k` LoRA 在 `instruction_following` 上确实有提升
- 但在 `accuracy` 上明显退步
- 在医学项目里很重要的 `context_awareness` 也没有改善

## Theme 对比

| theme | base | 1k LoRA | delta |
| --- | ---: | ---: | ---: |
| `theme:complex_responses` | `1.0000` | `0.5000` | `-0.5000` |
| `theme:context_seeking` | `0.0000` | `0.0000` | `0.0000` |
| `theme:emergency_referrals` | `0.5000` | `0.2500` | `-0.2500` |
| `theme:global_health` | `0.5000` | `0.2500` | `-0.2500` |
| `theme:health_data_tasks` | `0.5000` | `1.0000` | `+0.5000` |
| `theme:hedging` | `0.2222` | `0.1111` | `-0.1111` |

解读：

- `1k` LoRA 对 `health_data_tasks` 有一定帮助
- 但对 `hedging` 和 `context_seeking` 还没有形成改善
- 这意味着当前第一阶段 `SFT` 还不足以支撑“真实医疗交互质量显著提升”的结论

## 为什么这反而是亮点

一个很弱的项目故事是：

- “我微调了模型，分数提高了。”

更强的项目故事是：

- “我用开放式医疗 benchmark 验证第一阶段小规模 `SFT` 是否真的提升了目标行为。结果它只改善了局部切片，没有提升整体开放式质量，因此我把这当作下一阶段需要继续改进数据和对齐方式的证据。”

后一种表达明显更接近真实模型研发。

## 下一步建议

基于当前 smoke 对比，更合理的下一步是：

1. 把 `1k` 这一版定位为 pipeline 验证 checkpoint，而不是最终质量结论。
2. 对 `huatuo_5w` 跑完全相同的 benchmark。
3. 重点观察更大规模 `SFT` 是否能把 `accuracy` 和 `context_awareness` 拉回来。
4. 只有在这之后，再去讨论 `DPO / GRPO` 在 `hedging`、`context_seeking`、`communication_quality` 这类风格型切片上的收益。

## 结果来源

- [base summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [1k LoRA summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json)
- [base vs 1k compare README](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md)
