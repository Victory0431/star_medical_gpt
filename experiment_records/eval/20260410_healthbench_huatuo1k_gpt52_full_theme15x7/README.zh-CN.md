# HealthBench 1k LoRA 正式基线（Theme 15 x 7）

这份记录保存了 `huatuo_1k` LoRA 在正式分层评测中的结果。

运行信息：

- run name：`20260410_healthbench_huatuo1k_gpt52_full_theme15x7`
- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- evaluated model：`Qwen3-8B + huatuo_1k LoRA`
- adapter checkpoint：`final_model`
- generator device：`cuda:1`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.2508`
- `axis:accuracy`：`0.2039`
- `axis:communication_quality`：`0.2444`
- `theme:emergency_referrals`：`0.3000`
- `theme:hedging`：`0.3556`

解读：

- `huatuo_1k` 在正式 `105` 条分层样本上高于 base，说明它已经是有效的小规模 SFT baseline
- 它没有达到 `huatuo_5w checkpoint-75` 的水平，但已经足够承担 smoke test、流程验证、快速回归测试等角色
- 它在 `communication`、`emergency_referrals`、`hedging` 上比 base 更好，说明小数据 SFT 也能带来稳定收益
