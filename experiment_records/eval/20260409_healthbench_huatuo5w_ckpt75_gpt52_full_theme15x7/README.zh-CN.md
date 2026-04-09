# HealthBench 5w LoRA 正式基线（Checkpoint-75, Theme 15 x 7）

这份记录保存了 `huatuo_5w` LoRA 在当前训练任务中“验证集最优 checkpoint”上的正式分层评测结果。

运行信息：

- run name：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- evaluated model：`Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint：`checkpoint-75`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.2889`
- `axis:accuracy`：`0.2368`
- `axis:context_awareness`：`0.3704`
- `theme:emergency_referrals`：`0.4333`
- `theme:hedging`：`0.4222`

解读：

- 在这组 `105` 条正式小规模分层样本上，`checkpoint-75` 明显优于 base
- 提升最明显的方向是 `context_awareness`、`hedging`、`emergency_referrals`
- 这说明 `5w` SFT 不只是“记住更多医学内容”，还改善了部分开放式医疗交互行为
