# HealthBench 5w LoRA 正式对比点（Checkpoint-925, Theme 15 x 7）

这份记录保存了 `huatuo_5w` 同一条训练线中较晚 checkpoint 的正式分层评测结果。

运行信息：

- run name：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`
- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- evaluated model：`Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint：`checkpoint-925`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.2587`
- `axis:accuracy`：`0.1974`
- `axis:context_awareness`：`0.3333`
- `theme:context_seeking`：`0.1667`
- `theme:hedging`：`0.3778`

解读：

- `checkpoint-925` 仍然高于 base，但低于 `checkpoint-75`
- 它保留了部分 `instruction_following` 和 `context_seeking` 收益
- 但在 `accuracy`、`completeness`、`emergency_referrals` 上回退，说明继续训练并没有带来整体最优结果
