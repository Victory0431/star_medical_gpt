# HealthBench 基座 vs 5w LoRA 正式对比（Theme 15 x 7）

对比配置：

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- judge model：`gpt-5.2`
- judge actual model：`gpt-5.2`

参与对比的 run：

- base：`20260409_healthbench_base_gpt52_full_theme15x7`
- 5w 最优 checkpoint：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- 5w 较晚 checkpoint：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`

核心结果：

- base overall clipped mean：`0.2206`
- 5w checkpoint-75 overall clipped mean：`0.2889`
- 5w checkpoint-925 overall clipped mean：`0.2587`

解读：

- `huatuo_5w` 在正式分层样本上已经稳定超过 base，而不只是早期 smoke 偶然现象
- `checkpoint-75` 是当前最优 checkpoint，`checkpoint-925` 虽然仍高于 base，但已经出现回退
- `checkpoint-75` 的主要优势在 `accuracy`、`context_awareness`、`emergency_referrals` 和 `hedging`
- 这说明下一步应该把“最佳 checkpoint 选择/早停”写进标准流程，而不是简单继续延长 SFT 训练
