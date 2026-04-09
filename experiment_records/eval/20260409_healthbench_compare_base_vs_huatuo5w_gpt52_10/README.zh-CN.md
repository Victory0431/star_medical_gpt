# HealthBench 基座 vs 5w LoRA Smoke 对比

对比配置：

- benchmark：`HealthBench consensus`
- sample count：`10`
- judge model：`gpt-5.2`
- judge actual model：`gpt-5.2`

参与对比的 run：

- base：`20260409_healthbench_base_gpt52_full_10`
- 5w 最优 checkpoint：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10`
- 5w 较晚 checkpoint：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10`

核心结果：

- base overall clipped mean：`0.4167`
- 5w checkpoint-75 overall clipped mean：`0.4167`
- 5w checkpoint-925 overall clipped mean：`0.4000`

解读：

- `huatuo_5w` 基本修复了 `1k` LoRA 的退化
- 当前观测到的最优 `5w` checkpoint 在这组 smoke set 上追平了 base
- 更晚 checkpoint 略有回退，说明应该把“早停/最佳 checkpoint 选择”纳入标准流程
- 下一步提升重点应放在 `accuracy` 和 `emergency_referrals` 这类安全敏感切片上
