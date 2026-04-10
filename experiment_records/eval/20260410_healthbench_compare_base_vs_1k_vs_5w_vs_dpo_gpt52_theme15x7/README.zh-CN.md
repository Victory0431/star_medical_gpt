# HealthBench 正式对比：base vs 1k vs 5w vs DPO（Theme 15 x 7）

对比配置：

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- judge model：`gpt-5.2`
- judge actual model：`gpt-5.2`

参与对比的 run：

- base：`20260409_healthbench_base_gpt52_full_theme15x7`
- 1k：`20260410_healthbench_huatuo1k_gpt52_full_theme15x7`
- 5w 最优 checkpoint：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- 5w 较晚 checkpoint：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`
- DPO v1：`20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7`

核心结果：

- base overall clipped mean：`0.2206`
- 1k overall clipped mean：`0.2508`
- 5w checkpoint-75 overall clipped mean：`0.2889`
- 5w checkpoint-925 overall clipped mean：`0.2587`
- DPO checkpoint-100 overall clipped mean：`0.2111`

解读：

- 当前最优模型仍然是 `huatuo_5w checkpoint-75`
- `huatuo_1k` 已经可以正式超过 base，说明它是可靠的小规模 SFT baseline
- `checkpoint-925` 低于 `checkpoint-75`，再次证明最佳 checkpoint 选择非常关键
- `DPO v1` 低于最佳 SFT，甚至略低于 base，说明当前偏好数据和目标函数还不足以支撑正式收益
- 这不是坏消息，反而让项目更接近真实工业研发：不是每个阶段都会“自然涨分”，必须靠统一 benchmark 闭环去筛方案

建议主线：

- 主 baseline 保持 `huatuo_5w checkpoint-75`
- 快速试验和回归继续用 `huatuo_1k`
- DPO / 后续 ORPO / GRPO 继续沿用同一套 `theme 15 x 7` 正式评测
- 下一轮对齐重点改进 `communication`、`hedging`、`emergency_referrals` 三类能力
