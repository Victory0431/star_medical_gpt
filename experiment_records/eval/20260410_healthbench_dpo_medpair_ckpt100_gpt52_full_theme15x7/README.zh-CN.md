# HealthBench DPO v1 正式结果（medical_pairwise, checkpoint-100）

这份记录保存了 `medical_pairwise` 数据上 DPO v1 的正式分层评测结果。

运行信息：

- run name：`20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7`
- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- evaluated model：`Qwen3-8B + DPO medical_pairwise`
- adapter checkpoint：`checkpoint-100`
- generator device：`cuda:0`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.2111`
- `axis:accuracy`：`0.1447`
- `axis:communication_quality`：`0.1778`
- `theme:emergency_referrals`：`0.2000`
- `theme:hedging`：`0.3111`

解读：

- 这次 DPO v1 已经完成了“训练链路 + 外部 judge + 正式 benchmark”的完整闭环，但结果没有超过最佳 SFT，也略低于 base
- 当前问题不在于 DPO 框架没跑通，而更像是 `medical_pairwise` 数据分布和正式医疗 benchmark 的目标不完全一致
- 下一步应该继续保留这条 DPO 线，但需要重做偏好数据设计，尤其围绕 `communication`、`hedging`、`emergency_referrals` 做更贴近评测目标的 reward/偏好信号
