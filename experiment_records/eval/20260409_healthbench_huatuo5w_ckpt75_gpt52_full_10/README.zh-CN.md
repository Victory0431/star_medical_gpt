# HealthBench 5w LoRA Smoke Baseline（最优 Checkpoint）

这份记录保存了 `huatuo_5w` LoRA 在当前训练任务中“验证集最优 checkpoint”上的 smoke 评测结果。

运行信息：

- run name：`20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10`
- benchmark：`HealthBench consensus`
- sample count：`10`
- evaluated model：`Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint：`checkpoint-75`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.4167`
- overall raw mean：`0.4167`

解读：

- 这一 checkpoint 相比 `1k` LoRA 有明显恢复
- 在当前 `10` 条 smoke set 上，它已经追平 base
- 这一点更适合被当作当前 `5w` 训练线的主 SFT baseline
