# HealthBench 5w LoRA Smoke Baseline（较晚 Checkpoint）

这份记录保存了同一条 `huatuo_5w` 训练线中较晚 checkpoint 的 smoke 评测结果。

运行信息：

- run name：`20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10`
- benchmark：`HealthBench consensus`
- sample count：`10`
- evaluated model：`Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint：`checkpoint-925`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.4000`
- overall raw mean：`0.4000`

解读：

- 这一较晚 checkpoint 比 `checkpoint-75` 略差
- 这支持把“早停/最佳 checkpoint 选择”纳入标准训练流程，而不是简单训得更久
