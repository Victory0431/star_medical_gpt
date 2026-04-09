# HealthBench 1k LoRA Smoke Baseline

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/README.md)。

这份记录保存了 `huatuo_1k` LoRA checkpoint 第一次成功跑通的端到端 HealthBench judge 结果。

运行信息：

- run name：`20260409_healthbench_huatuo1k_gpt52_full_10`
- benchmark：`HealthBench consensus`
- sample count：`10`
- evaluated model：`Qwen3-8B + huatuo_1k LoRA`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.2833`
- overall raw mean：`0.2833`

重要说明：

- 这是用于 pipeline 验证的 smoke baseline
- 不能直接当成最终完整 benchmark 结论
