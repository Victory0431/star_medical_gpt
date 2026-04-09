# HealthBench 基座 vs 1k LoRA Smoke 对比

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md)。

对比配置：

- benchmark：`HealthBench consensus`
- sample count：`10`
- judge model：`gpt-5.2`
- judge actual model：`gpt-5.2`

核心结果：

- base overall clipped mean：`0.4167`
- 1k LoRA overall clipped mean：`0.2833`

解读：

- 当前 `1k` `SFT` checkpoint 还没有在这组开放式 smoke benchmark 上超过基座
- 它改善了 `instruction_following` 这类局部行为
- 但在 `accuracy` 和 `context_awareness` 上有退步

这个结果是有价值的研发信号，不是 benchmark 失败。
