# HealthBench 1k LoRA Smoke Baseline

This record captures the first successful end-to-end HealthBench judge run for the `huatuo_1k` LoRA checkpoint.

Run metadata:

- run name: `20260409_healthbench_huatuo1k_gpt52_full_10`
- benchmark: `HealthBench consensus`
- sample count: `10`
- evaluated model: `Qwen3-8B + huatuo_1k LoRA`
- judge API style: OpenAI-compatible chat completions
- requested judge model: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.2833`
- overall raw mean: `0.2833`

Important caveat:

- this is a smoke baseline for pipeline validation
- it should not be presented as a final full-benchmark number
