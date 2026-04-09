# HealthBench 5w LoRA Smoke Baseline (Best Checkpoint)

This record captures the `huatuo_5w` LoRA smoke run evaluated at the best validation checkpoint from the current training job.

Run metadata:

- run name: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10`
- benchmark: `HealthBench consensus`
- sample count: `10`
- evaluated model: `Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint: `checkpoint-75`
- judge API style: OpenAI-compatible chat completions
- requested judge model: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.4167`
- overall raw mean: `0.4167`

Interpretation:

- this checkpoint clearly improves over the `1k` LoRA smoke run
- it matches the base model overall on the current `10`-example smoke set
- it should be treated as the current main SFT baseline from the `5w` run
