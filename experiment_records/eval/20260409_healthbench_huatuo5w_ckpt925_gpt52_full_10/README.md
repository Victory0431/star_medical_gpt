# HealthBench 5w LoRA Smoke Baseline (Later Checkpoint)

This record captures the `huatuo_5w` LoRA smoke run evaluated at a later checkpoint from the same training job.

Run metadata:

- run name: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10`
- benchmark: `HealthBench consensus`
- sample count: `10`
- evaluated model: `Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint: `checkpoint-925`
- judge API style: OpenAI-compatible chat completions
- requested judge model: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.4000`
- overall raw mean: `0.4000`

Interpretation:

- this later checkpoint is slightly worse than `checkpoint-75`
- the result supports early stopping or best-checkpoint selection rather than blindly training longer
