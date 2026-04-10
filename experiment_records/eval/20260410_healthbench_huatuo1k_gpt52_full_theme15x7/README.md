# HealthBench 1k LoRA Formal Baseline (Theme 15 x 7)

This record stores the formal stratified evaluation for `huatuo_1k` LoRA.

Run details:

- run name: `20260410_healthbench_huatuo1k_gpt52_full_theme15x7`
- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- evaluated model: `Qwen3-8B + huatuo_1k LoRA`
- adapter checkpoint: `final_model`
- generator device: `cuda:1`
- judge model: `gpt-5.2`

Key results:

- overall clipped mean: `0.2508`
- `axis:accuracy`: `0.2039`
- `axis:communication_quality`: `0.2444`
- `theme:emergency_referrals`: `0.3000`
- `theme:hedging`: `0.3556`

Reading:

- `huatuo_1k` beats base on the formal slice, so it is a valid lightweight SFT baseline
- it is not as strong as the best `huatuo_5w` checkpoint, but it is already useful for smoke tests and fast iteration
