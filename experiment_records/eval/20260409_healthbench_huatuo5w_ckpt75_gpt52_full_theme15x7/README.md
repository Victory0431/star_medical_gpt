# HealthBench 5w LoRA Formal Baseline (Checkpoint-75, Theme 15 x 7)

This record stores the formal stratified evaluation result for the best validation checkpoint of the current `huatuo_5w` LoRA run.

Run details:

- run name: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- evaluated model: `Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint: `checkpoint-75`
- judge API style: OpenAI-compatible chat completions
- requested judge model: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.2889`
- `axis:accuracy`: `0.2368`
- `axis:context_awareness`: `0.3704`
- `theme:emergency_referrals`: `0.4333`
- `theme:hedging`: `0.4222`

Reading:

- on this `105`-example formal stratified slice, `checkpoint-75` is clearly better than the base model
- the most meaningful gains appear on `context_awareness`, `hedging`, and `emergency_referrals`
- this suggests the `5w` SFT run improved not only medical content coverage, but also parts of open-ended medical interaction behavior
