# HealthBench 5w LoRA Formal Comparison Point (Checkpoint-925, Theme 15 x 7)

This record stores the formal stratified evaluation result for a later checkpoint from the same `huatuo_5w` training run.

Run details:

- run name: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`
- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- evaluated model: `Qwen3-8B + huatuo_5w LoRA`
- adapter checkpoint: `checkpoint-925`
- judge API style: OpenAI-compatible chat completions
- requested judge model: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.2587`
- `axis:accuracy`: `0.1974`
- `axis:context_awareness`: `0.3333`
- `theme:context_seeking`: `0.1667`
- `theme:hedging`: `0.3778`

Reading:

- `checkpoint-925` still beats the base model, but it underperforms `checkpoint-75`
- it keeps some gains on `instruction_following` and `context_seeking`
- but it regresses on `accuracy`, `completeness`, and `emergency_referrals`, which reinforces the need for better checkpoint selection
