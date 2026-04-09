# HealthBench Base vs 5w LoRA Formal Comparison (Theme 15 x 7)

Comparison setup:

- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- judge model: `gpt-5.2`
- judge actual model: `gpt-5.2`

Compared runs:

- base: `20260409_healthbench_base_gpt52_full_theme15x7`
- 5w best checkpoint: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- 5w later checkpoint: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`

Core result:

- base overall clipped mean: `0.2206`
- 5w checkpoint-75 overall clipped mean: `0.2889`
- 5w checkpoint-925 overall clipped mean: `0.2587`

Reading:

- `huatuo_5w` now beats the base model on a formal stratified slice, not only on an early smoke test
- `checkpoint-75` is the best checkpoint, while `checkpoint-925` still beats base but has already regressed
- the strongest `checkpoint-75` gains are on `accuracy`, `context_awareness`, `emergency_referrals`, and `hedging`
- this is strong evidence that best-checkpoint selection and early stopping should be part of the default workflow
