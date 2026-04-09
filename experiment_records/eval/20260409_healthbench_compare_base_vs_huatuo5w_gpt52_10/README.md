# HealthBench Base vs 5w LoRA Smoke Comparison

Comparison setting:

- benchmark: `HealthBench consensus`
- sample count: `10`
- judge model: `gpt-5.2`
- judge actual model: `gpt-5.2`

Compared runs:

- base: `20260409_healthbench_base_gpt52_full_10`
- 5w best checkpoint: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10`
- 5w later checkpoint: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10`

Headline result:

- base overall clipped mean: `0.4167`
- 5w checkpoint-75 overall clipped mean: `0.4167`
- 5w checkpoint-925 overall clipped mean: `0.4000`

Interpretation:

- `huatuo_5w` fixes most of the degradation seen in the `1k` LoRA run
- the best observed `5w` checkpoint matches base overall on this smoke set
- the later checkpoint is slightly worse, which supports early stopping and best-checkpoint selection
- the next improvement target should focus on `accuracy` and safety-sensitive slices such as `emergency_referrals`
