# HealthBench Base vs 1k LoRA Smoke Comparison

Comparison setting:

- benchmark: `HealthBench consensus`
- sample count: `10`
- judge model: `gpt-5.2`
- judge actual model: `gpt-5.2`

Headline result:

- base overall clipped mean: `0.4167`
- 1k LoRA overall clipped mean: `0.2833`

Interpretation:

- the `1k` SFT checkpoint does not yet beat the base model on this smoke open-ended benchmark
- it improves some narrow behavior slices such as `instruction_following`
- but loses ground on `accuracy` and `context_awareness`

This is a useful development signal, not a failure of the benchmark.
